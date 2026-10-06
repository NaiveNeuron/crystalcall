// CrystalCall: deterministic GPU dense-retrieval kernel for `embed_search`.
//
// This is the one genuinely accelerator-backed tool in the CrystalCall tool
// surface.  It exists so the experiment can answer a question a CPU kernel
// cannot: does concurrent GPU work interfere with DiffusionGemma denoising, and
// does the resource gate correctly abstain when it would?
//
// Determinism is a hard requirement: a speculative execution and the later
// authoritative execution of the same canonical call must return bit-identical
// scores.  Therefore:
//
//   * exactly one CUDA thread per matrix row;
//   * each thread accumulates its D products in strict ascending index order
//     into a single `float` register;
//   * no atomics, no split-K, no cub/thrust reductions, no tensor cores, no
//     warp shuffles, no `-ffast-math`, and `--fmad=false` so the compiler may
//     not contract `a*b+c` into an FMA (which would change rounding);
//   * top-k selection happens on the HOST over the returned score vector with
//     a total order on (-score, row index), so ties can never reorder.
//
// Build (recorded verbatim in configs/tools.yml):
//   nvcc -O2 -arch=sm_121 --fmad=false -o embed_search embed_search.cu
//
// Usage:
//   embed_search <matrix.f32> <rows> <dim> <query.f32> <top_k>
//
// `matrix.f32` is a raw row-major float32 blob of shape [rows, dim] taken
// straight from the frozen embedding snapshot; `query.f32` is a raw float32
// blob of `dim` values.  Output on stdout is one `"<row_index>\t<score>"` line
// per hit, most similar first, with the score printed at full float precision.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime.h>

#define CUDA_CHECK(expr)                                                              \
    do {                                                                              \
        cudaError_t _err = (expr);                                                     \
        if (_err != cudaSuccess) {                                                     \
            std::fprintf(stderr, "cuda error %s at %s:%d: %s\n", #expr, __FILE__,      \
                         __LINE__, cudaGetErrorString(_err));                          \
            return 2;                                                                  \
        }                                                                              \
    } while (0)

// One thread per row.  Strict in-order accumulation; nothing here depends on
// block size, launch configuration, or the number of resident warps.
__global__ void dot_rows(const float* __restrict__ matrix,
                         const float* __restrict__ query,
                         float* __restrict__ scores,
                         long long rows,
                         int dim) {
    long long row = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    const float* r = matrix + row * static_cast<long long>(dim);
    float acc = 0.0f;
    for (int k = 0; k < dim; ++k) {
        acc = acc + r[k] * query[k];
    }
    scores[row] = acc;
}

// Bounded retry around cudaMalloc.  Only the allocation is retried; the kernel
// itself, and therefore the returned scores, are unaffected.
static bool alloc_retry(void** out, size_t bytes) {
    cudaError_t err = cudaSuccess;
    for (int attempt = 0; attempt < 8; ++attempt) {
        err = cudaMalloc(out, bytes);
        if (err == cudaSuccess) {
            return true;
        }
        cudaGetLastError();  // clear the sticky error before retrying
        ::usleep(150000);
    }
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaMemGetInfo(&free_bytes, &total_bytes);
    std::fprintf(stderr,
                 "cudaMalloc(%zu bytes) failed after 8 attempts: %s "
                 "(device free %.3f GB / total %.3f GB)\n",
                 bytes, cudaGetErrorString(err), free_bytes / 1e9, total_bytes / 1e9);
    return false;
}

static const float* map_f32(const char* path, size_t expect_bytes, void** handle, size_t* mapped) {
    int fd = ::open(path, O_RDONLY);
    if (fd < 0) {
        std::fprintf(stderr, "cannot open %s\n", path);
        return nullptr;
    }
    struct stat st;
    if (::fstat(fd, &st) != 0) {
        std::fprintf(stderr, "cannot stat %s\n", path);
        ::close(fd);
        return nullptr;
    }
    if (static_cast<size_t>(st.st_size) != expect_bytes) {
        std::fprintf(stderr, "%s: expected %zu bytes, found %lld\n", path, expect_bytes,
                     static_cast<long long>(st.st_size));
        ::close(fd);
        return nullptr;
    }
    void* p = ::mmap(nullptr, expect_bytes, PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) {
        std::fprintf(stderr, "cannot mmap %s\n", path);
        return nullptr;
    }
    *handle = p;
    *mapped = expect_bytes;
    return static_cast<const float*>(p);
}

int main(int argc, char** argv) {
    if (argc != 6) {
        std::fprintf(stderr,
                     "usage: %s <matrix.f32> <rows> <dim> <query.f32> <top_k>\n",
                     argv[0]);
        return 64;
    }
    const char* matrix_path = argv[1];
    const long long rows = std::atoll(argv[2]);
    const int dim = std::atoi(argv[3]);
    const char* query_path = argv[4];
    long long top_k = std::atoll(argv[5]);

    if (rows <= 0 || dim <= 0 || top_k <= 0) {
        std::fprintf(stderr, "rows, dim and top_k must all be positive\n");
        return 64;
    }
    if (top_k > rows) {
        top_k = rows;
    }

    void* m_handle = nullptr;
    size_t m_bytes = 0;
    const size_t matrix_bytes = static_cast<size_t>(rows) * static_cast<size_t>(dim) * sizeof(float);
    const float* h_matrix = map_f32(matrix_path, matrix_bytes, &m_handle, &m_bytes);
    if (h_matrix == nullptr) {
        return 3;
    }

    void* q_handle = nullptr;
    size_t q_bytes = 0;
    const float* h_query =
        map_f32(query_path, static_cast<size_t>(dim) * sizeof(float), &q_handle, &q_bytes);
    if (h_query == nullptr) {
        ::munmap(m_handle, m_bytes);
        return 3;
    }

    float* d_matrix = nullptr;
    float* d_query = nullptr;
    float* d_scores = nullptr;
    // GB10 shares one physical memory pool between host and device, so a
    // concurrent vLLM server plus a full page cache can make a device
    // allocation transiently fail.  Retry a bounded number of times (the kernel
    // launch is the contended resource this experiment is *about*), then report
    // the actual free/total so a real exhaustion is unambiguous.
    if (!alloc_retry(reinterpret_cast<void**>(&d_matrix), matrix_bytes)) {
        return 2;
    }
    if (!alloc_retry(reinterpret_cast<void**>(&d_query),
                     static_cast<size_t>(dim) * sizeof(float))) {
        return 2;
    }
    if (!alloc_retry(reinterpret_cast<void**>(&d_scores),
                     static_cast<size_t>(rows) * sizeof(float))) {
        return 2;
    }

    CUDA_CHECK(cudaMemcpy(d_matrix, h_matrix, matrix_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_query, h_query, static_cast<size_t>(dim) * sizeof(float),
                          cudaMemcpyHostToDevice));

    const int threads = 256;
    const long long blocks = (rows + threads - 1) / threads;
    dot_rows<<<static_cast<unsigned int>(blocks), threads>>>(d_matrix, d_query, d_scores, rows, dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<float> scores(static_cast<size_t>(rows));
    CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, static_cast<size_t>(rows) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_matrix);
    cudaFree(d_query);
    cudaFree(d_scores);
    ::munmap(m_handle, m_bytes);
    ::munmap(q_handle, q_bytes);

    // Host-side top-k: a total order on (-score, index) leaves no room for a
    // tie to be broken differently between two runs.
    std::vector<long long> idx(static_cast<size_t>(rows));
    std::iota(idx.begin(), idx.end(), 0LL);
    const size_t k = static_cast<size_t>(top_k);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&scores](long long a, long long b) {
                          const float sa = scores[static_cast<size_t>(a)];
                          const float sb = scores[static_cast<size_t>(b)];
                          if (sa != sb) {
                              return sa > sb;
                          }
                          return a < b;
                      });

    for (size_t i = 0; i < k; ++i) {
        std::printf("%lld\t%.9g\n", idx[i], static_cast<double>(scores[static_cast<size_t>(idx[i])]));
    }
    return 0;
}
