"""Environment inspection and version pinning.

``crystalcall doctor`` writes a single machine-readable record describing the
machine, the driver/CUDA stack, the pinned software revisions, the model
weights and their hashes, the sampler settings, cache locations, CPU affinity,
storage and the available power telemetry.  Every experiment record carries the
digest of this document so environment drift is detectable after the fact.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

__all__ = ["collect", "doctor_digest", "run_pins", "PinSpec"]

_TIMEOUT = 60


def _run(cmd: Sequence[str], *, timeout: int = _TIMEOUT) -> dict[str, Any]:
    exe = shutil.which(cmd[0])
    if exe is None:
        return {"cmd": list(cmd), "status": "missing"}
    try:
        p = subprocess.run(  # noqa: S603
            [exe, *cmd[1:]],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {"cmd": list(cmd), "status": "timeout"}
    return {
        "cmd": list(cmd),
        "status": "ok" if p.returncode == 0 else f"exit{p.returncode}",
        "stdout": p.stdout.strip(),
        "stderr": p.stderr.strip()[-2000:],
    }


def _text(res: Mapping[str, Any]) -> str:
    return str(res.get("stdout", "")) if res.get("status") == "ok" else ""


def _sha256_file(path: Path, *, limit: int | None = None) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if limit is not None and read >= limit:
                break
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# hardware / OS
# --------------------------------------------------------------------------- #


def _hardware() -> dict[str, Any]:
    out: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
        "boot_time_ns_offset": time.time_ns() - time.monotonic_ns(),
    }
    try:
        out["affinity"] = sorted(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        out["affinity"] = None
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        models: dict[str, int] = {}
        parts: dict[str, int] = {}
        for line in cpuinfo.read_text(errors="replace").splitlines():
            if line.startswith("model name"):
                v = line.split(":", 1)[1].strip()
                models[v] = models.get(v, 0) + 1
            elif line.startswith("CPU part"):
                v = line.split(":", 1)[1].strip()
                parts[v] = parts.get(v, 0) + 1
        out["cpu_models"] = models
        # ARM CPU part ids: 0xd87 = Cortex-X925, 0xd8a = Cortex-A725 on GB10.
        out["cpu_parts"] = parts
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        mem: dict[str, str] = {}
        for line in meminfo.read_text(errors="replace").splitlines()[:8]:
            k, _, v = line.partition(":")
            mem[k.strip()] = v.strip()
        out["meminfo"] = mem
    return out


def _gpu() -> dict[str, Any]:
    q = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pci.bus_id,memory.total,persistence_mode,compute_mode,clocks.max.sm,power.max_limit",
            "--format=csv,noheader",
        ]
    )
    power = _run(["nvidia-smi", "-q", "-d", "POWER"])
    power_text = _text(power)
    telemetry: dict[str, Any] = {
        "nvidia_smi_power_query": power.get("status"),
        "power_readings_supported": "Power Draw" in power_text
        and "N/A" not in power_text.split("Power Draw", 1)[-1][:40],
    }
    dcgm = _run(["dcgmi", "discovery", "-l"], timeout=20)
    telemetry["dcgmi"] = dcgm.get("status")
    # GB10 is an integrated Blackwell part: several nvidia-smi fields are
    # documented as "Not Supported"; record exactly which so the energy claim
    # can be scoped honestly.
    hwmon = sorted(str(p) for p in Path("/sys/class/hwmon").glob("hwmon*/name"))
    telemetry["hwmon"] = {
        p: Path(p).read_text(errors="replace").strip() for p in hwmon[:16]
    }
    return {
        "query_gpu": q,
        "smi_version": _text(_run(["nvidia-smi", "--version"]))[:400],
        "nvcc": _text(_run(["nvcc", "--version"]))[-200:],
        "telemetry": telemetry,
    }


def _storage(paths: Sequence[Path]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for p in paths:
        try:
            st = shutil.disk_usage(p)
            out[str(p)] = {
                "total_gb": round(st.total / 1e9, 2),
                "free_gb": round(st.free / 1e9, 2),
            }
        except OSError as exc:
            out[str(p)] = {"error": str(exc)}
    out["block_devices"] = _text(_run(["lsblk", "-dno", "NAME,ROTA,SIZE,MODEL"]))
    return out


# --------------------------------------------------------------------------- #
# pinned revisions
# --------------------------------------------------------------------------- #


class PinSpec:
    """Where each pinned component's revision is read from."""

    VENDOR_REPOS = {
        "bfcl": "experiments/vendor/bfcl",
        "toolsandbox": "experiments/vendor/toolsandbox",
        "sptc": "experiments/vendor/sptc",
    }


def _git_rev(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"status": "absent", "path": str(path)}
    head = _run(["git", "-C", str(path), "rev-parse", "HEAD"])
    remote = _run(["git", "-C", str(path), "config", "--get", "remote.origin.url"])
    dirty = _run(["git", "-C", str(path), "status", "--porcelain"])
    return {
        "status": "ok" if head.get("status") == "ok" else "error",
        "path": str(path),
        "commit": _text(head),
        "remote": _text(remote),
        "dirty": bool(_text(dirty)),
    }


class PinSpec:
    """Where each pinned component's revision is read from."""

    VENDOR_REPOS = {
        "bfcl": "experiments/vendor/bfcl",
        "toolsandbox": "experiments/vendor/toolsandbox",
        "sptc": "experiments/vendor/sptc",
    }
    #: Written by scripts/build_manifests.py when a repository is fetched as a
    #: pinned-SHA tarball rather than a git clone (for hosts where
    #: git-upload-pack is blocked).
    PROVENANCE = "experiments/manifests/vendor_sources.json"


def _vendor_pins(root: Path) -> dict[str, Any]:
    out = {name: _git_rev(root / rel) for name, rel in PinSpec.VENDOR_REPOS.items()}
    prov_path = root / PinSpec.PROVENANCE
    if prov_path.exists():
        try:
            prov = json.loads(prov_path.read_text())
        except json.JSONDecodeError:
            prov = {}
        raw = prov.get("sources", prov) if isinstance(prov, Mapping) else {}
        records: list[Mapping[str, Any]] = []
        if isinstance(raw, Mapping):
            for name, rec in raw.items():
                if isinstance(rec, Mapping):
                    records.append({"key": name, **rec})
        elif isinstance(raw, list):
            records = [r for r in raw if isinstance(r, Mapping)]
        for rec in records:
            name = str(rec.get("key") or rec.get("name") or "")
            commit = rec.get("commit") or rec.get("sha") or rec.get("rev")
            if not name or not commit:
                continue
            out[name] = {
                "status": "ok",
                "path": str(root / PinSpec.VENDOR_REPOS.get(name, f"experiments/vendor/{name}")),
                "commit": str(commit),
                "remote": rec.get("repo") or rec.get("archive_url") or "",
                "fetch": rec.get("fetch_command") or rec.get("git_clone_command") or "",
                "archive_sha256": rec.get("archive_sha256"),
                "source": "vendor_sources.json",
            }
        out["_provenance_sha256"] = _sha256_file(prov_path)
    return out


def _hf_model_pin(repo_id: str, cache: Path) -> dict[str, Any]:
    folder = cache / ("models--" + repo_id.replace("/", "--"))
    if not folder.exists():
        return {"repo_id": repo_id, "status": "absent", "expected_path": str(folder)}
    refs = folder / "refs"
    snapshots = (
        sorted((folder / "snapshots").glob("*")) if (folder / "snapshots").exists() else []
    )
    revisions = [s.name for s in snapshots]
    files: dict[str, Any] = {}
    total = 0
    if snapshots:
        snap = snapshots[-1]
        for f in sorted(snap.iterdir()):
            try:
                real = f.resolve()
                size = real.stat().st_size if real.is_file() else 0
            except OSError:
                size = 0
            total += size
            entry: dict[str, Any] = {"size": size}
            # Hash only the small metadata files in full; weight shards are
            # identified by their content-addressed blob name.
            if f.suffix in (".json", ".jinja", ".txt") and size < 8 << 20:
                entry["sha256"] = _sha256_file(real)
            elif real.is_file():
                entry["blob"] = real.name
            files[f.name] = entry
    return {
        "repo_id": repo_id,
        "status": "ok",
        "path": str(folder),
        "revisions": revisions,
        "refs": {p.name: p.read_text().strip() for p in refs.glob("*")} if refs.exists() else {},
        "total_bytes": total,
        "files": files,
    }


def _docker_pin(image: str) -> dict[str, Any]:
    res = _run(["docker", "image", "inspect", image, "--format", "{{json .}}"], timeout=90)
    if res.get("status") != "ok":
        return {"image": image, "status": res.get("status")}
    try:
        info = json.loads(_text(res))
    except json.JSONDecodeError:
        return {"image": image, "status": "unparseable"}
    return {
        "image": image,
        "status": "ok",
        "id": info.get("Id"),
        "repo_digests": info.get("RepoDigests"),
        "created": info.get("Created"),
        "architecture": info.get("Architecture"),
        "os": info.get("Os"),
    }


def _vllm_pin(image: str) -> dict[str, Any]:
    res = _run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            "python3",
            image,
            "-c",
            "import vllm,torch,json;print(json.dumps({'vllm':vllm.__version__,"
            "'torch':torch.__version__,'cuda':torch.version.cuda,'file':vllm.__file__}))",
        ],
        timeout=300,
    )
    if res.get("status") != "ok":
        return {"status": res.get("status"), "stderr": res.get("stderr", "")[-500:]}
    try:
        return {"status": "ok", **json.loads(_text(res))}
    except json.JSONDecodeError:
        return {"status": "unparseable", "stdout": _text(res)[:500]}


def run_pins(root: Path, *, image: str, models: Sequence[str], probe_image: bool = False) -> dict[str, Any]:
    cache = Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    pins: dict[str, Any] = {
        "omp": {
            "version": _text(_run(["omp", "--version"])),
            "binary": shutil.which("omp"),
            "binary_sha256": _sha256_file(Path(shutil.which("omp") or "/nonexistent")),
        },
        "uv": _text(_run(["uv", "--version"])),
        "docker": _text(_run(["docker", "--version"])),
        "ripgrep": _text(_run(["rg", "--version"]))[:80],
        "sqlite3": _text(_run(["sqlite3", "--version"]))[:80],
        "image": _docker_pin(image),
        "crystalcall": _git_rev(root),
        "vendor": _vendor_pins(root),
        "models": {m: _hf_model_pin(m, cache) for m in models},
        "hf_cache": str(cache),
    }
    if probe_image:
        pins["vllm"] = _vllm_pin(image)
    return pins


# --------------------------------------------------------------------------- #
# OMP-Lite verification
# --------------------------------------------------------------------------- #


def omp_lite_state(
    *,
    config: Path | None,
    system_prompt: Path | None,
    profile: str,
    model: str,
    tools: Sequence[str],
    host_tools: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Start OMP-Lite over RPC and hash its system prompt and tool inventory.

    This is the mechanised proof of "one fixed system prompt, one fixed tool
    representation": the digests must be identical on every run.
    """
    from .omp_bridge import OmpLiteProfile, OmpRpcClient, OmpRpcError

    prof = OmpLiteProfile(
        profile=profile,
        config=config,
        system_prompt=system_prompt,
        model=model,
        tools=tuple(tools),
    )
    client = OmpRpcClient(prof)
    try:
        ready = client.start(timeout=120)
    except (OmpRpcError, OSError) as exc:
        return {"status": "error", "error": str(exc), "argv": prof.argv()}
    try:
        if host_tools:
            client.request("set_host_tools", tools=list(host_tools))
        state = client.get_state()
    finally:
        client.close()
    sp = state.get("systemPrompt") or []
    dump = state.get("dumpTools") or []
    sp_blob = json.dumps(sp, sort_keys=True, ensure_ascii=False)
    tools_blob = json.dumps(
        sorted(
            ({"name": t.get("name"), "parameters": t.get("parameters")} for t in dump),
            key=lambda t: str(t["name"]),
        ),
        sort_keys=True,
        ensure_ascii=False,
    )
    return {
        "status": "ok",
        "argv": prof.argv(),
        "profile_fingerprint": prof.fingerprint(),
        "ready": ready,
        "protocol_version": client.protocol_version,
        "system_prompt_blocks": len(sp),
        "system_prompt_chars": sum(len(str(b)) for b in sp),
        "system_prompt_sha256": hashlib.sha256(sp_blob.encode()).hexdigest(),
        "tool_names": [t.get("name") for t in dump],
        "tool_representation_sha256": hashlib.sha256(tools_blob.encode()).hexdigest(),
        "model": (state.get("model") or {}).get("id"),
        "thinking_level": state.get("thinkingLevel"),
        "auto_compaction_enabled": state.get("autoCompactionEnabled"),
        "context_window": (state.get("model") or {}).get("contextWindow"),
    }


def omp_settings_snapshot() -> dict[str, Any]:
    res = _run(["omp", "config", "list"])
    blob = _text(res)
    return {
        "status": res.get("status"),
        "lines": len(blob.splitlines()),
        "sha256": hashlib.sha256(blob.encode()).hexdigest() if blob else None,
    }


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #


DEFAULT_MODELS = (
    "google/diffusiongemma-26B-A4B-it",
    "google/gemma-4-26B-A4B-it",
)


def collect(
    root: Path,
    *,
    image: str = "vllm/vllm-openai:gemma",
    models: Sequence[str] = DEFAULT_MODELS,
    probe_image: bool = False,
    probe_omp: bool = True,
    sampler_config: Path | None = None,
) -> dict[str, Any]:
    root = Path(root)
    doc: dict[str, Any] = {
        "schema_version": "crystalcall-doctor/1",
        "generated_unix_ns": time.time_ns(),
        "hardware": _hardware(),
        "gpu": _gpu(),
        "storage": _storage([root, Path.home() / ".cache/huggingface", Path("/")]),
        "pins": run_pins(root, image=image, models=models, probe_image=probe_image),
        "omp_settings": omp_settings_snapshot(),
        "env": {
            k: os.environ[k]
            for k in sorted(os.environ)
            if k.startswith(("CRYSTALCALL_", "HF_", "CUDA_", "PI_", "OMP_", "TORCH_"))
        },
    }
    cfg = sampler_config or (root / "configs/diffusiongemma.yaml")
    if cfg.exists():
        import yaml

        doc["sampler"] = yaml.safe_load(cfg.read_text())
        doc["sampler_config_sha256"] = _sha256_file(cfg)
    else:
        doc["sampler"] = {"status": "absent", "expected": str(cfg)}

    for name in ("configs/omp-lite.yml", "configs/models.yml", "configs/tools.yml"):
        p = root / name
        doc.setdefault("config_hashes", {})[name] = _sha256_file(p)

    if probe_omp:
        doc["omp_lite"] = omp_lite_state(
            config=(root / "configs/omp-lite.yml") if (root / "configs/omp-lite.yml").exists() else None,
            system_prompt=(root / "configs/system_prompt.md")
            if (root / "configs/system_prompt.md").exists()
            else None,
            profile="crystalcall",
            model="vllm/diffusiongemma-26b-a4b-it",
            tools=("eval",),
        )
    doc["digest"] = doctor_digest(doc)
    return doc


def doctor_digest(doc: Mapping[str, Any]) -> str:
    """Digest of the *stable* part of the document (excludes timestamps)."""
    stable = {
        k: v
        for k, v in doc.items()
        if k not in ("generated_unix_ns", "digest", "storage", "gpu", "hardware")
    }
    stable["hardware"] = {
        k: v
        for k, v in (doc.get("hardware") or {}).items()
        if k not in ("boot_time_ns_offset",)
    }
    stable["gpu"] = {"query_gpu": (doc.get("gpu") or {}).get("query_gpu")}
    return hashlib.sha256(
        json.dumps(stable, sort_keys=True, default=str, separators=(",", ":")).encode()
    ).hexdigest()
