"""CrystalCall / DenoisePTC.

Lossless speculative programmatic tool calling from uncommitted DiffusionGemma
denoising states, executed through a pinned OMP-Lite runtime.

The final model-generated program is always authoritative.  CrystalCall may
change *when* eligible external work starts; it never changes the final action
semantics.
"""

from __future__ import annotations

__version__ = "0.1.0"

SCHEMA_VERSIONS = {
    "trace": "crystalcall-trace/1",
    "revision": "crystalcall-revision/2",
    "manifest": "crystalcall-manifest/1",
    "results": "crystalcall-results/1",
}

__all__ = ["__version__", "SCHEMA_VERSIONS"]
