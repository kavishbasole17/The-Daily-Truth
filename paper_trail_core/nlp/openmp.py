"""
Let torch and the boosted-tree models share a process despite two OpenMP runtimes.

torch's macOS wheel bundles its own libomp (torch/lib/libomp.dylib), while
XGBoost and LightGBM link the system one (Homebrew's /opt/homebrew/opt/libomp).
Both end up in the API and pipeline process, and that fails in two ways
depending on which runtime starts first:

1. **torch's first → segfault.** The first XGBoost or LightGBM call afterwards
   crashes. The pipeline hits exactly this order: evidence retrieval loads the
   embedder before the stylistic ensemble loads its boosted-tree base models.
2. **System runtime first, already running threads → abort.** When torch's copy
   then initialises, libomp's duplicate-runtime check calls abort()
   (`__kmp_register_library_startup` -> `__kmp_fatal`). The test suite hits this:
   the API tests load the ensemble, later tests load the embedder.

`load_before_torch()` fixes both. It imports the system-runtime users first,
which removes case 1, and sets `KMP_DUPLICATE_LIB_OK=TRUE`, libomp's switch
for allowing a second runtime, which removes case 2. LLVM documents that switch
as unsupported because two runtimes could in principle interfere. Measured here,
they do not: embeddings, NLI probabilities and all 20 ensemble model outputs are
identical to six decimal places with both runtimes loaded and with each alone.

Every torch import in this package calls `load_before_torch()` first.

Limit: this holds with torch on one thread (TORCH_NUM_THREADS, default 1 in
embeddings.py and nli.py). With several torch threads the two runtimes' thread
pools still collide and the process segfaults.

The real fix is one runtime, and it needs no code: start the process with
torch's libomp resolved to the system copy, e.g.

    DYLD_LIBRARY_PATH=/opt/homebrew/opt/libomp/lib uvicorn api.main:app

dyld then loads a single libomp for everyone (checked with vmmap), torch can run
multi-threaded, and the pipeline is faster (18s vs 29s for a two-claim article
at 8 threads vs 1). It has to be set before Python starts, which is why it is
documented here rather than done here.
"""
from __future__ import annotations

import importlib
import logging
import os

log = logging.getLogger(__name__)

# Libraries that link the system libomp. Importing one loads it.
_OPENMP_USERS = ("xgboost", "lightgbm")

_done = False


def load_before_torch() -> None:
    """Make it safe to import torch next. Idempotent and cheap after the first call."""
    global _done
    if _done:
        return
    _done = True
    # Read by torch's libomp when it initialises, so it must be set before torch
    # is imported. setdefault: an explicit setting by the operator wins.
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    for name in _OPENMP_USERS:
        try:
            importlib.import_module(name)
        except ImportError:
            pass                    # not installed: nothing to conflict with
        except Exception as exc:
            log.warning("openmp: could not preload %s (%s)", name, exc)
