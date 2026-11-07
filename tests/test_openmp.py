"""
OpenMP load-order regression test.

torch's bundled libomp loaded before XGBoost/LightGBM's system libomp segfaults
the first boosted-tree call (see paper_trail_core/nlp/openmp.py). A segfault
kills the interpreter, so each check runs in a fresh subprocess and asserts on
its exit code. Marked slow: it loads the real embedding and NLI models.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_BOOSTED_TREES_AFTER = """
import numpy as np, lightgbm, xgboost
X, y = np.random.rand(40, 3), [0, 1] * 20
lightgbm.LGBMClassifier(n_estimators=2, verbose=-1).fit(X, y).predict(X[:1])
xgboost.XGBClassifier(n_estimators=2).fit(X, y).predict(X[:1])
print("ok")
"""


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], cwd=_ROOT,
                          capture_output=True, text=True, timeout=600)


@pytest.mark.slow
@pytest.mark.parametrize("load_torch_model", [
    "from paper_trail_core.nlp.embeddings import embed; embed(['hi'])",
    "from paper_trail_core.nlp.nli import infer_batch; infer_batch(['a dog'], ['an animal'])",
])
def test_boosted_trees_survive_torch_loading_first(load_torch_model):
    pytest.importorskip("xgboost")
    pytest.importorskip("lightgbm")
    proc = _run(load_torch_model + "\n" + _BOOSTED_TREES_AFTER)
    assert proc.returncode == 0, f"exit {proc.returncode}\n{proc.stderr[-2000:]}"
    assert proc.stdout.strip().endswith("ok")


def test_preload_is_idempotent_and_tolerates_missing_libraries(monkeypatch):
    from paper_trail_core.nlp import openmp

    monkeypatch.setattr(openmp, "_done", False)
    monkeypatch.setattr(openmp, "_OPENMP_USERS", ("definitely_not_installed_xyz",))
    openmp.load_before_torch()
    openmp.load_before_torch()
    assert openmp._done


@pytest.mark.slow
def test_torch_loads_after_boosted_trees_are_running():
    """The abort case: the system runtime's threads are live when torch's starts."""
    pytest.importorskip("lightgbm")
    proc = _run("""
        import os
        os.environ.pop("KMP_DUPLICATE_LIB_OK", None)
        import numpy as np, lightgbm
        lightgbm.LGBMClassifier(n_estimators=5, verbose=-1).fit(np.random.rand(200, 3), [0, 1] * 100)
        from paper_trail_core.nlp.embeddings import embed
        embed(["hi"])
        print("ok")
    """)
    assert proc.returncode == 0, f"exit {proc.returncode}\\n{proc.stderr[-2000:]}"
    assert proc.stdout.strip().endswith("ok")
