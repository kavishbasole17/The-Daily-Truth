# paper_trail_core — ML/agentic core for the Paper Trail fake-news detection platform.
import os as _os

# Keep `transformers` from importing its TensorFlow/Keras integration.
#
# The project uses both torch (NLI, embeddings) and TensorFlow (the .keras
# deep-learning base models) in one process. On import, transformers eagerly
# probes for TF support, and with Keras 3 installed that probe raises
#   "Your currently installed version of Keras is Keras 3, but this is not yet
#    supported in Transformers. Please install tf-keras."
# which surfaces as a plain ImportError inside sentence-transformers. The
# caller then silently falls back — evidence relevance gating degrades to a
# flat 0.5 for every snippet, which quietly disables the gate entirely.
#
# These flags disable transformers' *TF backend only*. TensorFlow itself still
# loads normally for api/model_loader.py's keras models.
_os.environ.setdefault("USE_TF", "0")
_os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
