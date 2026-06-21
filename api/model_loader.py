"""
Loads all exported models from the models/ directory into memory.
Called once at API startup via FastAPI lifespan.
"""
from __future__ import annotations

import os
import re
import json
import logging
import numpy as np
import joblib
from typing import Any

logger = logging.getLogger(__name__)

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR      = os.path.join(BASE_DIR, 'models')
MAX_TEXT_LENGTH = 50_000


# ── Text cleaning (mirrors preprocess.py) ────────────────────────────────────

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── ModelRegistry ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """
    Loads and holds all trained models in memory.

    Attributes
    ----------
    registry       : raw registry.json dict (metadata)
    classical      : {name: sklearn model}
    tfidf          : fitted TfidfVectorizer
    dense_pipeline : {'tfidf': ..., 'svd': ...}  for GB/CatBoost
    dl_models      : {name: keras model}
    tokenizer      : keras Tokenizer
    dl_config      : {'max_words':..., 'max_len':..., 'vocab_size':...}
    """

    def __init__(self):
        self.registry: dict = {}
        self.classical: dict[str, Any] = {}
        self.tfidf = None
        self.dense_pipeline: dict | None = None
        self.dl_models: dict[str, Any] = {}
        self.tokenizer = None
        self.dl_config: dict = {}
        self._tf_available = False

    # ── Loading ───────────────────────────────────────────────────

    def load_all(self) -> None:
        registry_path = os.path.join(MODELS_DIR, 'registry.json')
        if not os.path.exists(registry_path):
            raise FileNotFoundError(
                f'Model registry not found at {registry_path}. '
                'Run export_models.py first.'
            )

        with open(registry_path) as f:
            self.registry = json.load(f)

        self._load_classical()
        self._load_dl()

        total = len(self.classical) + len(self.dl_models)
        logger.info(
            f'ModelRegistry ready: {len(self.classical)} classical + '
            f'{len(self.dl_models)} DL = {total} models loaded.'
        )

    def _load_classical(self) -> None:
        tfidf_path = os.path.join(MODELS_DIR, 'classical', 'tfidf.joblib')
        if os.path.exists(tfidf_path):
            self.tfidf = joblib.load(tfidf_path)
            logger.info('TF-IDF loaded')

        dense_path = os.path.join(MODELS_DIR, 'classical', 'dense_pipeline.joblib')
        if os.path.exists(dense_path):
            self.dense_pipeline = joblib.load(dense_path)
            logger.info('Dense pipeline loaded')

        for name, info in self.registry.items():
            if info.get('category') != 'Classical ML':
                continue
            model_path = os.path.join(BASE_DIR, info['path'])
            if not os.path.exists(model_path):
                logger.warning(f'Model file missing: {model_path}')
                continue
            try:
                self.classical[name] = joblib.load(model_path)
                logger.info(f'Classical model loaded: {name}')
            except Exception as e:
                logger.error(f'Failed to load {name}: {e}')

    def _load_dl(self) -> None:
        # Try importing TF (it's optional — API still works without it)
        try:
            import tensorflow as tf
            from tensorflow.keras.preprocessing.text import tokenizer_from_json
            self._tf_available = True
        except ImportError:
            logger.warning('TensorFlow not available — DL models will not load.')
            return

        tok_path = os.path.join(MODELS_DIR, 'deep_learning', 'tokenizer.json')
        cfg_path = os.path.join(MODELS_DIR, 'deep_learning', 'config.json')

        if os.path.exists(tok_path):
            with open(tok_path) as f:
                self.tokenizer = tokenizer_from_json(f.read())
            logger.info('Tokenizer loaded')

        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                self.dl_config = json.load(f)

        for name, info in self.registry.items():
            if info.get('category') != 'Deep Learning':
                continue
            model_path = os.path.join(BASE_DIR, info['path'])
            if not os.path.exists(model_path):
                logger.warning(f'DL model file missing: {model_path}')
                continue
            try:
                self.dl_models[name] = tf.keras.models.load_model(model_path)
                logger.info(f'DL model loaded: {name}')
            except Exception as e:
                logger.error(f'Failed to load DL model {name}: {e}')

    # ── Prediction helpers ────────────────────────────────────────

    def _get_prob(self, model, X) -> np.ndarray:
        """Returns array of P(real) for each sample."""
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)[:, 1]
        elif hasattr(model, 'decision_function'):
            s = model.decision_function(X)
            s = s if s.ndim == 1 else s[:, 1]
            return (s - s.min()) / (s.max() - s.min() + 1e-10)
        return model.predict(X).astype(float)

    def _build_result(self, name: str, prob_real: float,
                      threshold: float = 0.5) -> dict:
        prediction = int(prob_real >= threshold)
        prob_fake  = round(1.0 - float(prob_real), 6)
        prob_real_ = round(float(prob_real), 6)
        info = self.registry.get(name, {})
        return {
            'model_name':   name,
            'display_name': info.get('display_name', name),
            'category':     info.get('category', 'Unknown'),
            'prediction':   prediction,
            'label':        'real' if prediction == 1 else 'fake',
            'confidence':   round(max(prob_real_, prob_fake), 6),
            'prob_fake':    prob_fake,
            'prob_real':    prob_real_,
        }

    # ── Public predict API ────────────────────────────────────────

    def predict(self, text: str, model_name: str,
                threshold: float = 0.5) -> dict:
        """
        Predict with a single model.

        Returns a dict matching SinglePrediction schema.
        Raises KeyError if model not found, ValueError if not loaded.
        """
        if model_name not in self.registry:
            available = list(self.registry.keys())
            raise KeyError(
                f"Unknown model '{model_name}'. "
                f"Available: {available}"
            )

        info = self.registry[model_name]
        cleaned = clean_text(text)

        if info['category'] == 'Classical ML':
            return self._predict_classical(cleaned, model_name,
                                           info['feature_type'], threshold)
        elif info['category'] == 'Deep Learning':
            return self._predict_dl(cleaned, model_name, threshold)
        else:
            raise ValueError(f"Unknown category: {info['category']}")

    def _predict_classical(self, cleaned_text: str, name: str,
                           feature_type: str, threshold: float) -> dict:
        if name not in self.classical:
            raise ValueError(f"Model '{name}' is listed in registry but not loaded.")

        model = self.classical[name]

        if feature_type == 'dense':
            if self.dense_pipeline is None:
                raise ValueError('Dense pipeline not loaded.')
            sparse = self.dense_pipeline['tfidf'].transform([cleaned_text])
            X = self.dense_pipeline['svd'].transform(sparse)
        else:
            if self.tfidf is None:
                raise ValueError('TF-IDF vectorizer not loaded.')
            X = self.tfidf.transform([cleaned_text])

        prob_real = float(self._get_prob(model, X)[0])
        return self._build_result(name, prob_real, threshold)

    def _predict_dl(self, cleaned_text: str, name: str,
                    threshold: float) -> dict:
        if not self._tf_available:
            raise ValueError('TensorFlow is not installed — DL predictions unavailable.')
        if name not in self.dl_models:
            raise ValueError(f"DL model '{name}' is listed but not loaded.")
        if self.tokenizer is None:
            raise ValueError('Tokenizer not loaded.')

        from tensorflow.keras.preprocessing.sequence import pad_sequences
        max_len = self.dl_config.get('max_len', 200)

        seq = self.tokenizer.texts_to_sequences([cleaned_text])
        X = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
        prob_real = float(self.dl_models[name].predict(X, verbose=0).flatten()[0])
        return self._build_result(name, prob_real, threshold)

    def predict_all(self, text: str, threshold: float = 0.5) -> list[dict]:
        """Run prediction across every loaded model."""
        cleaned = clean_text(text)
        results = []
        # classical
        for name, info in self.registry.items():
            if info['category'] == 'Classical ML' and name in self.classical:
                try:
                    r = self._predict_classical(cleaned, name,
                                                info['feature_type'], threshold)
                    results.append(r)
                except Exception as e:
                    logger.warning(f'Skipping {name}: {e}')
        # deep learning
        for name, info in self.registry.items():
            if info['category'] == 'Deep Learning' and name in self.dl_models:
                try:
                    r = self._predict_dl(cleaned, name, threshold)
                    results.append(r)
                except Exception as e:
                    logger.warning(f'Skipping {name}: {e}')
        return results

    def predict_batch(self, texts: list[str], model_name: str,
                      threshold: float = 0.5) -> list[dict]:
        """Run a single model across multiple texts."""
        info = self.registry.get(model_name)
        if info is None:
            raise KeyError(f"Unknown model '{model_name}'.")

        _error_template = {
            'model_name': model_name,
            'display_name': info.get('display_name', model_name),
            'category': info.get('category', 'Unknown'),
            'prediction': None,
            'label': None,
            'confidence': None,
            'prob_fake': None,
            'prob_real': None,
        }

        results = []
        for text in texts:
            if len(text) > MAX_TEXT_LENGTH:
                results.append({**_error_template,
                                 'error': f'Text exceeds {MAX_TEXT_LENGTH} character limit.'})
                continue
            cleaned = clean_text(text)
            try:
                if info['category'] == 'Classical ML':
                    r = self._predict_classical(cleaned, model_name,
                                                info['feature_type'], threshold)
                else:
                    r = self._predict_dl(cleaned, model_name, threshold)
                results.append(r)
            except Exception as e:
                results.append({**_error_template, 'error': str(e)})
        return results

    # ── Introspection ─────────────────────────────────────────────

    def list_models(self) -> list[dict]:
        out = []
        for name, info in self.registry.items():
            loaded = (name in self.classical) or (name in self.dl_models)
            out.append({
                'name':         name,
                'display_name': info.get('display_name', name),
                'category':     info.get('category', ''),
                'feature_type': info.get('feature_type', ''),
                'loaded':       loaded,
                'metrics':      info.get('metrics'),
            })
        return out

    @property
    def classical_count(self) -> int:
        return len(self.classical)

    @property
    def dl_count(self) -> int:
        return len(self.dl_models)

    @property
    def total_loaded(self) -> int:
        return self.classical_count + self.dl_count
