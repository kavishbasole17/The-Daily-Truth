"""
Run this script once before starting the API.
Trains all classical and deep learning models (up to 20 classical + 4 deep learning) and saves them to models/.

Usage:
    python export_models.py [--skip-dl]
"""

import os, sys, time, json, warnings, argparse
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR    = os.path.join(BASE_DIR, 'models')
CLASSICAL_DIR = os.path.join(MODELS_DIR, 'classical')
DL_DIR        = os.path.join(MODELS_DIR, 'deep_learning')

for d in [CLASSICAL_DIR, DL_DIR]:
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, BASE_DIR)

from preprocess import load_all_datasets, get_train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import (accuracy_score, f1_score,
                              roc_auc_score, precision_score, recall_score)


# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_metrics(model, X_te, y_te):
    y_pred = model.predict(X_te)
    try:
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_te)[:, 1]
        elif hasattr(model, 'decision_function'):
            s = model.decision_function(X_te)
            s = s if s.ndim == 1 else s[:, 1]
            y_prob = (s - s.min()) / (s.max() - s.min() + 1e-10)
        else:
            y_prob = y_pred.astype(float)
        auc = float(roc_auc_score(y_te, y_prob))
    except Exception:
        auc = None
    return {
        'accuracy':  float(accuracy_score(y_te, y_pred)),
        'f1':        float(f1_score(y_te, y_pred, average='weighted', zero_division=0)),
        'precision': float(precision_score(y_te, y_pred, average='weighted', zero_division=0)),
        'recall':    float(recall_score(y_te, y_pred, average='weighted', zero_division=0)),
        'auc':       auc,
    }


def save_model(model, name, subdir=CLASSICAL_DIR):
    path = os.path.join(subdir, f'{name}.joblib')
    joblib.dump(model, path)
    return path


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    print('\n' + '='*60)
    print('  Loading datasets...')
    print('='*60)
    df = load_all_datasets()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f'  Train: {len(X_train):,}  |  Test: {len(X_test):,}')
    return X_train, X_test, y_train, y_test


# ── Classical ML ──────────────────────────────────────────────────────────────

def train_and_save_classical(X_train, X_test, y_train, y_test):
    print('\n' + '='*60)
    print('  CLASSICAL ML — Training & Saving')
    print('='*60)

    # ── Sparse TF-IDF ────────────────────────────────────────────
    print('\n  Fitting TF-IDF (50K unigrams+bigrams)...')
    tfidf = TfidfVectorizer(
        max_features=50000, ngram_range=(1, 2),
        sublinear_tf=True, min_df=2, max_df=0.95,
        strip_accents='unicode'
    )
    X_tr = tfidf.fit_transform(X_train)
    X_te = tfidf.transform(X_test)
    joblib.dump(tfidf, os.path.join(CLASSICAL_DIR, 'tfidf.joblib'))
    print(f'  TF-IDF saved  ({X_tr.shape[1]:,} features)')

    # ── Dense pipeline for GB / CatBoost ─────────────────────────
    print('\n  Fitting dense pipeline (TF-IDF 5K + SVD-200, sample 20K)...')
    dense_tfidf = TfidfVectorizer(max_features=5000, sublinear_tf=True, min_df=2)
    SAMPLE = 20000
    np.random.seed(42)
    dense_idx = np.random.choice(len(X_train), min(SAMPLE, len(X_train)), replace=False)
    X_d_tr_sparse = dense_tfidf.fit_transform(X_train[dense_idx])
    X_d_te_sparse = dense_tfidf.transform(X_test)
    svd = TruncatedSVD(n_components=200, random_state=42)
    X_d_tr = svd.fit_transform(X_d_tr_sparse)
    X_d_te = svd.transform(X_d_te_sparse)
    y_d_tr = y_train[dense_idx]
    joblib.dump({'tfidf': dense_tfidf, 'svd': svd},
                os.path.join(CLASSICAL_DIR, 'dense_pipeline.joblib'))
    print('  Dense pipeline saved')

    # ── KNN subsample ─────────────────────────────────────────────
    KNN_SAMPLE = 30000
    knn_idx = np.random.choice(len(X_train), min(KNN_SAMPLE, len(X_train)), replace=False)
    X_knn = X_tr[knn_idx]
    y_knn = y_train[knn_idx]

    # ── Model definitions ─────────────────────────────────────────
    from sklearn.linear_model import (LogisticRegression, RidgeClassifier,
                                       SGDClassifier, PassiveAggressiveClassifier,
                                       Perceptron)
    from sklearn.naive_bayes import MultinomialNB, BernoulliNB, ComplementNB
    from sklearn.svm import LinearSVC
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                                   GradientBoostingClassifier, AdaBoostClassifier)
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    model_defs = [
        # (save_name, display_name, model, X_tr, y_tr, X_te, feature_type)
        ('logistic_regression', 'Logistic Regression',
         LogisticRegression(C=1.0, max_iter=1000, random_state=42, n_jobs=-1),
         X_tr, y_train, X_te, 'tfidf'),

        ('multinomial_nb', 'Multinomial Naive Bayes',
         MultinomialNB(alpha=0.1),
         X_tr, y_train, X_te, 'tfidf'),

        ('bernoulli_nb', 'Bernoulli Naive Bayes',
         BernoulliNB(),
         X_tr, y_train, X_te, 'tfidf'),

        ('complement_nb', 'Complement Naive Bayes',
         ComplementNB(),
         X_tr, y_train, X_te, 'tfidf'),

        ('linear_svm', 'Linear SVM',
         LinearSVC(C=1.0, max_iter=2000, random_state=42),
         X_tr, y_train, X_te, 'tfidf'),

        ('decision_tree', 'Decision Tree',
         DecisionTreeClassifier(max_depth=50, random_state=42),
         X_tr, y_train, X_te, 'tfidf'),

        ('random_forest', 'Random Forest',
         RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
         X_tr, y_train, X_te, 'tfidf'),

        ('extra_trees', 'Extra Trees',
         ExtraTreesClassifier(n_estimators=100, random_state=42, n_jobs=-1),
         X_tr, y_train, X_te, 'tfidf'),

        ('adaboost', 'AdaBoost',
         AdaBoostClassifier(n_estimators=100, random_state=42, algorithm='SAMME'),
         X_tr, y_train, X_te, 'tfidf'),

        ('ridge_classifier', 'Ridge Classifier',
         RidgeClassifier(),
         X_tr, y_train, X_te, 'tfidf'),

        ('sgd_classifier', 'SGD Classifier',
         SGDClassifier(loss='modified_huber', max_iter=1000, random_state=42, n_jobs=-1),
         X_tr, y_train, X_te, 'tfidf'),

        ('passive_aggressive', 'Passive Aggressive',
         PassiveAggressiveClassifier(max_iter=1000, random_state=42),
         X_tr, y_train, X_te, 'tfidf'),

        ('perceptron', 'Perceptron',
         Perceptron(max_iter=1000, random_state=42, n_jobs=-1),
         X_tr, y_train, X_te, 'tfidf'),

        ('mlp_classifier', 'MLP Classifier',
         MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=100,
                       random_state=42, early_stopping=True),
         X_tr, y_train, X_te, 'tfidf'),

        ('knn', 'KNN',
         KNeighborsClassifier(n_neighbors=5, n_jobs=-1, metric='cosine'),
         X_knn, y_knn, X_te, 'tfidf'),

        ('gradient_boosting', 'Gradient Boosting',
         GradientBoostingClassifier(n_estimators=100, random_state=42),
         X_d_tr, y_d_tr, X_d_te, 'dense'),
    ]

    # Optional boosting libraries
    for lib_name, model_name, display, Model, kwargs, feat in [
        ('xgboost', 'xgboost', 'XGBoost',
         None, dict(n_estimators=100, random_state=42, eval_metric='logloss',
                    n_jobs=-1, verbosity=0), 'tfidf'),
        ('lightgbm', 'lightgbm', 'LightGBM',
         None, dict(n_estimators=100, random_state=42, n_jobs=-1, verbose=-1), 'tfidf'),
        ('catboost', 'catboost', 'CatBoost',
         None, dict(iterations=100, random_seed=42, verbose=0), 'dense'),
    ]:
        try:
            if lib_name == 'xgboost':
                from xgboost import XGBClassifier
                Model = XGBClassifier
            elif lib_name == 'lightgbm':
                from lightgbm import LGBMClassifier
                Model = LGBMClassifier
            else:
                from catboost import CatBoostClassifier
                Model = CatBoostClassifier
            tr_x = X_tr if feat == 'tfidf' else X_d_tr
            te_x = X_te if feat == 'tfidf' else X_d_te
            tr_y = y_train if feat == 'tfidf' else y_d_tr
            model_defs.append((model_name, display, Model(**kwargs),
                               tr_x, tr_y, te_x, feat))
            print(f'  {display}: available')
        except ImportError:
            print(f'  {display}: not installed — skipping')

    # ── Train & save ─────────────────────────────────────────────
    registry = {}
    total = len(model_defs)
    for i, (name, display, model, Xtr, ytr, Xte, feat) in enumerate(model_defs, 1):
        print(f'\n  [{i:>2}/{total}] {display}')
        try:
            t0 = time.time()
            model.fit(Xtr, ytr)
            elapsed = round(time.time() - t0, 2)
            m = compute_metrics(model, Xte, y_test)
            path = save_model(model, name)
            registry[name] = {
                'display_name': display,
                'category':     'Classical ML',
                'feature_type': feat,
                'path':         os.path.relpath(path, BASE_DIR),
                'train_time':   elapsed,
                'metrics':      m,
            }
            auc_str = f'{m["auc"]:.4f}' if m["auc"] is not None else 'N/A'
            print(f'     Acc={m["accuracy"]:.4f}  F1={m["f1"]:.4f}  '
                  f'AUC={auc_str}  Time={elapsed}s  →  saved')
        except Exception as e:
            print(f'     ERROR: {e}')

    return registry


# ── Deep Learning ─────────────────────────────────────────────────────────────

def train_and_save_dl(X_train, X_test, y_train, y_test):
    print('\n' + '='*60)
    print('  DEEP LEARNING — Training & Saving')
    print('='*60)

    try:
        import tensorflow as tf
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from tensorflow.keras.callbacks import EarlyStopping
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import (Embedding, LSTM, GRU,
                                              Conv1D, MaxPooling1D,
                                              GlobalMaxPooling1D, Dense, Dropout)
        print(f'  TensorFlow {tf.__version__}  |  '
              f'GPU: {tf.config.list_physical_devices("GPU")}')
    except ImportError:
        print('  TensorFlow not installed — skipping deep learning models.')
        return {}

    MAX_WORDS = 30000
    MAX_LEN   = 200
    MAX_SAMPLES = 80000
    EMBEDDING_DIM = 128
    EPOCHS = 10
    BATCH_SIZE = 64

    # Subsample
    if len(X_train) > MAX_SAMPLES:
        np.random.seed(42)
        idx = np.random.choice(len(X_train), MAX_SAMPLES, replace=False)
        X_tr_raw, y_tr = X_train[idx], y_train[idx]
    else:
        X_tr_raw, y_tr = X_train, y_train

    # Tokenizer
    print(f'\n  Tokenizing (vocab={MAX_WORDS:,}, max_len={MAX_LEN})...')
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
    tokenizer.fit_on_texts(X_tr_raw)
    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    X_tr_pad = pad_sequences(tokenizer.texts_to_sequences(X_tr_raw),
                             maxlen=MAX_LEN, padding='post', truncating='post')
    X_te_pad = pad_sequences(tokenizer.texts_to_sequences(X_test),
                             maxlen=MAX_LEN, padding='post', truncating='post')

    # Save tokenizer config
    tok_path = os.path.join(DL_DIR, 'tokenizer.json')
    with open(tok_path, 'w') as f:
        f.write(tokenizer.to_json())
    # Save inference constants
    dl_config = {'max_words': MAX_WORDS, 'max_len': MAX_LEN, 'vocab_size': vocab_size}
    with open(os.path.join(DL_DIR, 'config.json'), 'w') as f:
        json.dump(dl_config, f)
    print(f'  Vocabulary: {vocab_size:,}  |  Tokenizer saved')

    # Model builders
    def build_lstm():
        m = Sequential([
            Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
            LSTM(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ], name='LSTM')
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return m

    def build_gru():
        m = Sequential([
            Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
            GRU(128, dropout=0.2, recurrent_dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ], name='GRU')
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return m

    def build_cnn():
        m = Sequential([
            Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
            Conv1D(128, kernel_size=5, activation='relu'),
            GlobalMaxPooling1D(),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ], name='CNN')
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return m

    def build_cnn_lstm():
        m = Sequential([
            Embedding(vocab_size, EMBEDDING_DIM, input_length=MAX_LEN),
            Conv1D(64, kernel_size=5, activation='relu'),
            MaxPooling1D(pool_size=4),
            LSTM(64, dropout=0.2),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ], name='CNN_LSTM')
        m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return m

    builders = [
        ('lstm',     'LSTM',       build_lstm),
        ('gru',      'GRU',        build_gru),
        ('cnn',      'CNN',        build_cnn),
        ('cnn_lstm', 'CNN + LSTM', build_cnn_lstm),
    ]

    early_stop = EarlyStopping(monitor='val_loss', patience=2,
                               restore_best_weights=True, verbose=0)

    registry = {}
    for i, (name, display, builder) in enumerate(builders, 1):
        print(f'\n  [{i}/4] {display}')
        try:
            model = builder()
            t0 = time.time()
            model.fit(X_tr_pad, y_tr,
                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                      validation_split=0.1,
                      callbacks=[early_stop],
                      verbose=1)
            elapsed = round(time.time() - t0, 2)

            # Evaluate
            y_prob = model.predict(X_te_pad, verbose=0).flatten()
            y_pred = (y_prob >= 0.5).astype(int)
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
            m = {
                'accuracy':  float(accuracy_score(y_test, y_pred)),
                'f1':        float(f1_score(y_test, y_pred, average='weighted', zero_division=0)),
                'auc':       float(roc_auc_score(y_test, y_prob)),
            }

            # Save model
            model_path = os.path.join(DL_DIR, f'{name}.keras')
            model.save(model_path)
            registry[name] = {
                'display_name': display,
                'category':     'Deep Learning',
                'feature_type': 'sequence',
                'path':         os.path.relpath(model_path, BASE_DIR),
                'train_time':   elapsed,
                'metrics':      m,
            }
            print(f'     Acc={m["accuracy"]:.4f}  F1={m["f1"]:.4f}  '
                  f'AUC={m["auc"]:.4f}  Time={elapsed}s  →  saved')
        except Exception as e:
            print(f'     ERROR: {e}')

    return registry


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-dl', action='store_true',
                        help='Skip deep learning models (faster)')
    parser.add_argument('--skip-classical', action='store_true',
                        help='Skip classical ML models')
    args = parser.parse_args()

    print('\n' + '='*60)
    print('  FAKE NEWS DETECTION — MODEL EXPORT')
    print('='*60)

    X_train, X_test, y_train, y_test = load_data()

    registry = {}

    if not args.skip_classical:
        cl_registry = train_and_save_classical(X_train, X_test, y_train, y_test)
        registry.update(cl_registry)

    if not args.skip_dl:
        dl_registry = train_and_save_dl(X_train, X_test, y_train, y_test)
        registry.update(dl_registry)

    # Save combined registry
    registry_path = os.path.join(MODELS_DIR, 'registry.json')
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)

    # Save results CSV (used by the API /results endpoint and the comparison notebook)
    results_dir = os.path.join(BASE_DIR, 'results')
    os.makedirs(results_dir, exist_ok=True)
    rows = []
    for info in registry.values():
        m = info['metrics']
        rows.append({
            'model':      info['display_name'],
            'category':   info['category'],
            'accuracy':   m.get('accuracy'),
            'f1':         m.get('f1'),
            'precision':  m.get('precision'),
            'recall':     m.get('recall'),
            'auc':        m.get('auc'),
            'train_time': info.get('train_time'),
        })
    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(results_dir, 'all_results.csv')
    results_df.to_csv(csv_path, index=False)
    print(f'\n  Results CSV saved: {csv_path}')

    print('\n' + '='*60)
    print(f'  Export complete!  {len(registry)} models saved.')
    print(f'  Registry: {registry_path}')
    print('='*60)
    print('\n  Summary:')
    for name, info in registry.items():
        m = info['metrics']
        print(f'    {info["display_name"]:<28} '
              f'Acc={m["accuracy"]:.4f}  F1={m["f1"]:.4f}  '
              f'Time={info["train_time"]}s')


if __name__ == '__main__':
    main()
