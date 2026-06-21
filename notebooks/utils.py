"""
Shared utilities for all algorithm notebooks.
Import this from any notebook in the capstone-final/notebooks/ folder.
"""
import os, sys, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.titleweight': 'bold',
})

# ── Paths ─────────────────────────────────────────────────────────────────────
def setup_paths():
    notebook_dir = os.getcwd()
    base_dir = os.path.abspath(os.path.join(notebook_dir, '..'))
    figures_dir = os.path.join(notebook_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    sys.path.insert(0, base_dir)
    return base_dir, figures_dir


# ── Probability extraction ─────────────────────────────────────────────────────
def get_proba(model, X_test):
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        s = model.decision_function(X_test)
        s = s if s.ndim == 1 else s[:, 1]
        return (s - s.min()) / (s.max() - s.min() + 1e-10)
    return model.predict(X_test).astype(float)


# ── Metrics table ─────────────────────────────────────────────────────────────
def evaluate_and_print(model, X_test, y_test, model_name, train_time=None):
    y_pred = model.predict(X_test)
    y_prob = get_proba(model, X_test)
    metrics = {
        'Accuracy':      accuracy_score(y_test, y_pred),
        'Precision':     precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'Recall':        recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'F1 Score':      f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'AUC-ROC':       roc_auc_score(y_test, y_prob),
    }
    if train_time is not None:
        metrics['Train Time (s)'] = round(train_time, 2)

    df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    print(f"\n{'='*45}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*45}")
    for _, row in df.iterrows():
        if row['Metric'] == 'Train Time (s)':
            print(f"  {row['Metric']:<18}: {row['Value']:.2f}s")
        else:
            print(f"  {row['Metric']:<18}: {row['Value']:.4f}")
    print(f"{'='*45}\n")
    print(classification_report(y_test, y_pred, target_names=['Fake (0)', 'Real (1)']))
    return metrics, y_pred, y_prob


# ── Plot: Confusion Matrix ────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, title, figures_dir):
    cm = confusion_matrix(y_true, y_pred)
    cmn = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, fmt, suffix in zip(axes, [cm, cmn], ['d', '.2%'], ['Counts', 'Normalized']):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues', ax=ax,
                    xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'],
                    linewidths=0.5, cbar_kws={'shrink': 0.8})
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        ax.set_title(f'{title}\nConfusion Matrix ({suffix})')
    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fname = title.lower().replace(' ', '_').replace('+', 'plus') + '_confusion_matrix.png'
    plt.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches='tight')
    plt.show()


# ── Plot: ROC Curve ───────────────────────────────────────────────────────────
def plot_roc_curve(y_true, y_prob, title, figures_dir):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(fpr, tpr, color='#3498db', lw=2.5, label=f'ROC Curve (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Classifier (AUC = 0.50)')
    ax.fill_between(fpr, tpr, alpha=0.12, color='#3498db')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve — {title}')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    plt.tight_layout()
    fname = title.lower().replace(' ', '_').replace('+', 'plus') + '_roc_curve.png'
    plt.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches='tight')
    plt.show()


# ── Plot: Precision-Recall Curve ──────────────────────────────────────────────
def plot_pr_curve(y_true, y_prob, title, figures_dir):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    baseline = y_true.mean()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(rec, prec, color='#e74c3c', lw=2.5, label=f'PR Curve (AP = {ap:.4f})')
    ax.axhline(baseline, color='grey', linestyle='--', lw=1.2,
               label=f'Baseline (class ratio = {baseline:.2f})')
    ax.fill_between(rec, prec, alpha=0.12, color='#e74c3c')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve — {title}')
    ax.legend(loc='lower left', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.02])
    plt.tight_layout()
    fname = title.lower().replace(' ', '_').replace('+', 'plus') + '_pr_curve.png'
    plt.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches='tight')
    plt.show()


# ── Plot: Learning Curve (accuracy vs training size) ─────────────────────────
def plot_learning_curve(model, X_tr, y_tr, X_te, y_te, title, figures_dir):
    from sklearn.base import clone
    n = X_tr.shape[0]
    fractions = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    sizes, tr_accs, te_accs = [], [], []
    np.random.seed(42)
    for frac in fractions:
        k = max(100, int(n * frac))
        idx = np.random.choice(n, k, replace=False)
        X_sub = X_tr[idx] if hasattr(X_tr, 'tocsr') else X_tr[idx]
        y_sub = y_tr[idx]
        m = clone(model)
        try:
            m.fit(X_sub, y_sub)
            tr_accs.append(accuracy_score(y_sub, m.predict(X_sub)))
            te_accs.append(accuracy_score(y_te, m.predict(X_te)))
            sizes.append(k)
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.plot(sizes, tr_accs, 'o-', color='#3498db', lw=2, ms=7, label='Training Accuracy')
    ax.plot(sizes, te_accs, 's-', color='#e74c3c', lw=2, ms=7, label='Test Accuracy')
    ax.fill_between(sizes, [a - 0.005 for a in tr_accs], [a + 0.005 for a in tr_accs],
                    alpha=0.1, color='#3498db')
    ax.fill_between(sizes, [a - 0.005 for a in te_accs], [a + 0.005 for a in te_accs],
                    alpha=0.1, color='#e74c3c')
    ax.set_xlabel('Number of Training Samples')
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Learning Curve — {title}')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = title.lower().replace(' ', '_').replace('+', 'plus') + '_learning_curve.png'
    plt.savefig(os.path.join(figures_dir, fname), dpi=150, bbox_inches='tight')
    plt.show()
    return sizes, tr_accs, te_accs


# ── Dense feature preparation (for GB, CatBoost) ─────────────────────────────
def get_dense_features(X_train, X_test, max_features=5000, n_components=200, sample_size=20000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    tfidf = TfidfVectorizer(max_features=max_features, sublinear_tf=True, min_df=2)
    if len(X_train) > sample_size:
        np.random.seed(42)
        idx = np.random.choice(len(X_train), sample_size, replace=False)
        X_tr_sparse = tfidf.fit_transform(X_train[idx])
        y_sample_idx = idx
    else:
        X_tr_sparse = tfidf.fit_transform(X_train)
        y_sample_idx = None
    X_te_sparse = tfidf.transform(X_test)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_tr_d = svd.fit_transform(X_tr_sparse)
    X_te_d = svd.transform(X_te_sparse)
    return X_tr_d, X_te_d, y_sample_idx
