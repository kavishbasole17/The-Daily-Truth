import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _check_path(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset file not found: {path}\n"
            "Please download the required datasets and place them in the dataset/ directory."
        )


def load_isot():
    print("  [1/6] Loading ISOT Dataset...")
    fake_path = os.path.join(DATASET_PATH, "ISOT", "Fake.csv")
    true_path = os.path.join(DATASET_PATH, "ISOT", "True.csv")
    _check_path(fake_path)
    _check_path(true_path)
    fake = pd.read_csv(fake_path, encoding='utf-8', on_bad_lines='skip')
    true = pd.read_csv(true_path, encoding='utf-8', on_bad_lines='skip')
    fake['label'] = 0
    true['label'] = 1
    fake['text_content'] = fake['title'].fillna('') + ' ' + fake['text'].fillna('')
    true['text_content'] = true['title'].fillna('') + ' ' + true['text'].fillna('')
    df = pd.concat([fake[['text_content', 'label']], true[['text_content', 'label']]], ignore_index=True)
    df['source'] = 'ISOT'
    print(f"      Loaded {len(df):,} records  (Fake={( df.label==0).sum():,}  Real={(df.label==1).sum():,})")
    return df


def load_fakenewsnet():
    print("  [2/6] Loading FakeNewsNet Dataset...")
    files = [
        ("gossipcop_fake.csv", 0),
        ("gossipcop_real.csv", 1),
        ("politifact_fake.csv", 0),
        ("politifact_real.csv", 1),
    ]
    dfs = []
    for fname, label in files:
        path = os.path.join(DATASET_PATH, "FakeNewsNet", fname)
        _check_path(path)
        df = pd.read_csv(path, on_bad_lines='skip')
        df['label'] = label
        df['text_content'] = df['title'].fillna('')
        dfs.append(df[['text_content', 'label']])
    result = pd.concat(dfs, ignore_index=True)
    result['source'] = 'FakeNewsNet'
    print(f"      Loaded {len(result):,} records  (Fake={(result.label==0).sum():,}  Real={(result.label==1).sum():,})")
    return result


def load_liar():
    print("  [3/6] Loading LIAR Dataset...")
    cols = [
        'id', 'label', 'statement', 'subject', 'speaker', 'speaker_title',
        'state', 'party', 'mostly_true_counts', 'false_counts',
        'half_true_counts', 'barely_true_counts', 'pants_fire_counts', 'context'
    ]
    dfs = []
    for fname in ['train.tsv', 'test.tsv', 'valid.tsv']:
        path = os.path.join(DATASET_PATH, "LIAR Dataset", fname)
        _check_path(path)
        df = pd.read_csv(path, sep='\t', header=None, names=cols, on_bad_lines='skip')
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    label_map = {
        'false': 0, 'pants-fire': 0, 'barely-true': 0,
        'half-true': 1, 'mostly-true': 1, 'true': 1
    }
    df['label'] = df['label'].map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df['text_content'] = df['statement'].fillna('')
    df['source'] = 'LIAR'
    result = df[['text_content', 'label', 'source']]
    print(f"      Loaded {len(result):,} records  (Fake={(result.label==0).sum():,}  Real={(result.label==1).sum():,})")
    return result


def load_multifc():
    print("  [4/6] Loading MultiFC Dataset...")
    path = os.path.join(DATASET_PATH, "MultiFC", "merged_fakenews.csv")
    _check_path(path)
    df = pd.read_csv(path, on_bad_lines='skip')
    df['text_content'] = df['text1'].fillna('') + ' ' + df['text2'].fillna('')
    df['label'] = pd.to_numeric(df['label'], errors='coerce')
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df['source'] = 'MultiFC'
    result = df[['text_content', 'label', 'source']]
    print(f"      Loaded {len(result):,} records  (Fake={(result.label==0).sum():,}  Real={(result.label==1).sum():,})")
    return result


def load_buzzfeed():
    print("  [5/6] Loading BuzzFeed Dataset...")
    path = os.path.join(DATASET_PATH, "BuzzFeed", "facebook-fact-check.csv")
    _check_path(path)
    df = pd.read_csv(path, on_bad_lines='skip')
    label_map = {
        'mostly true': 1,
        'mostly false': 0,
        'no factual content': 0,
        'mixture of true and false': 0
    }
    df['label'] = df['Rating'].str.lower().str.strip().map(label_map)
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    df['text_content'] = df['Category'].fillna('') + ' ' + df['Page'].fillna('')
    df['source'] = 'BuzzFeed'
    result = df[['text_content', 'label', 'source']]
    print(f"      Loaded {len(result):,} records  (Fake={(result.label==0).sum():,}  Real={(result.label==1).sum():,})")
    return result


def load_adversarial():
    print("  [6/6] Loading Adversarial (style-matched hard negatives) Dataset...")
    path = os.path.join(DATASET_PATH, "Adversarial", "adversarial_style.csv")
    _check_path(path)
    df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
    df['text_content'] = df['text_content'].fillna('')
    df['source'] = 'Adversarial'
    result = df[['text_content', 'label', 'source']]
    print(f"      Loaded {len(result):,} records  (Fake={(result.label==0).sum():,}  Real={(result.label==1).sum():,})")
    return result


def load_all_datasets():
    print("\n" + "="*60)
    print("  LOADING ALL DATASETS")
    print("="*60)

    datasets = [
        load_isot(),
        load_fakenewsnet(),
        load_liar(),
        load_multifc(),
        load_buzzfeed(),
        load_adversarial(),
    ]

    df = pd.concat(datasets, ignore_index=True)
    print(f"\n  Combined total   : {len(df):,} records")
    print(f"  Fake (0)         : {(df.label==0).sum():,}")
    print(f"  Real (1)         : {(df.label==1).sum():,}")

    print("\n  Cleaning text...")
    df['text_clean'] = df['text_content'].apply(clean_text)
    df = df[df['text_clean'].str.len() > 20].reset_index(drop=True)
    df = df.drop_duplicates(subset=['text_clean']).reset_index(drop=True)

    print(f"  After clean+dedup: {len(df):,} records")
    print(f"\n  Source distribution:\n{df['source'].value_counts().to_string()}")
    return df


def get_train_test_split(df, test_size=0.2, random_state=42,
                          oversample_source=None, oversample_factor=1):
    """
    Splits into train/test. If oversample_source is given, rows from that
    'source' landing in the *training* split are repeated oversample_factor
    times so a small but important slice of data (e.g. hard negatives) carries
    proportionally more weight during training. The test set is left untouched
    so evaluation metrics stay honest.
    """
    X = df['text_clean'].values
    y = df['label'].values

    if oversample_source is None or 'source' not in df.columns:
        return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

    src = df['source'].values
    X_train, X_test, y_train, y_test, src_train, _ = train_test_split(
        X, y, src, test_size=test_size, random_state=random_state, stratify=y
    )

    if oversample_factor > 1:
        mask = src_train == oversample_source
        if mask.any():
            X_train = np.concatenate([X_train] + [X_train[mask]] * (oversample_factor - 1))
            y_train = np.concatenate([y_train] + [y_train[mask]] * (oversample_factor - 1))

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    df = load_all_datasets()
    X_train, X_test, y_train, y_test = get_train_test_split(df)
    print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")
