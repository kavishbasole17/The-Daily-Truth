# Fake News Detection — Capstone Project

A comprehensive benchmark of **22 machine learning and deep learning models** for fake news classification, trained across 6 real-world datasets. Includes a FastAPI backend for live inference.

---

## Project Structure

```
capstone-final/
├── dataset/              # Raw datasets (tracked via Git LFS)
│   ├── BSDetector/
│   ├── BuzzFeed/
│   ├── FakeNewsNet/
│   ├── ISOT/
│   ├── LIAR Dataset/
│   └── MultiFC/
├── notebooks/            # One notebook per model + EDA + comparison
│   ├── 00_preprocessing_eda.ipynb
│   ├── 01_logistic_regression.ipynb
│   │   ... (22 model notebooks)
│   └── 23_comparison_all_models.ipynb
├── models/               # Saved model files (tracked via Git LFS)
│   ├── classical/        # .joblib files (sklearn/boosting models + TF-IDF)
│   └── deep_learning/    # .keras files (LSTM, GRU, CNN, CNN-LSTM) + tokenizer
├── api/                  # FastAPI backend
│   ├── main.py
│   ├── model_loader.py
│   └── schemas.py
├── results/              # Benchmark CSVs and chart PNGs
├── export_models.py      # Train and save all models
├── preprocess.py         # Text cleaning utilities
└── requirements.txt
```

---

## Datasets

| Dataset | Description |
|---|---|
| **ISOT** | ~44 000 real/fake news articles from Reuters and fake news websites |
| **FakeNewsNet** | PolitiFact and GossipCop articles with fact-checked labels |
| **BuzzFeed** | Facebook fact-check dataset |
| **LIAR Dataset** | 12 800 short political statements with truthfulness ratings |
| **MultiFC** | Multi-domain fake news corpus |
| **BSDetector** | Web-scraped articles from known unreliable sources |

> Large CSV files are stored with **Git LFS**. Run `git lfs pull` after cloning to download them.

---

## Models

### Classical ML (18 models)

| Model | File |
|---|---|
| Logistic Regression | `models/classical/logistic_regression.joblib` |
| Bernoulli Naive Bayes | `models/classical/bernoulli_nb.joblib` |
| Complement Naive Bayes | `models/classical/complement_nb.joblib` |
| Multinomial Naive Bayes | `models/classical/multinomial_nb.joblib` |
| Linear SVM | `models/classical/linear_svm.joblib` |
| Decision Tree | `models/classical/decision_tree.joblib` |
| Random Forest | `models/classical/random_forest.joblib` |
| Extra Trees | `models/classical/extra_trees.joblib` |
| AdaBoost | `models/classical/adaboost.joblib` |
| Gradient Boosting | `models/classical/gradient_boosting.joblib` |
| LightGBM | `models/classical/lightgbm.joblib` |
| CatBoost | `models/classical/catboost.joblib` |
| Ridge Classifier | `models/classical/ridge_classifier.joblib` |
| SGD Classifier | `models/classical/sgd_classifier.joblib` |
| Passive Aggressive | `models/classical/passive_aggressive.joblib` |
| Perceptron | `models/classical/perceptron.joblib` |
| KNN | `models/classical/knn.joblib` |
| MLP Classifier | `models/classical/mlp_classifier.joblib` |

All classical models use TF-IDF vectorization (`models/classical/tfidf.joblib`).

### Hybrid (1 model)

| Model | File |
|---|---|
| Hybrid Stacking (LR) | `models/hybrid/stacking_lr.joblib` |

A two-layer stacking ensemble. All 22 base model probability outputs are concatenated into a 22-feature vector, which is fed into a Logistic Regression meta-learner. The meta-learner learns which base models to trust and combines them optimally.

### Deep Learning (4 models)

| Model | File |
|---|---|
| LSTM | `models/deep_learning/lstm.keras` |
| GRU | `models/deep_learning/gru.keras` |
| CNN | `models/deep_learning/cnn.keras` |
| CNN-LSTM | `models/deep_learning/cnn_lstm.keras` |

Deep learning models use a shared Keras tokenizer (`models/deep_learning/tokenizer.json`).

> Model files are stored with **Git LFS**. Run `git lfs pull` after cloning.

> The hybrid model (`stacking_lr.joblib`) is generated **after** all 22 base models are trained. Run `python export_models.py` (or notebook `24_hybrid_stacking.ipynb`) once base models exist.

---

## Setup

**Prerequisites**: Python 3.9+, [Git LFS](https://git-lfs.github.com/)

```bash
# 1. Clone and pull LFS files
git clone https://github.com/kavishbasole17/capstone.git
cd capstone
git lfs pull

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Retrain all models from scratch
python export_models.py
```

---

## Running the API

```bash
uvicorn api.main:app --reload
```

Open `http://localhost:8000/docs` for the interactive Swagger UI.

### Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API info and version |
| `GET` | `/health` | Health check + loaded model counts |
| `GET` | `/models` | All available models with metrics |
| `POST` | `/predict` | Classify text with a single named model |
| `POST` | `/predict/all` | Classify with all models + majority-vote consensus |
| `POST` | `/predict/batch` | Classify up to 100 texts with one model |
| `GET` | `/results` | Benchmark results as JSON |
| `GET` | `/results/charts` | List available benchmark chart images |
| `GET` | `/results/charts/{name}` | Serve a benchmark chart PNG |

### Example Request

```bash
curl -X POST "http://localhost:8000/predict/all" \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists discover new evidence of water on Mars."}'
```

**Response** includes per-model predictions (`fake`/`real`) with confidence scores and a majority-vote consensus across all 22 models.

---

## Notebooks

Run notebooks in order for a full walkthrough:

- `00_preprocessing_eda.ipynb` — Data loading, cleaning, EDA
- `01` – `17` — One notebook per classical ML model
- `18`, `20`, `21`, `22` — LSTM, GRU, CNN, CNN-LSTM
- `23_comparison_all_models.ipynb` — Cross-model comparison charts
- `24_hybrid_stacking.ipynb` — Hybrid stacking ensemble (runs after base models are saved)

---

## Results

Benchmark outputs are saved to `results/`:

- `all_results.csv` — Accuracy, F1, Precision, Recall, AUC, training time for all 22 models
- `classical_results.csv` — Classical models only
- `figures/` — Comparison bar charts and AUC plots

---

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM, CatBoost
- **DL**: TensorFlow / Keras
- **API**: FastAPI, Uvicorn, Pydantic
- **Data**: pandas, NumPy, NLTK
- **Viz**: Matplotlib, Seaborn
