# CardioCare — End-to-End Heart Disease ML System

**伦理声明 / Ethics:** 本系统仅作为心脏科医生的决策辅助工具，不具备独立诊断资格，所有最终医疗决策必须由执业医师做出。The system informs clinical decisions; it does not replace physicians.

## Dataset

- **Source:** [UCI Heart Disease — Cleveland subset](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Version:** `processed.cleveland.data` downloaded deterministically via `data/download_data.py`
- **Target:** binarized as `0 = healthy`, `1 = heart disease`

## Quick Start (3 commands)

```bash
pip install -r requirements.txt
python data/download_data.py
python src/train.py
python -m unittest discover -s tests -v
```

Windows one-click reproduction:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/reproduce.ps1
```

View MLflow experiments:

```bash
mlflow ui
```

## Full Reproduction

```bash
git clone <your-repo-url>
cd ml
pip install -r requirements.txt
python data/download_data.py
python src/train.py
python src/inference.py --input data/sample_batch.csv
python src/monitor.py
python -m unittest discover -s tests -v
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
python reports/build_report_pdf.py
```

## Repository Structure

```
├── data/
│   ├── download_data.py
│   ├── heart_disease.csv          # generated
│   └── sample_batch.csv
├── notebooks/
│   └── 01_eda_preprocessing.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── inference.py
│   └── monitor.py
├── tests/
│   └── test_pipeline.py
├── artifacts/
│   ├── model.pkl
│   └── model_metadata.json
├── mlruns/                        # MLflow experiment tracking
├── Dockerfile
├── requirements.txt
├── .github/workflows/ci.yml
├── report.pdf                     # generated
└── README.md
```

## Design Notes

### No Data Leakage
1. `train_test_split` happens **before** any fit operation.
2. Imputation, IQR clipping, scaling, and feature selection are inside sklearn `Pipeline` and fit **only on training data**.

### Reproducibility
- Global random seed: `42`
- Pinned dependency versions in `requirements.txt`

### Model Selection
- Compare Logistic Regression, SVC, and Random Forest
- Track all runs in MLflow
- Final model chosen by **highest recall** to reduce false negatives in clinical use

### Feature Store / Model Registry (conceptual)
- **Feature Store candidate:** `chol` — frequently drift-prone and clinically meaningful; centralizing it enables consistent validation and monitoring.
- **Model Registry metadata:** `training_data_hash`, `recall_on_holdout`, `random_seed`, `selected_features` — required for safe rollback and audit trails.

### MLOps Loop
`Code Commit → CI Tests → CD Deploy → CM Drift Monitoring → CT Retrain → Model Update`

Human-in-the-loop review is required before any automated retraining result reaches production.

## AI Tool Disclosure

See `report.pdf` appendix. This repository used AI assistants for boilerplate scaffolding and debugging; all experiments, metrics, and design decisions were validated by the author.

## GitHub Submission Steps

```bash
cd c:\Users\34364\Desktop\ml
git init
git add .
git commit -m "Complete CardioCare end-to-end ML project"
git branch -M main
git remote add origin https://github.com/<your-username>/cardiocare.git
git push -u origin main
```

After pushing, confirm GitHub Actions CI is green on the `main` branch.

## Docker (optional local check)

```bash
docker build -t cardiocare:1.0 .
docker run --rm cardiocare:1.0
```

## Report

```bash
python reports/build_report_pdf.py
```

This generates `report.pdf` with metrics, ethics statements, model comparison, and embedded figures.
