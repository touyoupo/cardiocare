# MLflow Tracking Directory

`mlruns/` is generated locally when you run `python src/train.py`.

It is listed in `.gitignore` because experiment files are large and may contain
machine-specific paths. MLflow UI screenshots (3+ runs) are included in
`report.pdf` (Figure 7) and `artifacts/report/mlflow_comparison.png`.

To reproduce locally:

```bash
python src/train.py
mlflow ui
```
