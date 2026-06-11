MLflow screenshot for report.pdf (section 5.5.12)
================================================

1. Run training (creates >=3 runs in mlruns/):
   python src/train.py

2. Start MLflow UI:
   mlflow ui

3. Open http://127.0.0.1:5000 in browser.

4. Select experiment "cardiocare" (or default).
   Screenshot the runs table showing >=3 runs with metrics.

5. Save screenshot as:
   artifacts/report/mlflow_comparison.png

6. Rebuild report:
   python reports/build_report_pdf.py

The report will auto-embed mlflow_comparison.png when this file exists.
