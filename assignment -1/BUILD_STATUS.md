# Build and execution status

Date: 21 November 2026

## Completed

- Project structure and reusable Python modules created.
- Official JHU `processed_acl` downloader added.
- Parser unit tests pass (2 tests).
- Synthetic smoke test completed for all five model implementations and both in-domain and cross-domain evaluation paths. Synthetic data and their scores are not included in coursework results.
- LaTeX report compiled successfully to a 7-page PDF.
- Jupyter notebook created and validated, but remains unexecuted.

## Not completed

The official JHU dataset was not downloaded and the five models were not trained on the coursework data. The build environment could not resolve the JHU host (network/DNS unavailable). Therefore no real-data metrics, plots or confusion matrices are claimed. The report PDF contains a pending-results notice until the official data run completes.

## Required next steps

1. Install dependencies with `pip install -r requirements.txt`.
2. Run `python scripts/download_dataset.py --variant processed_acl` on a machine with internet access.
3. Run `python scripts/run_experiments.py --data-dir data/raw/processed_acl --source-domain books --target-domains dvd electronics kitchen`.
4. Run every cell in `notebooks/assignment_1_sentiment_analysis.ipynb` so the submitted notebook contains visible outputs.
5. Inspect `results/metrics_summary.csv` and the generated figures. Write your own interpretation, then compile `report/main.tex` twice with `pdflatex`.
6. Replace the pending-results notice and enter your own student details before submission.

The original dataset should be cited using Blitzer, Dredze and Pereira (2007). Raw data are not committed to this public repository.
