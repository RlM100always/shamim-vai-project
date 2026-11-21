# Applied AI Assignment 1 — Multi-Domain Sentiment Analysis

This folder contains the project code, Jupyter notebook and LaTeX report for the Applied AI coursework. It selects **one** of the four allowed datasets: the Johns Hopkins Multi-Domain Sentiment Dataset (version 2.0).

## Models

- Traditional ML: Multinomial Naive Bayes, Logistic Regression, Linear SVM
- Deep learning: feed-forward Deep Neural Network, Residual MLP

The selected `processed_acl` variant stores bag-of-words feature counts rather than original word order, so feed-forward neural networks are used instead of sequence models.

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/download_dataset.py --variant processed_acl
python scripts/run_experiments.py --data-dir data/raw/processed_acl --source-domain books --target-domains dvd electronics kitchen
```

Open and run all cells in `notebooks/assignment_1_sentiment_analysis.ipynb`. The experiment writes metrics, confusion matrices and comparison charts under `results/`. Recompile `report/main.tex` after the experiment to insert the measured metrics.

## Current build status

The code and report draft are prepared. The official JHU archive could not be downloaded in the build environment because its DNS/network access was unavailable. Therefore the models have **not** been trained on the real coursework data and no real-data scores are claimed. Read `BUILD_STATUS.md` before using the report.

The raw dataset is not committed to this public repository. Download it from the official source: https://www.cs.jhu.edu/~mdredze/datasets/sentiment/

**Important:** the project ZIP is for convenience only. The assignment brief says not to upload a ZIP to the submission portal; submit the notebook and report as separate files. Do not submit the report until the real results are populated and interpreted in your own words.
