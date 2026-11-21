# Applied AI Assignment 2

**Task:** cross-domain binary sentiment classification.  
**Dataset:** Johns Hopkins Multi-Domain Sentiment Dataset v2.0 (processed ACL).  
**Source:** Books. **Targets:** DVD, Electronics and Kitchen.

## Models

- Traditional: Logistic Regression, Linear SVM.
- Deep learning: Deep Neural Network, Residual MLP.
- Each model has a baseline and a refined version.

## Run locally

```bash
python -m pip install -r requirements.txt
python scripts/download_dataset.py
jupyter nbconvert --execute --to notebook --inplace notebooks/assignment_1_sentiment_analysis.ipynb
python scripts/build_report.py
cd report
pdflatex main.tex
pdflatex main.tex
```

The experiment saves metrics and figures under results/. The report inserts the measured metrics, dataset counts and results discussion automatically.

The official raw archive is excluded from Git. Do not submit a ZIP: the brief requires the notebook and report as separate files.
