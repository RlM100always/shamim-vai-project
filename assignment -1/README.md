# Applied AI Assignment 2

**Task:** cross-domain binary sentiment classification.  
**Dataset:** Johns Hopkins Multi-Domain Sentiment Dataset v2.0.  
**Source domain:** Books. **Target domains:** DVD, Electronics and Kitchen.

## Models

- Traditional: Logistic Regression and Linear SVM.
- Deep learning: Deep Neural Network and Residual MLP.
- Baseline and refined versions are evaluated for all four methods.

## Dataset sample

Books: 2000 reviews (1000 positive, 1000 negative); Dvd: 2000 reviews (1000 positive, 1000 negative); Electronics: 2000 reviews (1000 positive, 1000 negative); Kitchen: 2000 reviews (1000 positive, 1000 negative)

## Measured results

- Best refined Books test F1: Deep Neural Network (F1 0.867, accuracy 0.860).
- Highest mean target-domain F1: Deep Neural Network (mean F1 0.779).
- Smallest Books-to-target F1 gap: Logistic Regression (0.060).
- Hardest target: Electronics (mean F1 0.746); easiest: Dvd (mean F1 0.792).

The executed notebook, unified metrics table, refinement log, figures, trained refined model files, preprocessing objects, LaTeX source and compiled PDF are included in this folder.

## Run locally

    python -m pip install -r requirements.txt
    python scripts/download_dataset.py
    jupyter nbconvert --execute --to notebook --inplace notebooks/assignment_1_sentiment_analysis.ipynb
    python scripts/build_report.py
    cd report
    pdflatex main.tex
    pdflatex main.tex

Raw data are downloaded from the official source and excluded from Git. The project ZIP is for convenience only; the coursework brief says to submit the notebook and report separately, not as a ZIP.
