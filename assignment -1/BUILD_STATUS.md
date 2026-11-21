# Build status

Status: Successful

- Official Multi-Domain Sentiment Dataset v2.0 downloaded and processed.
- Four selected domains: 2,000 labelled reviews each (1,000 positive and 1,000 negative; 8,000 total).
- Exactly two traditional models and two deep models trained.
- Baseline and refined versions evaluated.
- Validation-only hyperparameter and decision-threshold selection.
- Trained refined model weights and preprocessing objects saved under results/models.
- Notebook executed with visible outputs.
- Unified metrics table, refinement log and figures generated.
- LaTeX report compiled; TeXcount total: 2,662 words (maximum allowed: 3,000).

## Measured results

- Best refined Books test F1: Deep Neural Network (F1 0.867, accuracy 0.860).
- Highest mean target-domain F1: Deep Neural Network (mean F1 0.779).
- Smallest Books-to-target F1 gap: Logistic Regression (0.060).
- Hardest target: Electronics (mean F1 0.746); easiest: DVD (mean F1 0.792).

## Files

- Notebook: notebooks/assignment_1_sentiment_analysis.ipynb
- Report source: report/main.tex
- Compiled report: report/main.pdf
- Model metrics: results/metrics_summary.csv
- Refinement log: results/refinement_comparison.csv
- Trained artifacts: results/models/
- Figures: results/figures/

The raw dataset is not committed to Git. The download script retrieves it from the official JHU source.
