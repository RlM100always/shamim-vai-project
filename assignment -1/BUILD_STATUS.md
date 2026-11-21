# Build status

The project now follows the coursework specification:

- One approved dataset: Multi-Domain Sentiment Dataset v2.0.
- One NLP task: Books-to-DVD/Electronics/Kitchen cross-domain binary sentiment classification.
- Exactly two traditional models: Logistic Regression and Linear SVM.
- Exactly two deep models: DNN and Residual MLP.
- Baseline and refined versions for all four models.
- Stratified source split, validation-only tuning, target-domain holdout.
- Unified evaluation table, class-balance plot, feature-frequency plot, adaptation-gap plot and confusion matrices.
- LaTeX report with a designed cover page and generated word count.

The full workflow downloads the official JHU archive, executes the notebook, generates metrics and figures, inserts them into the report, compiles the PDF and commits generated outputs. The official archive is not stored in Git.

No real-data metrics are included until the GitHub Actions run completes successfully. Check the workflow run and the generated files before submission.
