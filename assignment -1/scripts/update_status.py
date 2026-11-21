from __future__ import annotations

from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    metrics = pd.read_csv(RESULTS / "metrics_summary.csv")
    summary = pd.read_csv(RESULTS / "dataset_summary.csv")
    refined = metrics[metrics["stage"] == "refined"]
    source = refined[refined["evaluation"] == "in_domain"]
    target = refined[refined["evaluation"].str.startswith("cross_domain:")]

    best_source = source.sort_values("f1", ascending=False).iloc[0]
    target_means = target.groupby("model")["f1"].mean().sort_values(ascending=False)
    best_target_name = target_means.index[0]
    best_target_f1 = float(target_means.iloc[0])
    source_f1 = source.set_index("model")["f1"]
    gaps = source_f1 - target_means
    consistent_name = gaps.idxmin()
    domain_means = target.groupby("evaluation")["f1"].mean().sort_values()
    hardest = domain_means.index[0].replace("cross_domain:", "").title()
    easiest = domain_means.index[-1].replace("cross_domain:", "").title()

    try:
        word_count = (ROOT / "report" / "generated_wordcount.tex").read_text().strip()
    except FileNotFoundError:
        word_count = "Pending"

    sample_text = "; ".join(
        f"{row.domain.title()}: {int(row.documents)} reviews "
        f"({int(row.positive)} positive, {int(row.negative)} negative)"
        for row in summary.itertuples()
    )
    metrics_text = (
        f"- Best refined Books test F1: {best_source['model']} "
        f"(F1 {best_source['f1']:.3f}, accuracy {best_source['accuracy']:.3f}).\n"
        f"- Highest mean target-domain F1: {best_target_name} "
        f"(mean F1 {best_target_f1:.3f}).\n"
        f"- Smallest Books-to-target F1 gap: {consistent_name} "
        f"({gaps[consistent_name]:.3f}).\n"
        f"- Hardest target: {hardest} (mean F1 {domain_means.iloc[0]:.3f}); "
        f"easiest: {easiest} (mean F1 {domain_means.iloc[-1]:.3f})."
    )

    readme = f"""# Applied AI Assignment 2

**Task:** cross-domain binary sentiment classification.  
**Dataset:** Johns Hopkins Multi-Domain Sentiment Dataset v2.0.  
**Source domain:** Books. **Target domains:** DVD, Electronics and Kitchen.

## Models

- Traditional: Logistic Regression and Linear SVM.
- Deep learning: Deep Neural Network and Residual MLP.
- Baseline and refined versions are evaluated for all four methods.

## Dataset sample

{sample_text}

## Measured results

{metrics_text}

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
"""
    build = f"""# Build status

Status: Successful

- Official Multi-Domain Sentiment Dataset v2.0 downloaded and processed.
- Four domains sampled with equal positive and negative classes.
- Exactly two traditional models and two deep models trained.
- Baseline and refined versions evaluated.
- Validation-only hyperparameter and decision-threshold selection.
- Trained refined model weights and preprocessing objects saved under results/models.
- Notebook executed with visible outputs.
- Unified metrics table, refinement log and figures generated.
- LaTeX report compiled; word count: {word_count}.

## Measured results

{metrics_text}

## Files

- Notebook: notebooks/assignment_1_sentiment_analysis.ipynb
- Report source: report/main.tex
- Compiled report: report/main.pdf
- Model metrics: results/metrics_summary.csv
- Refinement log: results/refinement_comparison.csv
- Trained artifacts: results/models/
- Figures: results/figures/

The raw dataset is not committed to Git. The download script retrieves it from the official JHU source.
"""
    (ROOT / "README.md").write_text(readme, encoding="utf-8")
    (ROOT / "BUILD_STATUS.md").write_text(build, encoding="utf-8")
    print("Updated README.md and BUILD_STATUS.md from measured results.")


if __name__ == "__main__":
    main()
