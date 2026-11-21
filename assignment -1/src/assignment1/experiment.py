from __future__ import annotations

from pathlib import Path
import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, ConfusionMatrixDisplay,
)

from .data import load_multidomain_dataset, dataset_summary, records_to_xy
from .models import build_traditional_models, train_deep_model


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", value.strip()).strip("_").lower()


def _score_vector(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def _evaluate(model_name: str, evaluation_name: str, model, X, y):
    y_true = np.asarray(y, dtype=int)
    y_pred = np.asarray(model.predict(X), dtype=int)
    scores = np.asarray(_score_vector(model, X), dtype=float)
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            pass
    row = {
        "model": model_name,
        "evaluation": evaluation_name,
        "n_test": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": auc,
    }
    return row, y_true, y_pred


def _save_confusion(y_true, y_pred, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1],
        display_labels=["Negative", "Positive"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_latex_results(metrics: pd.DataFrame, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    output = report_dir / "generated_results.tex"
    if metrics.empty:
        output.write_text(
            r"\textit{No experiment results are available yet. Run the experiment script after downloading the official dataset.}",
            encoding="utf-8",
        )
        return
    preferred = ["model", "evaluation", "accuracy", "precision", "recall", "f1", "roc_auc"]
    view = metrics[preferred].copy()
    for col in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        view[col] = view[col].map(lambda x: "--" if pd.isna(x) else f"{x:.3f}")
    latex = view.to_latex(
        index=False, escape=True, column_format="llrrrrr",
        caption="Measured model performance", label="tab:measured-results"
    )
    output.write_text(latex, encoding="utf-8")


def run_experiment(
    data_dir: str | Path,
    output_dir: str | Path,
    source_domain: str = "books",
    target_domains: list[str] | None = None,
    max_features: int = 10000,
    epochs: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """Run in-domain and source-to-target sentiment classification experiments."""
    output = Path(output_dir)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    dataset = load_multidomain_dataset(data_dir)
    domains = sorted(dataset)
    source_domain = source_domain.lower()
    if source_domain not in dataset:
        raise ValueError(f"Source domain '{source_domain}' not found. Available domains: {', '.join(domains)}")

    if target_domains is None:
        preferred = [d for d in ["dvd", "electronics", "kitchen"] if d in dataset and d != source_domain]
        target_domains = preferred or [d for d in domains if d != source_domain][:3]
    target_domains = [d.lower() for d in target_domains if d.lower() in dataset and d.lower() != source_domain]

    pd.DataFrame(dataset_summary(dataset)).to_csv(output / "dataset_summary.csv", index=False)
    source_x, source_y = records_to_xy(dataset[source_domain])
    X_core, X_test, y_core, y_test = train_test_split(
        source_x, source_y, test_size=0.20, random_state=seed, stratify=source_y
    )
    X_train_records, X_valid_records, y_train, y_valid = train_test_split(
        X_core, y_core, test_size=0.20, random_state=seed, stratify=y_core
    )

    # Fit vocabulary, TF-IDF weights and feature selection on training data only.
    vectorizer = DictVectorizer(sparse=True, sort=True)
    X_train_counts = vectorizer.fit_transform(X_train_records)
    X_valid_counts = vectorizer.transform(X_valid_records)
    X_test_counts = vectorizer.transform(X_test)
    tfidf = TfidfTransformer(sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    X_valid_tfidf = tfidf.transform(X_valid_counts)
    X_test_tfidf = tfidf.transform(X_test_counts)

    k = min(max_features, X_train_tfidf.shape[1])
    selector = SelectKBest(score_func=chi2, k=k)
    X_train = selector.fit_transform(X_train_tfidf, np.asarray(y_train))
    X_valid = selector.transform(X_valid_tfidf)
    X_source_test = selector.transform(X_test_tfidf)

    target_matrices = {}
    for target in target_domains:
        tx, ty = records_to_xy(dataset[target])
        matrix = selector.transform(tfidf.transform(vectorizer.transform(tx)))
        target_matrices[target] = (matrix, ty)

    model_objects = {}
    for name, model in build_traditional_models().items():
        model.fit(X_train, np.asarray(y_train))
        model_objects[name] = model

    # The processed ACL representation has no word order. Feed-forward networks
    # avoid inventing token order for recurrent or transformer architectures.
    X_train_dense = X_train.toarray().astype(np.float32)
    X_valid_dense = X_valid.toarray().astype(np.float32)
    model_objects["Deep Neural Network"] = train_deep_model(
        "dnn", X_train_dense, np.asarray(y_train, dtype=np.float32),
        X_valid_dense, np.asarray(y_valid, dtype=np.float32),
        epochs=epochs, seed=seed,
    )
    model_objects["Residual MLP"] = train_deep_model(
        "residual_mlp", X_train_dense, np.asarray(y_train, dtype=np.float32),
        X_valid_dense, np.asarray(y_valid, dtype=np.float32),
        epochs=epochs, seed=seed,
    )

    evaluations = [("in_domain", X_source_test, y_test)]
    evaluations.extend((f"cross_domain:{name}", matrix, labels)
                       for name, (matrix, labels) in target_matrices.items())
    rows = []
    for model_name, model in model_objects.items():
        for evaluation_name, matrix, labels in evaluations:
            X_eval = matrix.toarray().astype(np.float32) if model_name in {
                "Deep Neural Network", "Residual MLP"
            } else matrix
            row, y_true, y_pred = _evaluate(model_name, evaluation_name, model, X_eval, labels)
            rows.append(row)
            _save_confusion(
                y_true, y_pred, f"{model_name} — {evaluation_name}",
                figures / f"confusion_{_safe_name(model_name)}_{_safe_name(evaluation_name)}.png",
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics_summary.csv", index=False)
    metrics.to_json(output / "metrics_summary.json", orient="records", indent=2)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    metrics.pivot(index="model", columns="evaluation", values="f1").plot(kind="bar", ax=ax)
    ax.set_ylabel("F1-score")
    ax.set_ylim(0, 1)
    ax.set_title("F1-score by model and evaluation domain")
    ax.legend(title="Evaluation", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "model_f1_comparison.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "source_domain": source_domain,
        "target_domains": target_domains,
        "domains_available": domains,
        "n_train": len(y_train),
        "n_validation": len(y_valid),
        "n_source_test": len(y_test),
        "selected_features": int(X_train.shape[1]),
        "seed": seed,
        "models": list(model_objects),
    }
    (output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_latex_results(metrics, Path(__file__).resolve().parents[2] / "report")
    return metrics
