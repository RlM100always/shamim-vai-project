from __future__ import annotations

from pathlib import Path
import json
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split

from .data import (
    load_multidomain_dataset, dataset_summary, records_to_xy,
    sample_balanced_domains,
)
from .models import (
    make_logistic_regression, make_linear_svm, train_deep_model,
)


def _score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X).astype(float)


def _default_threshold(model):
    return 0.5 if hasattr(model, "predict_proba") else 0.0


def _predict_at_threshold(model, X, threshold):
    return (_score(model, X) >= threshold).astype(int)


def _best_threshold(model, X_valid, y_valid):
    scores = np.asarray(_score(model, X_valid), dtype=float)
    candidates = np.unique(np.quantile(scores, np.linspace(0.0, 1.0, 101)))
    candidates = np.unique(np.concatenate((
        [scores.min() - 1e-8], candidates, [scores.max() + 1e-8],
        [_default_threshold(model)],
    )))
    default = _default_threshold(model)
    best_threshold = default
    best_f1 = -1.0
    best_distance = float("inf")
    for threshold in candidates:
        prediction = (scores >= threshold).astype(int)
        score = float(f1_score(y_valid, prediction, zero_division=0))
        distance = abs(float(threshold) - default)
        if score > best_f1 or (score == best_f1 and distance < best_distance):
            best_f1 = score
            best_threshold = float(threshold)
            best_distance = distance
    return best_threshold, best_f1


def _evaluate(model_name, stage, evaluation, model, X, y, validation_f1, threshold):
    y_true = np.asarray(y, dtype=int)
    scores = np.asarray(_score(model, X), dtype=float)
    prediction = (scores >= threshold).astype(int)
    auc = float("nan")
    if len(np.unique(y_true)) == 2:
        try:
            auc = float(roc_auc_score(y_true, scores))
        except ValueError:
            pass
    return {
        "model": model_name,
        "stage": stage,
        "evaluation": evaluation,
        "n_test": len(y_true),
        "validation_f1": float(validation_f1),
        "accuracy": float(accuracy_score(y_true, prediction)),
        "precision": float(precision_score(y_true, prediction, zero_division=0)),
        "recall": float(recall_score(y_true, prediction, zero_division=0)),
        "f1": float(f1_score(y_true, prediction, zero_division=0)),
        "roc_auc": auc,
    }, y_true, prediction


def _safe(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _save_confusion(y_true, y_pred, title, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=[0, 1],
        display_labels=["Negative", "Positive"],
        cmap="Blues", colorbar=False, ax=ax,
    )
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _latex_escape(value) -> str:
    text = str(value)
    for char, replacement in [
        ("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
        ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
        ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        text = text.replace(char, replacement)
    return text


def _write_report_tables(metrics, sampled_summary, report_dir):
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset_lines = [
        r"\begin{table}[ht]",
        r"\centering\small",
        r"\caption{Number of documents used in each domain after balanced sampling.}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Domain & Positive & Negative & Total \\",
        r"\midrule",
    ]
    for row in sampled_summary:
        dataset_lines.append(
            f"{_latex_escape(row['domain'].title())} & {row['positive']} & {row['negative']} & {row['documents']} " + r"\\"
        )
    dataset_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (report_dir / "generated_dataset.tex").write_text("\n".join(dataset_lines), encoding="utf-8")

    table = metrics.copy()
    lines = [
        r"\begin{longtable}{ll l r r r r r}",
        r"\caption{Unified evaluation results for all model variants.}\\",
        r"\toprule",
        r"Model & Stage & Test domain & Acc. & Prec. & Recall & F1 & ROC-AUC \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Model & Stage & Test domain & Acc. & Prec. & Recall & F1 & ROC-AUC \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for _, row in table.iterrows():
        evaluation = "Books test" if row["evaluation"] == "in_domain" else row["evaluation"].replace("cross_domain:", "").title()
        values = []
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            value = row[key]
            values.append("--" if pd.isna(value) else f"{value:.3f}")
        lines.append(
            f"{_latex_escape(row['model'])} & {_latex_escape(row['stage'])} & "
            f"{_latex_escape(evaluation)} & " + " & ".join(values) + r" \\"
        )
    lines.append(r"\end{longtable}")
    (report_dir / "generated_results.tex").write_text("\n".join(lines), encoding="utf-8")


def _write_discussion(metrics, report_dir, refinement_rows):
    refined = metrics[metrics["stage"] == "refined"]
    source = refined[refined["evaluation"] == "in_domain"]
    target = refined[refined["evaluation"].str.startswith("cross_domain:")]
    best_source = source.sort_values("f1", ascending=False).iloc[0]
    cross_mean = target.groupby("model", as_index=False)["f1"].mean().sort_values("f1", ascending=False)
    best_cross = cross_mean.iloc[0]
    domain_means = target.groupby("evaluation")["f1"].mean().sort_values()
    hardest = domain_means.index[0].replace("cross_domain:", "").title()
    easiest = domain_means.index[-1].replace("cross_domain:", "").title()

    source_line = (
        f"The strongest refined in-domain result was {best_source['model']} "
        f"(F1={best_source['f1']:.3f}; accuracy={best_source['accuracy']:.3f})."
    )
    cross_line = (
        f"The highest mean target-domain F1 was obtained by {best_cross['model']} "
        f"(mean F1={best_cross['f1']:.3f})."
    )
    domain_line = (
        f"Across the refined models, {hardest} was the most difficult target domain "
        f"(mean F1={domain_means.iloc[0]:.3f}), while {easiest} was the easiest "
        f"(mean F1={domain_means.iloc[-1]:.3f})."
    )
    refinement_lines = []
    for model_name in ["Logistic Regression", "Linear SVM", "Deep Neural Network", "Residual MLP"]:
        before = metrics[(metrics["model"] == model_name) & (metrics["stage"] == "baseline")]["validation_f1"].iloc[0]
        after = metrics[(metrics["model"] == model_name) & (metrics["stage"] == "refined")]["validation_f1"].iloc[0]
        change = after - before
        direction = "increased" if change > 0 else ("decreased" if change < 0 else "did not change")
        refinement_lines.append(
            f"Validation F1 for {model_name} {direction} from {before:.3f} to {after:.3f} "
            f"after refinement (change {change:+.3f})."
        )
    text = "\n".join([source_line, cross_line, domain_line] + refinement_lines)
    (report_dir / "generated_discussion.tex").write_text(text, encoding="utf-8")

def run_experiment(
    data_dir: str | Path,
    output_dir: str | Path,
    source_domain: str = "books",
    target_domains: list[str] | None = None,
    max_per_class: int = 1500,
    max_features: int = 5000,
    seed: int = 42,
):
    """Run two traditional and two deep models, each before and after refinement."""
    output = Path(output_dir)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    report_dir = Path(__file__).resolve().parents[2] / "report"

    full = load_multidomain_dataset(data_dir)
    available = sorted(full)
    source_domain = source_domain.lower()
    if source_domain not in full:
        raise ValueError(f"Source domain '{source_domain}' not found. Available: {available}")

    if target_domains is None:
        target_domains = ["dvd", "electronics", "kitchen"]
    targets = [d.lower() for d in target_domains if d.lower() in full and d.lower() != source_domain]
    if not targets:
        raise ValueError(f"None of the requested target domains are available. Available: {available}")

    selected_domains = [source_domain] + targets
    full = {domain: full[domain] for domain in selected_domains}
    full_summary = dataset_summary(full)
    sampled = sample_balanced_domains(full, max_per_class=max_per_class, seed=seed)
    sampled_summary = dataset_summary(sampled)
    pd.DataFrame(full_summary).to_csv(output / "dataset_summary_full.csv", index=False)
    pd.DataFrame(sampled_summary).to_csv(output / "dataset_summary.csv", index=False)

    # Exploratory class-balance plot.
    summary_frame = pd.DataFrame(sampled_summary).set_index("domain")
    ax = summary_frame[["positive", "negative"]].plot(kind="bar", figsize=(8, 4))
    ax.set_title("Balanced review sample by product domain")
    ax.set_xlabel("Product domain")
    ax.set_ylabel("Number of reviews")
    ax.legend(["Positive", "Negative"])
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(figures / "domain_class_balance.png", dpi=180, bbox_inches="tight")
    plt.close()

    # Token-frequency plot from the selected source domain.
    feature_counts = {}
    for row in sampled[source_domain]:
        for feature, count in row.features.items():
            feature_counts[feature] = feature_counts.get(feature, 0.0) + count
    top_tokens = sorted(feature_counts.items(), key=lambda item: item[1], reverse=True)[:15]
    token_frame = pd.DataFrame(top_tokens, columns=["token", "count"]).sort_values("count")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(token_frame["token"], token_frame["count"])
    ax.set_title(f"Most frequent features in {source_domain.title()} training domain")
    ax.set_xlabel("Total feature count")
    fig.tight_layout()
    fig.savefig(figures / "top_source_features.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    source_x, source_y = records_to_xy(sampled[source_domain])
    X_train_records, X_test_records, y_train, y_test = train_test_split(
        source_x, source_y, test_size=0.20, random_state=seed, stratify=source_y
    )
    X_train_records, X_valid_records, y_train, y_valid = train_test_split(
        X_train_records, y_train, test_size=0.20, random_state=seed, stratify=y_train
    )

    vectorizer = DictVectorizer(sparse=True, sort=True)
    X_train_counts = vectorizer.fit_transform(X_train_records)
    X_valid_counts = vectorizer.transform(X_valid_records)
    X_test_counts = vectorizer.transform(X_test_records)
    tfidf = TfidfTransformer(sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train_counts)
    X_valid_tfidf = tfidf.transform(X_valid_counts)
    X_test_tfidf = tfidf.transform(X_test_counts)

    k = min(max_features, X_train_tfidf.shape[1])
    selector = SelectKBest(score_func=chi2, k=k)
    X_train = selector.fit_transform(X_train_tfidf, np.asarray(y_train))
    X_valid = selector.transform(X_valid_tfidf)
    X_source_test = selector.transform(X_test_tfidf)

    target_sets = {}
    for domain in targets:
        target_x, target_y = records_to_xy(sampled[domain])
        target_sets[domain] = (selector.transform(tfidf.transform(vectorizer.transform(target_x))), target_y)

    models = {}
    validation_scores = {}
    thresholds = {}
    refinement_rows = []
    model_settings = {}

    # Traditional model 1: Logistic Regression.
    lr_base = make_logistic_regression(1.0)
    lr_base.fit(X_train, y_train)
    models[("Logistic Regression", "baseline")] = lr_base
    validation_scores[("Logistic Regression", "baseline")] = f1_score(
        y_valid, lr_base.predict(X_valid), zero_division=0
    )

    lr_candidates = []
    for c_value in [0.01, 0.1, 1.0, 10.0]:
        candidate = make_logistic_regression(c_value)
        candidate.fit(X_train, y_train)
        score = f1_score(y_valid, candidate.predict(X_valid), zero_division=0)
        lr_candidates.append((score, c_value, candidate))
    lr_best_score, lr_best_c, lr_refined = max(lr_candidates, key=lambda item: item[0])
    models[("Logistic Regression", "refined")] = lr_refined
    validation_scores[("Logistic Regression", "refined")] = lr_best_score
    refinement_rows.extend([
        {"model": "Logistic Regression", "stage": "baseline", "validation_f1": validation_scores[("Logistic Regression", "baseline")], "setting": "C=1.0"},
        {"model": "Logistic Regression", "stage": "refined", "validation_f1": lr_best_score, "setting": f"validation-selected C={lr_best_c}"},
    ])

    # Traditional model 2: Linear SVM.
    svm_base = make_linear_svm(1.0)
    svm_base.fit(X_train, y_train)
    models[("Linear SVM", "baseline")] = svm_base
    validation_scores[("Linear SVM", "baseline")] = f1_score(
        y_valid, svm_base.predict(X_valid), zero_division=0
    )

    svm_candidates = []
    for c_value in [0.01, 0.1, 1.0, 10.0]:
        candidate = make_linear_svm(c_value)
        candidate.fit(X_train, y_train)
        score = f1_score(y_valid, candidate.predict(X_valid), zero_division=0)
        svm_candidates.append((score, c_value, candidate))
    svm_best_score, svm_best_c, svm_refined = max(svm_candidates, key=lambda item: item[0])
    models[("Linear SVM", "refined")] = svm_refined
    validation_scores[("Linear SVM", "refined")] = svm_best_score
    refinement_rows.extend([
        {"model": "Linear SVM", "stage": "baseline", "validation_f1": validation_scores[("Linear SVM", "baseline")], "setting": "C=1.0"},
        {"model": "Linear SVM", "stage": "refined", "validation_f1": svm_best_score, "setting": f"validation-selected C={svm_best_c}"},
    ])

    # Deep models use the same selected source-domain features.
    X_train_dense = X_train.toarray().astype(np.float32)
    X_valid_dense = X_valid.toarray().astype(np.float32)
    y_train_array = np.asarray(y_train, dtype=np.float32)
    y_valid_array = np.asarray(y_valid, dtype=np.float32)

    for model_name, kind in [
        ("Deep Neural Network", "dnn"),
        ("Residual MLP", "residual_mlp"),
    ]:
        for stage in ["baseline", "refined"]:
            predictor, _, settings = train_deep_model(
                kind, stage, X_train_dense, y_train_array,
                X_valid_dense, y_valid_array, seed=seed,
            )
            models[(model_name, stage)] = predictor
            model_settings[(model_name, stage)] = settings

    # Select decision thresholds on validation data for refined models only.
    refinement_rows = []
    for (model_name, stage), model in models.items():
        is_deep = model_name in {"Deep Neural Network", "Residual MLP"}
        X_val = X_valid_dense if is_deep else X_valid
        if stage == "refined":
            threshold, val_f1 = _best_threshold(model, X_val, y_valid)
        else:
            threshold = _default_threshold(model)
            val_f1 = float(f1_score(
                y_valid, _predict_at_threshold(model, X_val, threshold), zero_division=0
            ))
        thresholds[(model_name, stage)] = threshold
        validation_scores[(model_name, stage)] = val_f1

        if model_name == "Logistic Regression":
            setting = "C=1.0" if stage == "baseline" else f"validation-selected C={lr_best_c}"
        elif model_name == "Linear SVM":
            setting = "C=1.0" if stage == "baseline" else f"validation-selected C={svm_best_c}"
        else:
            setting = json.dumps(model_settings[(model_name, stage)], sort_keys=True)
        setting += f"; decision_threshold={threshold:.5f}"
        refinement_rows.append({
            "model": model_name,
            "stage": stage,
            "validation_f1": val_f1,
            "setting": setting,
        })

    # Save the fitted preprocessing pipeline and the four refined model artifacts.
    model_dir = output / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"vectorizer": vectorizer, "tfidf": tfidf, "selector": selector},
        model_dir / "preprocessing.joblib",
    )
    for model_name in ["Logistic Regression", "Linear SVM", "Deep Neural Network", "Residual MLP"]:
        model = models[(model_name, "refined")]
        filename = f"{_safe(model_name)}_refined"
        if model_name in {"Deep Neural Network", "Residual MLP"}:
            import torch
            torch.save(
                {"state_dict": model.model.state_dict(), "input_dim": int(X_train.shape[1])},
                model_dir / f"{filename}.pt",
            )
        else:
            joblib.dump(model, model_dir / f"{filename}.joblib")
    artifact_metadata = {
        f"{name}__{stage}": {
            "decision_threshold": thresholds[(name, stage)],
            "validation_f1": validation_scores[(name, stage)],
        }
        for name, stage in models
    }
    (model_dir / "model_metadata.json").write_text(
        json.dumps(artifact_metadata, indent=2), encoding="utf-8"
    )

    # Majority-class reference baseline, calculated independently for each evaluation set.
    evaluations = [("in_domain", X_source_test, y_test)]
    evaluations.extend(
        (f"cross_domain:{domain}", matrix, labels)
        for domain, (matrix, labels) in target_sets.items()
    )
    rows = []
    refined_confusions = []
    for evaluation, matrix, labels in evaluations:
        majority_label = int(pd.Series(y_train).mode().iloc[0])
        majority_predictions = np.full(len(labels), majority_label, dtype=int)
        majority_scores = majority_predictions.astype(float)
        majority_row = {
            "model": "Majority baseline",
            "stage": "reference",
            "evaluation": evaluation,
            "n_test": len(labels),
            "validation_f1": float("nan"),
            "accuracy": float(accuracy_score(labels, majority_predictions)),
            "precision": float(precision_score(labels, majority_predictions, zero_division=0)),
            "recall": float(recall_score(labels, majority_predictions, zero_division=0)),
            "f1": float(f1_score(labels, majority_predictions, zero_division=0)),
            "roc_auc": float("nan"),
        }
        rows.append(majority_row)

        for (model_name, stage), model in models.items():
            is_deep = model_name in {"Deep Neural Network", "Residual MLP"}
            X_eval = matrix.toarray().astype(np.float32) if is_deep else matrix
            row, y_true, y_pred = _evaluate(
                model_name, stage, evaluation, model, X_eval, labels,
                validation_scores[(model_name, stage)],
                thresholds[(model_name, stage)],
            )
            rows.append(row)
            if stage == "refined":
                refined_confusions.append((model_name, evaluation, y_true, y_pred))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "metrics_summary.csv", index=False)
    metrics.to_json(output / "metrics_summary.json", orient="records", indent=2)
    pd.DataFrame(refinement_rows).to_csv(output / "refinement_comparison.csv", index=False)

    # Confusion matrices for the four refined models only.
    for model_name, evaluation, y_true, y_pred in refined_confusions:
        _save_confusion(
            y_true, y_pred, f"{model_name} — {evaluation}",
            figures / f"confusion_{_safe(model_name)}_{_safe(evaluation)}.png",
        )

    # Adaptation-gap chart: source-domain F1 minus mean target-domain F1.
    refined = metrics[metrics["stage"] == "refined"]
    source_scores = refined[refined["evaluation"] == "in_domain"].set_index("model")["f1"]
    target_scores = refined[refined["evaluation"].str.startswith("cross_domain:")].groupby("model")["f1"].mean()
    gaps = (source_scores - target_scores).sort_values()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(gaps.index, gaps.values)
    ax.axvline(0, linewidth=1)
    ax.set_title("In-domain to cross-domain F1 gap after refinement")
    ax.set_xlabel("Books test F1 minus mean target-domain F1")
    ax.set_ylabel("Refined model")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "model_adaptation_gap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "dataset": "Multi-Domain Sentiment Dataset version 2.0",
        "task": "Cross-domain binary sentiment classification",
        "source_domain": source_domain,
        "target_domains": targets,
        "domains_available": available,
        "max_per_class_per_domain": max_per_class,
        "sampled_documents": {row["domain"]: row["documents"] for row in sampled_summary},
        "source_train": len(y_train),
        "source_validation": len(y_valid),
        "source_test": len(y_test),
        "selected_features": int(X_train.shape[1]),
        "seed": seed,
        "models": ["Logistic Regression", "Linear SVM", "Deep Neural Network", "Residual MLP"],
        "refinement": "Validation-tuned C for traditional models; refined architecture, dropout, weight decay and early stopping for neural models.",
    }
    (output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_report_tables(metrics, sampled_summary, report_dir)
    _write_discussion(metrics, report_dir, refinement_rows)
    return metrics
