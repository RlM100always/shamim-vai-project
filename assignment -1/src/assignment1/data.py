from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass
class ReviewRecord:
    domain: str
    features: dict[str, float]
    label: int
    source_file: str


def _normalise_label(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip().lower().strip("#")
    if value in {"positive", "pos", "+1", "1.0", "1"}:
        return 1
    if value in {"negative", "neg", "-1", "0.0", "0"}:
        return 0
    try:
        rating = float(value)
    except ValueError:
        return None
    if rating >= 4:
        return 1
    if rating <= 2:
        return 0
    return None


def parse_sparse_review_line(line: str) -> tuple[dict[str, float], int | None]:
    """Parse v2.0 preprocessed data: feature:count ... #label#:label."""
    features: dict[str, float] = {}
    label: int | None = None
    for token in line.strip().split():
        if token.startswith("#label#:"):
            label = _normalise_label(token.split(":", 1)[1])
            continue
        if ":" not in token:
            continue
        feature, raw_count = token.rsplit(":", 1)
        if not feature or feature.startswith("#label#"):
            continue
        try:
            count = float(raw_count)
        except ValueError:
            continue
        if count > 0:
            features[feature] = features.get(feature, 0.0) + count
    return features, label


def _read_feature_file(path: Path, label_hint: int | None = None, domain: str = "unknown") -> list[ReviewRecord]:
    records: list[ReviewRecord] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            features, parsed_label = parse_sparse_review_line(line)
            label = parsed_label if parsed_label is not None else label_hint
            if features and label in (0, 1):
                records.append(ReviewRecord(domain, features, int(label), f"{path.name}:{line_number}"))
    return records


def _domain_directories(data_dir: Path) -> list[Path]:
    candidates = [data_dir]
    candidates.extend(p for p in data_dir.iterdir() if p.is_dir())
    found: list[Path] = []
    for base in candidates:
        if not base.exists() or not base.is_dir():
            continue
        for child in base.iterdir():
            if not child.is_dir():
                continue
            names = {p.name.lower() for p in child.iterdir() if p.is_file()}
            if names.intersection({"positive.review", "negative.review", "processed.review", "processed.review.balanced"}):
                found.append(child)
    own_names = {p.name.lower() for p in data_dir.iterdir() if p.is_file()} if data_dir.exists() else set()
    if own_names.intersection({"positive.review", "negative.review", "processed.review", "processed.review.balanced"}):
        found.insert(0, data_dir)
    return sorted(set(found), key=lambda p: p.name.lower())


def load_multidomain_dataset(data_dir: str | Path) -> dict[str, list[ReviewRecord]]:
    """Load labelled sparse reviews from the processed_acl archive extraction."""
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}. Run scripts/download_dataset.py first.")
    nested = root / "processed_acl"
    if nested.is_dir():
        root = nested
    domain_dirs = _domain_directories(root)
    if not domain_dirs:
        raise FileNotFoundError(
            f"No labelled domain folders found under {root}. Expected positive.review / negative.review "
            "or processed.review.balanced files."
        )

    dataset: dict[str, list[ReviewRecord]] = {}
    for domain_dir in domain_dirs:
        domain = domain_dir.name.lower().replace(" ", "_")
        records: list[ReviewRecord] = []
        positive = domain_dir / "positive.review"
        negative = domain_dir / "negative.review"
        if positive.exists() or negative.exists():
            if positive.exists():
                records.extend(_read_feature_file(positive, label_hint=1, domain=domain))
            if negative.exists():
                records.extend(_read_feature_file(negative, label_hint=0, domain=domain))
        else:
            for filename in ("processed.review.balanced", "processed.review"):
                candidate = domain_dir / filename
                if candidate.exists():
                    records.extend(_read_feature_file(candidate, domain=domain))
                    if records:
                        break
        if records:
            dataset[domain] = records
    if not dataset:
        raise ValueError(
            "Domain folders were found, but no labelled sparse reviews could be parsed. "
            "The project expects processed_acl, not the raw XML archive."
        )
    return dataset


def dataset_summary(dataset: dict[str, list[ReviewRecord]]) -> list[dict[str, int | str]]:
    summary = []
    for domain, records in sorted(dataset.items()):
        summary.append({
            "domain": domain,
            "documents": len(records),
            "positive": sum(r.label == 1 for r in records),
            "negative": sum(r.label == 0 for r in records),
            "unique_features": len({f for r in records for f in r.features}),
        })
    return summary


def records_to_xy(records: Iterable[ReviewRecord]) -> tuple[list[dict[str, float]], list[int]]:
    rows = list(records)
    return [r.features for r in rows], [r.label for r in rows]
