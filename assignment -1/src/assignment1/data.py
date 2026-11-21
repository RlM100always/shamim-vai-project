from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from html import unescape
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np


@dataclass
class ReviewRecord:
    domain: str
    features: dict[str, float]
    label: int
    source_file: str


def normalise_label(value: str | None) -> int | None:
    if value is None:
        return None
    value = value.strip().lower().replace("#label#:", "").strip("#")
    if value in {"positive", "pos", "+1", "1", "1.0"}:
        return 1
    if value in {"negative", "neg", "-1", "0", "0.0"}:
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
    """Parse the official v2.0 format: feature:count ... #label#:label."""
    features: dict[str, float] = {}
    label = None
    for token in line.strip().split():
        if token.startswith("#label#:"):
            label = normalise_label(token.split(":", 1)[1])
            continue
        if ":" not in token:
            continue
        feature, raw_count = token.rsplit(":", 1)
        if not feature or feature.startswith("#"):
            continue
        try:
            count = float(raw_count)
        except ValueError:
            continue
        if count > 0:
            features[feature] = features.get(feature, 0.0) + count
    return features, label


def _read_sparse_file(path: Path, domain: str, label_hint: int | None = None) -> list[ReviewRecord]:
    records = []
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line_no, line in enumerate(stream, 1):
            features, parsed_label = parse_sparse_review_line(line)
            label = parsed_label if parsed_label is not None else label_hint
            if features and label in (0, 1):
                records.append(ReviewRecord(domain, features, int(label), f"{path.name}:{line_no}"))
    return records


def _xml_review_features(block: str) -> dict[str, float]:
    text = ""
    try:
        root = ET.fromstring(block)
        for node in root.iter():
            if node.tag.lower().endswith("review_text") or node.tag.lower().endswith("text"):
                text = " ".join(node.itertext())
                break
    except ET.ParseError:
        match = re.search(r"<review_text>(.*?)</review_text>", block, flags=re.I | re.S)
        if match:
            text = re.sub(r"<[^>]+>", " ", match.group(1))
    text = unescape(re.sub(r"<[^>]+>", " ", text)).lower()
    tokens = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    return dict(Counter(tokens))


def _read_xml_file(path: Path, domain: str, label: int) -> list[ReviewRecord]:
    content = path.read_text(encoding="utf-8", errors="replace")
    blocks = re.findall(r"<review\b.*?</review>", content, flags=re.I | re.S)
    if not blocks:
        blocks = [line for line in content.splitlines() if "<review" in line.lower()]
    records = []
    for index, block in enumerate(blocks, 1):
        features = _xml_review_features(block)
        if features:
            records.append(ReviewRecord(domain, features, label, f"{path.name}:{index}"))
    return records


def canonical_domain(name: str) -> str:
    key = name.lower().replace(" ", "_")
    if key in {"kitchen_&_housewares", "kitchen_and_housewares", "kitchen-housewares"}:
        return "kitchen"
    return key


def _looks_like_domain_folder(path: Path) -> bool:
    names = {p.name.lower() for p in path.iterdir() if p.is_file()}
    expected = {
        "processed.review.balanced", "processed.review",
        "positive.review", "negative.review",
    }
    return bool(names.intersection(expected))


def find_domain_directories(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {data_dir}")
    found = []
    for path in data_dir.rglob("*"):
        if path.is_dir() and _looks_like_domain_folder(path):
            found.append(path)
    if _looks_like_domain_folder(data_dir):
        found.insert(0, data_dir)
    return sorted(set(found), key=lambda p: str(p).lower())


def load_multidomain_dataset(data_dir: str | Path) -> dict[str, list[ReviewRecord]]:
    """Load labelled documents from the official processed ACL archive."""
    root = Path(data_dir).expanduser().resolve()
    domain_dirs = find_domain_directories(root)
    if not domain_dirs:
        raise FileNotFoundError(
            f"No domain folders found under {root}. Expected processed.review.balanced "
            "or processed.review files in domain directories."
        )

    dataset: dict[str, list[ReviewRecord]] = {}
    for folder in domain_dirs:
        domain = canonical_domain(folder.name)
        records: list[ReviewRecord] = []

        # Prefer the official preprocessed feature-count files over the XML files.
        for filename in ("processed.review.balanced", "processed.review"):
            candidate = folder / filename
            if candidate.exists():
                records = _read_sparse_file(candidate, domain)
                if records:
                    break

        if not records:
            positive = folder / "positive.review"
            negative = folder / "negative.review"
            if positive.exists():
                records.extend(_read_sparse_file(positive, domain, label_hint=1))
                if not records:
                    records.extend(_read_xml_file(positive, domain, 1))
            if negative.exists():
                records.extend(_read_sparse_file(negative, domain, label_hint=0))
                if not any(r.label == 0 for r in records):
                    records.extend(_read_xml_file(negative, domain, 0))

        if records:
            if domain in dataset:
                dataset[domain].extend(records)
            else:
                dataset[domain] = records

    if not dataset:
        raise ValueError(
            "Domain folders were found, but no labelled documents could be parsed. "
            "Check that processed_acl.tar.gz was downloaded and extracted."
        )
    return dataset


def sample_balanced_domains(
    dataset: dict[str, list[ReviewRecord]],
    max_per_class: int = 1000,
    seed: int = 42,
) -> dict[str, list[ReviewRecord]]:
    """Deterministically sample up to max_per_class reviews from each class/domain."""
    rng = np.random.default_rng(seed)
    sampled = {}
    for domain, records in sorted(dataset.items()):
        indices_by_class = {
            label: np.array([i for i, row in enumerate(records) if row.label == label])
            for label in (0, 1)
        }
        # Use equal counts from both classes, capped at max_per_class.
        n = min(max_per_class, len(indices_by_class[0]), len(indices_by_class[1]))
        if n == 0:
            continue
        chosen = []
        for label in (0, 1):
            indices = rng.choice(indices_by_class[label], size=n, replace=False)
            chosen.extend(records[int(i)] for i in indices)
        rng.shuffle(chosen)
        sampled[domain] = chosen
    return sampled


def dataset_summary(dataset: dict[str, list[ReviewRecord]]) -> list[dict[str, int | str]]:
    rows = []
    for domain, records in sorted(dataset.items()):
        rows.append({
            "domain": domain,
            "documents": len(records),
            "positive": sum(r.label == 1 for r in records),
            "negative": sum(r.label == 0 for r in records),
            "unique_features": len({feature for row in records for feature in row.features}),
        })
    return rows


def records_to_xy(records: list[ReviewRecord]):
    return [row.features for row in records], [row.label for row in records]
