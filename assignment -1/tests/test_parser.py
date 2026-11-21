from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assignment1.data import (
    parse_sparse_review_line,
    load_multidomain_dataset,
    sample_balanced_domains,
)


def test_parse_sparse_review_line():
    features, label = parse_sparse_review_line(
        "great:2 product:1 very_good:3 #label#:positive"
    )
    assert features == {"great": 2.0, "product": 1.0, "very_good": 3.0}
    assert label == 1


def test_parse_numeric_binary_labels():
    _, positive = parse_sparse_review_line("good:2 #label#:1")
    _, negative = parse_sparse_review_line("poor:1 #label#:-1")
    assert positive == 1
    assert negative == 0


def test_load_domain_positive_negative_files(tmp_path):
    domain = tmp_path / "books"
    domain.mkdir()
    (domain / "positive.review").write_text(
        "great:2 product:1\n", encoding="utf-8"
    )
    (domain / "negative.review").write_text(
        "bad:1 broken:2\n", encoding="utf-8"
    )
    dataset = load_multidomain_dataset(tmp_path)
    assert len(dataset["books"]) == 2
    assert {row.label for row in dataset["books"]} == {0, 1}


def test_balanced_sampling_is_reproducible():
    from assignment1.data import ReviewRecord
    rows = {
        "books": [
            ReviewRecord("books", {"good": 1.0}, 1, "p1"),
            ReviewRecord("books", {"great": 1.0}, 1, "p2"),
            ReviewRecord("books", {"bad": 1.0}, 0, "n1"),
        ]
    }
    first = sample_balanced_domains(rows, max_per_class=10, seed=7)
    second = sample_balanced_domains(rows, max_per_class=10, seed=7)
    assert len(first["books"]) == 2
    assert sum(row.label == 1 for row in first["books"]) == 1
    assert sum(row.label == 0 for row in first["books"]) == 1
    assert [row.source_file for row in first["books"]] == [
        row.source_file for row in second["books"]
    ]
