from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assignment1.data import parse_sparse_review_line, load_multidomain_dataset


def test_parse_sparse_review_line():
    features, label = parse_sparse_review_line("great:2 product:1 very_good:3 #label#:positive")
    assert features == {"great": 2.0, "product": 1.0, "very_good": 3.0}
    assert label == 1


def test_load_domain_positive_negative_files(tmp_path):
    domain = tmp_path / "books"
    domain.mkdir()
    (domain / "positive.review").write_text("great:2 product:1\n", encoding="utf-8")
    (domain / "negative.review").write_text("bad:1 broken:2\n", encoding="utf-8")
    dataset = load_multidomain_dataset(tmp_path)
    assert len(dataset["books"]) == 2
    assert {r.label for r in dataset["books"]} == {0, 1}
