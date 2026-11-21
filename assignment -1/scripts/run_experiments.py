from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from assignment1.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Applied AI Assignment 1 experiments.")
    parser.add_argument("--data-dir", default=str(ROOT / "data" / "raw" / "processed_acl"))
    parser.add_argument("--output-dir", default=str(ROOT / "results"))
    parser.add_argument("--source-domain", default="books")
    parser.add_argument("--target-domains", nargs="*", default=None)
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = run_experiment(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        source_domain=args.source_domain,
        target_domains=args.target_domains,
        max_features=args.max_features,
        epochs=args.epochs,
        seed=args.seed,
    )
    print(metrics.to_string(index=False))
    print(f"Saved results to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
