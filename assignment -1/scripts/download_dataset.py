from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
SOURCES = {
    "processed_acl": "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz",
}


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe archive path rejected: {member.name}")
        tar.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the official JHU processed ACL dataset.")
    parser.add_argument("--variant", choices=sorted(SOURCES), default="processed_acl")
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / f"{args.variant}.tar.gz"
    if not archive.exists():
        print(f"Downloading from {SOURCES[args.variant]} ...")
        urllib.request.urlretrieve(SOURCES[args.variant], archive)
    destination = DATA_DIR / args.variant
    destination.mkdir(parents=True, exist_ok=True)
    safe_extract(archive, destination)
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
    print(f"Dataset extracted to: {destination}")


if __name__ == "__main__":
    main()
