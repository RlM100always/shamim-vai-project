from __future__ import annotations

import argparse
from pathlib import Path
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw"
SOURCE = "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/processed_acl.tar.gz"


def download_file(url: str, destination: Path, attempts: int = 3) -> None:
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "AppliedAI-coursework/1.0"}
            )
            with urllib.request.urlopen(request, timeout=90) as response, destination.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
            if destination.stat().st_size < 1024 * 1024:
                raise RuntimeError("Downloaded archive is unexpectedly small.")
            return
        except Exception:
            destination.unlink(missing_ok=True)
            if attempt == attempts:
                raise
            time.sleep(attempt * 3)


def safe_extract(archive: Path, destination: Path) -> None:
    destination_resolved = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            target = (destination / member.name).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise RuntimeError(f"Unsafe archive path rejected: {member.name}")
            if member.issym() or member.islnk():
                raise RuntimeError(f"Archive links are not allowed: {member.name}")
        tar.extractall(destination, members=members)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download the official Multi-Domain Sentiment Dataset v2.0 ACL archive."
    )
    parser.add_argument("--keep-archive", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive = DATA_DIR / "processed_acl.tar.gz"
    destination = DATA_DIR / "processed_acl"

    if not archive.exists():
        print(f"Downloading official v2.0 archive: {SOURCE}")
        download_file(SOURCE, archive)
    else:
        print(f"Using existing archive: {archive}")

    destination.mkdir(parents=True, exist_ok=True)
    safe_extract(archive, destination)
    if not args.keep_archive:
        archive.unlink(missing_ok=True)
    print(f"Dataset extracted to: {destination}")


if __name__ == "__main__":
    main()
