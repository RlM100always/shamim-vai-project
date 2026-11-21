from __future__ import annotations

from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"


def main() -> None:
    count_file = REPORT / "generated_wordcount.tex"
    # Keep the included field to one token while counting, so the final number
    # does not change the total word count.
    count_file.write_text("0\n", encoding="utf-8")
    result = subprocess.run(
        ["texcount", "-inc", "-incbib", "-sum", "-total", "main.tex"],
        cwd=REPORT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout
    match = re.search(r"Sum count:\s*([\d,]+)", output)
    if not match:
        match = re.search(r"Words in text:\s*([\d,]+)", output)
    if not match:
        raise RuntimeError(f"Could not read word count from texcount:\n{output}")
    count = int(match.group(1).replace(",", ""))
    if count > 3000:
        raise RuntimeError(f"Report has {count} words; the assignment limit is 3,000.")
    count_file.write_text(f"{count:,}\n", encoding="utf-8")
    print(f"Report word count: {count}")


if __name__ == "__main__":
    main()
