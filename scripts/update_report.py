"""Regenerate the results block of REPRODUCTION.md (between the RESULTS markers)
from the JSON files in results/.

    python scripts/update_report.py
"""
import contextlib
import io
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aggregate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def capture(argv):
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["aggregate.py", "--results", os.path.join(ROOT, "results")] + argv
    try:
        with contextlib.redirect_stdout(buf):
            aggregate.main()
    finally:
        sys.argv = old
    return buf.getvalue().strip()


def main():
    block = "\n\n".join([
        "## 0. Results\n",
        capture([]),
        capture(["--ablations", "mnist"]),
    ])
    path = os.path.join(ROOT, "REPRODUCTION.md")
    text = open(path).read()
    new = f"<!-- RESULTS -->\n{block}\n<!-- /RESULTS -->"
    if "<!-- /RESULTS -->" in text:
        text = re.sub(r"<!-- RESULTS -->.*?<!-- /RESULTS -->", lambda m: new, text, flags=re.S)
    else:
        text = text.replace("<!-- RESULTS -->", new)
    open(path, "w").write(text)
    print("updated", path)


if __name__ == "__main__":
    main()
