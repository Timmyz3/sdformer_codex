#!/usr/bin/env python3
"""Fail closed on TCAS-II's exact five-page/last-column format contract."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


def run(*args: str) -> str:
    return subprocess.run(args, check=True, text=True, capture_output=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", nargs="?", default="build/main.pdf")
    parser.add_argument(
        "--draft-underfill-ok",
        action="store_true",
        help="report but do not fail the 4.5-page content-fill gate",
    )
    args = parser.parse_args()
    pdf = Path(args.pdf)
    assert pdf.is_file(), pdf

    info = run("pdfinfo", str(pdf))
    pages_line = next(line for line in info.splitlines() if line.startswith("Pages:"))
    pages = int(pages_line.split(":", 1)[1])

    with tempfile.NamedTemporaryFile(suffix=".html") as tmp:
        subprocess.run(
            ["pdftotext", "-f", "5", "-l", "5", "-bbox-layout", str(pdf), tmp.name],
            check=True,
        )
        root = ET.parse(tmp.name).getroot()

    page = next(element for element in root.iter() if element.tag.endswith("page"))
    page_width = float(page.attrib["width"])
    split_x = page_width / 2.0
    words = []
    for word in (element for element in page.iter() if element.tag.endswith("word")):
        words.append(
            {
                "text": word.text or "",
                "x": float(word.attrib["xMin"]),
                "y0": float(word.attrib["yMin"]),
                "y1": float(word.attrib["yMax"]),
            }
        )

    semantic = [w for w in words if w["y0"] > 45.0]
    left = [w for w in semantic if w["x"] < split_x]
    right = [w for w in semantic if w["x"] >= split_x]
    right_ordered = sorted(right, key=lambda w: (w["y0"], w["x"]))
    right_text = " ".join(w["text"] for w in right_ordered)
    left_text = " ".join(w["text"] for w in sorted(left, key=lambda w: (w["y0"], w["x"])))
    left_ymax = max((w["y1"] for w in left), default=0.0)

    checks = {
        "exactly_five_pages": pages == 5,
        "page5_right_begins_references": bool(right_ordered)
        and right_ordered[0]["text"].upper() == "REFERENCES",
        "page5_right_has_no_body_heading": not any(
            token in right_text.upper()
            for token in ("CONCLUSION", "DISCUSSION", "LIMITATIONS", "EVALUATION")
        ),
        "page5_left_has_no_references": "REFERENCES" not in left_text.upper(),
        # A conservative mechanical proxy for filling the content-only column.
        "page5_left_content_reaches_650pt": left_ymax >= 650.0,
    }
    result = {
        "status": "PASS_TCASII_SUBMISSION_PDF"
        if all(checks.values())
        else "FAIL_TCASII_SUBMISSION_PDF",
        "pdf": str(pdf),
        "pages": pages,
        "page5_left_content_ymax_pt": left_ymax,
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    hard = {k: v for k, v in checks.items() if k != "page5_left_content_reaches_650pt"}
    assert all(hard.values()), result
    if not args.draft_underfill_ok:
        assert checks["page5_left_content_reaches_650pt"], result


if __name__ == "__main__":
    main()
