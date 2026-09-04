#!/usr/bin/env python3
"""Fail closed when the TCAS-II brief promotes a scoped result beyond evidence."""

from pathlib import Path
import re

PAPER = Path(__file__).with_name("main.tex")


def between(text: str, begin: str, end: str) -> str:
    assert begin in text and end in text, (begin, end)
    return text.split(begin, 1)[1].split(end, 1)[0]


def abstract_words(abstract: str) -> int:
    plain = re.sub(r"\\[a-zA-Z]+(\[[^\]]*\])?(\{[^}]*\})?", " ", abstract)
    plain = re.sub(r"[{}$\\]", " ", plain)
    return len(plain.split())


def main() -> None:
    text = PAPER.read_text(encoding="utf-8")
    abstract = between(text, r"\begin{abstract}", r"\end{abstract}")
    nwords = abstract_words(abstract)
    assert 100 <= nwords <= 250, ("abstract word count", nwords)

    # This brief is C1-only.  TSBG/C2 headline numbers must not enter abstract.
    for forbidden in (
        "2.4438", "12,522,876", "5,124,365", "4.5411", "1,913", "1,945",
        "0.0118", "77.61", "64.25", "59.08", "4.76", "1.770", "0.0118 W",
        "99.4", "system FPS", "1.869", "2.230",
    ):
        assert forbidden not in abstract, ("C2/TSBG or illegal fact in abstract", forbidden)

    for required in (
        "1.6945", "40.99", "51.84", "166", "514", "16", "549",
        "29.08", "22.07", "prelayout", "whole-network",
    ):
        assert required in abstract, ("missing C1 fact in abstract", required)

    assert r"\modeltag" in text
    assert "cycle-model" in text or "cycle model" in text
    assert "single-blind" in text
    assert r"\documentclass[journal]{IEEEtran}" in text
    assert "not claimed here" in text or "are not claimed" in text
    assert "no whole-network speedup claim" in abstract
    assert "K1$\\times$8" not in abstract
    assert "never multiplied" in text.replace("\n", " ")
    assert "prelayout" in abstract
    assert "compatibility gate" in text
    assert "10 of 18 sequences regress" in text
    assert "152,898" not in text
    assert "M1665" not in text
    assert "M2063" not in text
    assert "0.0118 W" not in text
    assert "docs/359" not in text

    keywords = between(text, r"\begin{IEEEkeywords}", r"\end{IEEEkeywords}")
    nkw = len([k.strip() for k in keywords.replace("\n", " ").split(",") if k.strip()])
    assert nkw >= 5, ("need >=5 keywords", nkw)

    print("TCAS-II claim-boundary check passed; abstract_words=", nwords, "keywords=", nkw)


if __name__ == "__main__":
    main()
