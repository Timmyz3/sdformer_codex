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
    evaluation = between(text, r"\section{Evaluation}", r"\section{Related Work}")
    before_evaluation = text.split(r"\section{Evaluation}", 1)[0]
    after_evaluation = text.split(r"\section{Related Work}", 1)[1]
    nwords = abstract_words(abstract)
    assert 100 <= nwords <= 250, ("abstract word count", nwords)

    # This brief admits C1 and C2/TSBG component evidence but no system claim.
    for forbidden in (
        "4.76", "1.770", "0.0118 W", "99.4", "system FPS", "1.869",
        "2.230", "full-network speedup", "whole-network speedup of",
    ):
        assert forbidden not in abstract, ("illegal claim in abstract", forbidden)

    for required in (
        "1.6945", "40.99", "51.84", "166", "514", "setup/hold",
        "2,880", "1.8345", "45.49", "58.13", "0.0118", "4.541", "77.61",
        "prelayout", "whole-network",
    ):
        assert required in abstract, ("missing admitted component fact in abstract", required)

    assert r"\modeltag" in text
    assert "cycle-model" in text or "cycle model" in text
    assert "single-blind" in text
    assert r"\documentclass[journal]{IEEEtran}" in text
    assert "no whole-network speedup is inferred" in abstract
    assert "K1$\\times$8" in abstract
    assert r"\newcommand{\dctag}{\textsuperscript{[DC]}}" in text
    assert r"\newcommand{\pttag}{\textsuperscript{[DC/PT]}}" in text
    assert "never multiplied" in text.replace("\n", " ")
    assert "prelayout" in abstract
    assert "compatibility check" in text
    assert "hold closure and SRAM-inclusive power remain open" in text.replace("\n", " ")
    assert "ungated logic estimates" in text
    assert "33.65" in evaluation and "65.86" in evaluation
    assert "logic-only" in text
    assert "not deployed SRAM area" in text
    assert "39.72\\% area reduction" in text
    assert "1.687$\\times$" in text
    assert "not an integrated macro placement" in text
    assert "152,898" not in text
    assert "M1665" not in text
    assert "M2063" not in text
    assert "0.0118 W" not in text
    assert "docs/359" not in text
    assert "parent masks and psums" not in text
    assert "parent-product scratch" in text
    assert "18{,}432 B / 9 macros" in text
    assert "214{,}912 B / 105 macros / 0.988 mm$^2$" in text
    assert "complete ledger is an area model, not integrated PPA" in text
    assert "deterministic directed INT8 verification weights" in text
    assert "Naturally nonzero descriptors" in text
    assert "all 12 FC1 and 12 FC2 layer identities" in text
    assert "not full FC" in text
    for required in (
        "11.16 million", "313.604", "150.234", "2.0874", "52.09",
        "64.68", "2.0807--2.0968", "2.1766", "1.9005", "0.99755",
        "VCS-calibrated FC component model", "observed-envelope sensitivity",
        "2.0874$\\times$ ratio of sums", "same-area", "779,040", "780,000",
        "99.877", "3.10\\%",
    ):
        assert required in evaluation, ("missing full-population boundary", required)
    assert "2.0874" not in abstract
    assert "2.0874" not in before_evaluation
    assert "2.0874" not in after_evaluation
    assert "whole-network execution" in text
    assert "not a formal bound" in text
    for required in (
        "Pre-read causality", "2,304/2,304/576", "4,608 signed products",
        "24 commits with zero mismatch", "not matched area or energy",
        "separate from the",
    ):
        assert required in evaluation, ("causal ablation boundary", required)
    assert "2,304/2,304/576" not in abstract
    assert "Prosperity already discovers subset/prefix parents" in text
    assert "FireFly-T already broadcasts" in text

    keywords = between(text, r"\begin{IEEEkeywords}", r"\end{IEEEkeywords}")
    nkw = len([k.strip() for k in keywords.replace("\n", " ").split(",") if k.strip()])
    assert nkw >= 5, ("need >=5 keywords", nkw)

    print("TCAS-II claim-boundary check passed; abstract_words=", nwords, "keywords=", nkw)


if __name__ == "__main__":
    main()
