#!/usr/bin/env python3
"""Fail closed when the ISCAS draft promotes a scoped result beyond evidence."""

from pathlib import Path
import re


PAPER = Path(__file__).with_name("main.tex")


def between(text: str, begin: str, end: str) -> str:
    assert begin in text and end in text, (begin, end)
    return text.split(begin, 1)[1].split(end, 1)[0]


def main() -> None:
    text = PAPER.read_text(encoding="utf-8")
    abstract = between(text, r"\begin{abstract}", r"\end{abstract}")
    # The main evidence table is intentionally two-column.  Bind its starred
    # environment exactly so later single-column model tables cannot be folded
    # into this claim-critical region.
    main_table = between(text, r"\label{tab:main}", r"\end{table*}")

    # TSBG's production-trace ratio and traffic reduction are CPU-model facts.
    for forbidden in ("2.5338", "29.41", "11.61", "65.20"):
        assert forbidden not in abstract, ("TSBG model fact in abstract", forbidden)
        assert forbidden not in main_table, ("TSBG model fact in main table", forbidden)
    for required in ("12,522,876", "5,124,365", "2.4438", "59.08", "64.25"):
        assert required in abstract, ("missing TSBG G48 RTL fact in abstract", required)
        assert required in main_table, ("missing TSBG G48 RTL fact in main table", required)
    for required in ("matched", "logic-only", "0.0118", "hold and power remain open"):
        assert required in abstract, ("missing TSBG DC boundary in abstract", required)
    for required in ("C2/TSBG", "1,920 fixed ep34 real-activity workloads",
                     "same G48 engine",
                     "identical 383-cycle preloads excluded",
                     "no full-FC/system claim", "0.0118"):
        assert required in main_table, ("missing TSBG G48 boundary", required)
    assert "full-FC or system speedup" in text
    for required in ("all 40 captured samples", "all 12 FC1 layers", "four FC2 layers",
                     "first/middle/last aligned B4", "286 all-zero workloads",
                     "2.3814--2.5458", "8,774,304", "3,136,608",
                     "1,343 workloads improve", "seven are slower", "0.9935"):
        assert required in text, ("missing TSBG distribution boundary", required)
    assert "excludes eight FC2 layers above G48" in text
    assert re.search(r"seven\s+nonempty microbenchmarks are marginally slower", text)
    assert re.search(r"all naturally\s+nonzero codes in this measured population are \$\+1\$", text)
    assert "1,917+3 same-simulation-image" in text
    assert "failed parent attempt remains non-citable" in text
    assert "combines 1,917 inherited logs with three successor logs" in text
    assert "same compiled simulation image" in text
    assert "M2053 is not promoted as a successful" in text
    assert "individually valid logs are inherited with explicit lineage" in text
    assert "synthetic recovery phase checks signed products" in text
    assert "Deterministic INT8 weights" in text
    assert "do not affect scheduling" in text

    # The C1 cycle ratio must remain visibly tagged as a model result.
    assert r"1.6945$\times$" in main_table
    assert r"\modeltag" in main_table

    # C2's modest cycle result must travel with the equal-bandwidth area result.
    for required in ("1,913", "1,945", "1.0167", "4.5411", "77.61"):
        assert required in abstract, ("missing C2 denominator in abstract", required)
        assert required in main_table, ("missing C2 denominator in main table", required)
    assert r"K1$\times$8" in main_table
    assert r"directed cycles (1.0167$\times$)\vcstag" in main_table
    assert "directed-VCS throughput/DC logic-cell area" in main_table
    assert "logic-only" in main_table
    assert "five directed" in abstract
    assert "logic-only" in abstract
    assert "directed-throughput/logic-area" in abstract
    assert r"ten \texttt{zurich\_city\_09\_a} samples" in abstract
    assert r"ten \texttt{zurich\_city\_09\_a} ep34 samples" in text
    assert "mapped-to-mapped Formality compare points" in abstract
    assert "one admitted campaign" in main_table
    assert "131,086" in main_table and "585,479" in main_table
    assert "separately evaluated mixed-precision deployment" in abstract
    assert "not a full" in text and "common-charge ledger" in text

    # Freeze the selected evaluation identity and the no-system-speedup boundary.
    for required in ("Motion C12 ep34", "1.199514", "5.6709"):
        assert required in text, ("missing workload identity", required)
    for required in (r"\texttt{sn2\_q}", "runtime-bypassed and never called",
                     "invokes 93", r"48 $T{=}2$, 45 $T{=}10$",
                     r"\texttt{attn\_sn}",
                     "leaving 81 graph-live services under that fixed call graph",
                     r"36 $T{=}2$, 45 $T{=}10$",
                     "All 93 captured ATLIF outputs are binary"):
        assert required in text, ("missing 105/93/81 split", required)
    assert "12 of those are output-dead" not in text
    for required in (r"$\alpha{=}0.125$", "K as its value carrier",
                     "hardware quantization", "disabled",
                     "Only the separately evaluated deployment candidate",
                     "Q7 round-to-nearest", "Q1.7 gates"):
        assert required in text, ("missing baseline/candidate attention split", required)
    assert "The frozen H60 attention path uses Motion-XOR scoring, Q7" not in text
    assert "make no whole-network speedup claim" in abstract
    assert "two architectural contributions" in text
    assert "whole-network RTL speedup" in text

    # M2045 admits only a frozen-population deployment-subset accuracy row.
    # Its historical baseline and deterministic candidate use different GPU
    # backend flags, so the negative AEE delta must never become a causal
    # quantization-improvement claim.
    accuracy_table = between(text, r"\label{tab:accuracy}", r"\end{table}")
    for required in ("1.199514", "1.197367", "-0.002147", "5.412808",
                     "5.328834"):
        assert required in accuracy_table, ("missing M2045 accuracy fact", required)
    for required in ("825 frames", "18 sequences", "48,152,523 valid pixels",
                     "hardware-order attention", "four C1 Conv3x3",
                     "four decoder", "Other operators remain at checkpoint precision",
                     "enabled TF32/cuDNN benchmarking", "not a causal accuracy"):
        assert required in text, ("missing M2045 claim boundary", required)
    assert re.search(r"825-frame local DSEC\s+validation set", abstract)
    assert "AEE compatibility gate" in abstract
    assert "10 of 18 sequences regress" in text
    for forbidden in ("quantization improves", "full-network INT8 speedup",
                      "end-to-end INT8 speedup"):
        assert forbidden not in text, ("forbidden M2045 promotion", forbidden)

    # Bind two easy-to-misstate physical scopes and the matched TSBG axis.
    assert r"1.695$\times$" not in text
    assert "pre- and post-hold-repair mapped netlists, not RTL-to-gate" in main_table
    assert r"\texttt{SCHEDULE\_MODE=0/1}" in text
    assert "same B4 RTL" in text
    for required in ("prelayout", "ideal clocks", "ZeroWireload",
                     "C1 area includes nine SRAM macros",
                     "C2 and C3 are logic-only",
                     "backend-mismatched baseline",
                     "AEE (average endpoint error) is the preselected"):
        assert required in text, ("missing evaluation-contract qualifier", required)

    # Submission prose must not leak internal milestone IDs.  Keep the abstract
    # compact enough for a circuit-conference paper rather than turning it into
    # an evidence ledger.
    for forbidden in ("M2045", "M917", "M1454", "valid825"):
        assert forbidden not in text, ("internal milestone term in paper", forbidden)
    abstract_words = re.findall(r"[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*", abstract)
    assert len(abstract_words) <= 250, ("abstract too long", len(abstract_words))

    print("PASS_ISCAS2027_CLAIM_BOUNDARY_LINTER")


if __name__ == "__main__":
    main()
