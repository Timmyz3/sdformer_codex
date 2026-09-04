#!/usr/bin/env python3
"""Audit how the official Prosperity artifact aggregates headline speedups.

This is an evidence extractor, not an H67 performance model.  It reads the
official OOXML workbook without third-party Python packages, recomputes the
per-workload ratios, and publishes both arithmetic and geometric means so the
paper cannot silently choose an averaging convention.
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[3]
INPUT = Path(
    "/home/zhumd/work/literature_artifacts/Prosperity/reference/time_reference.xlsx"
)
INPUT_SHA256 = "47a05d06a0e762b9a67490875803441eac2bcec9a24a14576896f945452ba563"
OFFICIAL_COMMIT = "6ee1c6f1cb419fcf942f2eda63db84ca28248f4b"
OUTPUT = ROOT / "hw_autoresearch_nts07/results/m526_prosperity_headline_method_audit_r2_20260827"
OFFICIAL_REPO = Path("/home/zhumd/work/literature_artifacts/Prosperity")
STRICT_NS = "http://purl.oclc.org/ooxml/spreadsheetml/main"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_grid(path):
    ns = "{%s}" % STRICT_NS
    with ZipFile(str(path), "r") as archive:
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        shared = [
            "".join(node.text or "" for node in item.iter(ns + "t"))
            for item in shared_root.findall(ns + "si")
        ]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
    grid = {}
    formulas = {}
    for cell in sheet.iter(ns + "c"):
        reference = cell.attrib["r"]
        value_node = cell.find(ns + "v")
        if value_node is None:
            continue
        value = value_node.text
        if cell.attrib.get("t") == "s":
            value = shared[int(value)]
        grid[reference] = value
        formula = cell.find(ns + "f")
        if formula is not None:
            formulas[reference] = formula.text
    return grid, formulas


def geometric_mean(values):
    require(values and all(value > 0 for value in values), "geomean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def arithmetic_mean(values):
    require(values, "mean requires nonempty values")
    return sum(values) / len(values)


def ratio_of_sums(numerators, denominators):
    require(len(numerators) == len(denominators) and numerators,
            "ratio-of-sums requires matched nonempty populations")
    require(sum(denominators) > 0, "ratio-of-sums denominator must be positive")
    return sum(numerators) / sum(denominators)


def main():
    require(INPUT.is_file() and not INPUT.is_symlink(), "official workbook missing or symlinked")
    require(sha256(INPUT) == INPUT_SHA256, "official workbook SHA drift")
    require(OFFICIAL_REPO.is_dir() and not OFFICIAL_REPO.is_symlink(),
            "official repository missing or symlinked")
    observed_head = subprocess.check_output(
        ["git", "-C", str(OFFICIAL_REPO), "rev-parse", "HEAD"],
        universal_newlines=True,
    ).strip()
    observed_status = subprocess.check_output(
        ["git", "-C", str(OFFICIAL_REPO), "status", "--porcelain"],
        universal_newlines=True,
    )
    require(observed_head == OFFICIAL_COMMIT, "official repository HEAD drift")
    require(observed_status == "", "official repository is dirty")
    require(not OUTPUT.exists(), "refuse to overwrite canonical output")
    grid, formulas = load_grid(INPUT)
    columns = [chr(ord("B") + index) for index in range(16)]
    workloads = [grid[column + "1"] + "/" + grid[column + "2"] for column in columns]
    require(len(set(workloads)) == 16, "expected sixteen distinct workloads")
    row_ids = {
        "eyeriss": 6,
        "eyeriss_linear": 7,
        "ptb": 8,
        "sato": 9,
        "mint": 10,
        "prosperity": 13,
    }
    runtimes = {
        name: [float(grid[column + str(row)]) for column in columns]
        for name, row in row_ids.items()
    }
    prosperity = runtimes["prosperity"]
    comparisons = {}
    for baseline in ("eyeriss", "eyeriss_linear", "ptb", "sato", "mint"):
        ratios = [value / candidate for value, candidate in zip(runtimes[baseline], prosperity)]
        comparisons[baseline + "_over_prosperity"] = {
            "per_workload_speedup": [
                {"workload": workload, "speedup": ratio}
                for workload, ratio in zip(workloads, ratios)
            ],
            "arithmetic_mean_speedup": arithmetic_mean(ratios),
            "geometric_mean_speedup": geometric_mean(ratios),
            "ratio_of_summed_runtimes": ratio_of_sums(
                runtimes[baseline], prosperity),
            "cnn4_geometric_mean_speedup": geometric_mean(ratios[:4]),
            "transformer12_geometric_mean_speedup": geometric_mean(ratios[4:]),
            "minimum_speedup": min(ratios),
            "maximum_speedup": max(ratios),
        }
    official_eyeriss_geomean = float(grid["R22"])
    observed_eyeriss_geomean = comparisons["eyeriss_over_prosperity"]["geometric_mean_speedup"]
    require(abs(official_eyeriss_geomean - observed_eyeriss_geomean) < 1e-12,
            "official row R22 geometric mean mismatch")
    require("R22" in formulas, "official R22 formula cell missing")
    ptb_arithmetic = comparisons["ptb_over_prosperity"]["arithmetic_mean_speedup"]
    ptb_geometric = comparisons["ptb_over_prosperity"]["geometric_mean_speedup"]
    payload = {
        "schema": "m526_prosperity_headline_method_audit_v2",
        "status": "PASS_OFFICIAL_ARTIFACT_AGGREGATION_AUDIT__NOT_H67_PERFORMANCE",
        "official_artifact": {
            "repository": "https://github.com/dubcyfor3/Prosperity",
            "commit": OFFICIAL_COMMIT,
            "observed_head": observed_head,
            "observed_git_status": "clean",
            "workbook": str(INPUT),
            "workbook_sha256": INPUT_SHA256,
            "population": "16 model/dataset workloads",
        },
        "comparisons": comparisons,
        "headline_observation": {
            "ptb_over_prosperity_arithmetic_mean": ptb_arithmetic,
            "ptb_over_prosperity_geometric_mean": ptb_geometric,
            "ptb_over_prosperity_ratio_of_summed_runtimes":
                comparisons["ptb_over_prosperity"]["ratio_of_summed_runtimes"],
            "absolute_distance_from_paper_7p4": {
                "arithmetic_mean": abs(ptb_arithmetic - 7.4),
                "geometric_mean": abs(ptb_geometric - 7.4),
                "ratio_of_summed_runtimes": abs(
                    comparisons["ptb_over_prosperity"]["ratio_of_summed_runtimes"] - 7.4),
            },
            "paper_aggregation_convention_identified": False,
            "official_workbook_explicitly_uses_geomean_for_eyeriss_row_R22": True,
            "interpretation": (
                "The artifact yields about 7.46x arithmetic mean, 7.31x geometric mean, "
                "and 6.73x ratio-of-summed-runtimes for PTB/Prosperity. The paper's exact "
                "headline aggregation convention cannot be inferred from proximity alone."
            ),
        },
        "claim_boundary": {
            "official_prosperity_evidence": True,
            "h67_result": False,
            "ours_speedup": False,
            "system_speedup": False,
            "paper_headline": False,
        },
        "evidence_status": {
            "prosperity_artifact_recomputed": True,
            "paper_text_verified_by_this_script": False,
            "phi_or_firefly_artifact_recomputed": False,
            "h67_run": False,
        },
    }
    temp_root = Path(tempfile.mkdtemp(prefix="m526_", dir=str(OUTPUT.parent)))
    try:
        json_path = temp_root / "m526_prosperity_headline_method_audit_r2.json"
        json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report = [
            "# M526 official Prosperity headline-method audit",
            "",
            "Status: `PASS_OFFICIAL_ARTIFACT_AGGREGATION_AUDIT__NOT_H67_PERFORMANCE`",
            "",
            "The official 16-workload workbook gives PTB/Prosperity speedup of "
            "`%.6fx` by arithmetic mean and `%.6fx` by geometric mean."
            % (ptb_arithmetic, ptb_geometric),
            "",
            "This result audits aggregation only. It is not an H67 result, ours speedup, "
            "or permission to relabel external simulator evidence.",
            "",
            "| Baseline / Prosperity | Arithmetic mean | Geometric mean | Ratio of sums | Min | Max |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for baseline in ("eyeriss", "ptb", "sato", "mint"):
            item = comparisons[baseline + "_over_prosperity"]
            report.append(
                "| %s | %.6fx | %.6fx | %.6fx | %.6fx | %.6fx |"
                % (baseline, item["arithmetic_mean_speedup"],
                   item["geometric_mean_speedup"],
                   item["ratio_of_summed_runtimes"], item["minimum_speedup"],
                   item["maximum_speedup"])
            )
        (temp_root / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
        files = sorted(path for path in temp_root.iterdir() if path.is_file())
        manifest = "".join("%s  %s\n" % (sha256(path), path.name) for path in files)
        manifest_path = temp_root / "SHA256SUMS"
        manifest_path.write_text(manifest, encoding="ascii")
        (temp_root / "SHA256SUMS.seal.sha256").write_text(
            "%s  SHA256SUMS\n" % sha256(manifest_path), encoding="ascii"
        )
        os.replace(str(temp_root), str(OUTPUT))
    except Exception:
        shutil.rmtree(str(temp_root), ignore_errors=True)
        raise
    print("PASS M526 official Prosperity aggregation audit at %s" % OUTPUT)


if __name__ == "__main__":
    main()
