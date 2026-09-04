#!/usr/bin/env python3
"""Local non-independent postrun checks for the sealed M430 chain."""

from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import py_compile


ROOT = Path(__file__).resolve().parents[2]


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def check_double_seal(directory):
    manifest = directory / "SHA256SUMS"
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, name = line.split("  ", 1)
        require(sha256(directory / name) == expected,
                "inner seal mismatch: " + str(directory / name))
    expected, name = (directory / "SHA256SUMS.seal.sha256").read_text(
        encoding="utf-8").strip().split("  ", 1)
    require(name == "SHA256SUMS" and sha256(manifest) == expected,
            "outer seal mismatch: " + str(directory))


def main():
    out = ROOT / "results/m430c_local_postrun_verification_r1_20260826"
    require(not out.exists(), "refusing local verification overwrite")
    train_contract = load(ROOT / "contracts/m430a_trainonly_dualaware_q32_catalog_contract_r1_20260826.json")
    held_contract = load(ROOT / "contracts/m430b_h67_dualaware_q32_heldout_once_contract_r1_20260826.json")
    for contract in (train_contract, held_contract):
        for name, identity in contract["inputs"].items():
            path = ROOT / identity["path"]
            require(path.is_file() and sha256(path) == identity["sha256"],
                    "contract identity mismatch: " + name)
    py_compile.compile(str(ROOT / train_contract["inputs"]["builder"]["path"]),
                       doraise=True)
    py_compile.compile(str(ROOT / held_contract["inputs"]["analyzer"]["path"]),
                       doraise=True)

    train_dir = ROOT / "results/m430a_trainonly_dualaware_q32_catalog_r1_20260826"
    held_dir = ROOT / "results/m430b_h67_dualaware_q32_heldout_once_r1_20260826"
    check_double_seal(train_dir)
    check_double_seal(held_dir)
    catalog = load(train_dir / "m430_trainonly_dualaware_q32_catalog_r1.json")
    m338 = load(ROOT / held_contract["inputs"]["m338_catalog"]["path"])
    prefix_mismatch = tail_pool_mismatch = partitions = 0
    for op in range(4):
        for partition in range(432):
            centers = [int(value, 16) for value in
                       catalog["operators"][op]["partitions"][partition]
                       ["nested_patterns"]]
            old = [int(value, 16) for value in
                   m338["operators"][op]["partitions"][partition]
                   ["nested_patterns"]]
            prefix_mismatch += int(centers[:16] != old[:16])
            pool = set(old[16:128])
            tail_pool_mismatch += sum(value not in pool for value in centers[16:])
            require(len(centers) == 32 and len(set(centers)) == 32,
                    "catalog center extent/uniqueness drift")
            partitions += 1
    require(partitions == 1728 and prefix_mismatch == 0 and
            tail_pool_mismatch == 0, "catalog prefix/pool audit failure")

    train_manifest = load(ROOT / train_contract["inputs"]
                          ["m73_train_trace_manifest"]["path"])
    held_manifest = load(ROOT / held_contract["inputs"]["m40_trace"]["path"])
    train_keys = set(train_manifest["split_audit"]["selected_sample_keys"])
    held_keys = {record["sample_key"] for record in held_manifest["records"]}
    overlap = sorted(train_keys & held_keys)
    require(len(train_keys) == 32 and len(held_keys) == 10 and not overlap,
            "train/heldout overlap")

    result = load(held_dir / "m430b_h67_dualaware_q32_heldout_r1.json")
    phase_total = Counter()
    with (held_dir / "per_phase_heldout_dual_replay.csv").open(
            "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 17280, "heldout phase extent drift")
    for row in rows:
        for key in ("active_rows", "eligible_rows", "pwp_rows",
                    "exact_pwp_rows", "fallback_rows",
                    "correction_ops_per_block", "used_pwp_patterns",
                    "used_center_runs", "early_matcher"):
            phase_total[key] += int(row[key])
    require(phase_total["pwp_rows"] == 15909646 and
            phase_total["exact_pwp_rows"] == 5048754 and
            phase_total["pwp_rows"] - phase_total["exact_pwp_rows"] ==
            10860892 and phase_total["fallback_rows"] == 11395922 and
            phase_total["correction_ops_per_block"] == 38055489,
            "heldout population anchor drift")

    components = result["component_ledger"]
    component_cycles = sum(components[key] for key in (
        "config_data", "config_command", "matcher", "bitmap_seal",
        "tile0_dma_data", "tile0_dma_commands", "replay0",
        "tile1_dma_exposed", "replay1", "tail", "commit"))
    candidate = result["comparisons"]["m430_catalog_dual_cycles"]
    require(component_cycles == candidate == 517041352 and
            result["comparisons"]["strong_zero_cycles"] == 742148386 and
            result["comparisons"]["m423_catalog_dual_diagnostic_cycles"] ==
            527837132 and result["decision"] ==
            "GO_M430_DUALAWARE_CATALOG", "cycle/decision anchor drift")
    require(result["traffic_and_port_ledger"]
            ["peak_dual_pwp_logical_bytes_per_cycle"] == 144 and
            result["traffic_and_port_ledger"]
            ["peak_dual_pwp_padded_signal_bytes_per_cycle"] == 160 and
            result["traffic_and_port_ledger"]
            ["strong_zero_reference_source_bytes_per_cycle"] == 96,
            "port ledger drift")
    require(sha256(ROOT / "docs/359_DATE终局冻结_20260813.md") ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "docs359 drift")

    out.mkdir(parents=True, exist_ok=False)
    receipt = {
        "schema": "m430c_local_postrun_verification_v1",
        "status": "PASS_LOCAL_NON_INDEPENDENT_M430_CHAIN_CHECK",
        "role": "local consistency check only; independent hammer remains required",
        "compile": {"train_builder": "PASS", "heldout_analyzer": "PASS"},
        "contracts": {"input_sha_mismatches": 0},
        "double_seals": {"train": "PASS", "heldout": "PASS"},
        "split": {"train_keys": len(train_keys),
                  "heldout_keys": len(held_keys), "overlap": overlap},
        "catalog": {"partitions": partitions, "q16_prefix_mismatches": 0,
                    "tail_outside_m338_pool_mismatches": 0},
        "heldout_population": dict(phase_total),
        "cycle_recompute": {"component_sum": component_cycles,
                            "reported_cycles": candidate, "mismatch": 0},
        "port": {"dual_logical_bytes_per_cycle": 144,
                 "dual_padded_signal_bytes_per_cycle": 160,
                 "strong_zero_source_bytes_per_cycle": 96},
        "docs359_sha256":
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
        "claim_boundary": {
            "independent_review": False,
            "four_h67_bottleneck_conv_only": True,
            "resource_normalized_speedup": False,
            "rtl_or_synopsys": False,
            "system_speedup": False,
            "date_headline": False
        }
    }
    receipt_path = out / "m430c_local_postrun_verification_receipt_r1.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    manifest = out / "SHA256SUMS"
    manifest.write_text("{}  {}\n".format(
        sha256(receipt_path), receipt_path.name), encoding="utf-8")
    (out / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")
    print("M430C_LOCAL_PASS cycles={} exact={} positive={} overlap=0".format(
        candidate, phase_total["exact_pwp_rows"],
        phase_total["pwp_rows"] - phase_total["exact_pwp_rows"]))


if __name__ == "__main__":
    main()
