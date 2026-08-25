#!/usr/bin/env python3
"""Fail-closed additive release validator for the Python-3.6 rescue of M66 r1."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def no_duplicate_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError("non-finite JSON constant: " + value)


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicate_object,
                      parse_constant=reject_constant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    contract_path = repo / "hw_autoresearch_nts07/contracts/m66_s00_additive_release_contract_r2_20260823.json"
    contract = load_json(contract_path)
    require(contract["status"] == "POSTRUN_ADDITIVE_RELEASE_FROZEN", "bad release status")
    for entry in contract["entries"]:
        path = repo / entry["path"]
        require(path.is_file(), "release entry missing: " + entry["path"])
        require(sha(path) == entry["sha256"], "release entry SHA drift: " + entry["path"])

    run = repo / contract["paths"]["failed_r1_run"]
    old_contract_path = repo / contract["paths"]["r1_contract"]
    old_contract = load_json(old_contract_path)
    receipt_path = run / "m66_s00_exact_sha_vcs_receipt.json"
    receipt = load_json(receipt_path)
    require(receipt["contract_sha256"] == sha(old_contract_path), "r1 receipt/contract drift")
    require(int((run / "sim.rc").read_text()) == 0 and
            int((run / "gzip.rc").read_text()) == 0 and
            int((run / "replay.rc").read_text()) == 0,
            "r1 hardware/replay rc is not clean")
    require(int((run / "validator.rc").read_text()) == 1, "r1 validator failure was rewritten")
    require((run / "validator.raw.log").read_text(encoding="utf-8") ==
            "FAIL M66 validator: __init__() got an unexpected keyword argument 'text'\n",
            "r1 Python compatibility failure diagnostic drift")
    require((run / "FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt").read_text(encoding="utf-8") ==
            "FAILED_OR_INCOMPLETE_DO_NOT_CITE\n", "r1 fail-closed marker removed/changed")
    require(old_contract["claim_boundary"]["system_speedup_admitted"] is False,
            "r1 claim boundary widened")

    compat_validator = repo / contract["paths"]["compat_validator"]
    process = subprocess.run([
        sys.executable, str(compat_validator), "--repo", str(repo),
        "--receipt", str(receipt_path),
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
       universal_newlines=True)
    require(process.returncode == 0, "compat validator failed: " + process.stdout)
    require(process.stdout == "PASS M66 full-S00 exact-SHA VCS receipt validator\n",
            "compat validator terminal drift: " + process.stdout)
    replay = receipt["functional_and_protocol"]
    expected = contract["expected"]
    require(replay["rtl_cycles"] == expected["m66_rtl_cycles"] and
            replay["seamless_launches"] == expected["seamless_launches"] and
            replay["functional_mismatch_count"] == 0 and
            receipt["same_trace_m57_to_m66"]["cycles_saved"] == expected["cycles_saved"],
            "released metric drift")
    require(receipt["claim_boundary"]["system_speedup_admitted"] is False and
            receipt["claim_boundary"]["paper_ppa_ready"] is False and
            receipt["claim_boundary"]["power_or_energy_admitted"] is False,
            "released claim boundary widened")
    print("PASS M66 additive r2 exact-SHA release; r1 failed marker preserved")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("FAIL M66 additive r2 release: {}".format(error))
        raise SystemExit(1)
