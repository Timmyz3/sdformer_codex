#!/usr/bin/env python3
from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CONTRACT = HW / "contracts/m64_online_adaptive_parent_selector_directed_vcs_contract_r1_20260823.json"
FILES = {
    "rtl": HW / "rtl_m64/qfit_adaptive_parent_selector_p256.sv",
    "sva": HW / "verif_m64/qfit_adaptive_parent_selector_p256_assertions.sv",
    "tb": HW / "tb_m64/tb_qfit_adaptive_parent_selector_p256.sv",
    "filelist": HW / "dc_handoff/filelists/date_m64_parent_selector_directed_vcs.f",
    "runner": HW / "dc_handoff/scripts/run_vcs_m64_parent_selector_directed_sva.sh",
    "builder": HW / "dc_handoff/scripts/build_m64_parent_selector_directed_vcs_receipt.py",
    "validator": Path(__file__).resolve(),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("non-standard constant: " + value)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    args = parser.parse_args()
    receipt_path = args.run / "m64_directed_vcs_receipt_r1.json"
    receipt = load(receipt_path)
    contract = load(CONTRACT)
    require(receipt["status"] == "PASS_EXACT_SHA_DIRECTED_VCS_SVA",
            "receipt status drift")
    require(receipt["contract_sha256"] == sha(CONTRACT), "contract SHA drift")
    for name, path in FILES.items():
        require(receipt["identity"][name + "_sha256"] == sha(path),
                name + " SHA drift")
    require(receipt["simv_sha256"] == sha(args.run / "simv"), "simv SHA drift")
    require(receipt["results"]["tests"] == 4096, "test count drift")
    require(receipt["results"]["outputs"] == 4096, "output count drift")
    require(receipt["results"]["parent_hits"] == {
        "zero": 1271, "left": 974, "up": 988,
        "previous_timestep": 863}, "parent cover drift")
    require(receipt["results"]["functional_mismatches"] == 0,
            "functional mismatch admitted")
    require(receipt["admission"] == {
        "all10_trace_admitted": False,
        "cycles_or_system_speedup_admitted": False,
        "dc_sta_formality_admitted": False,
        "directed_vcs_sva_admitted": True,
        "headline_admitted": False,
        "power_energy_ppa_admitted": False,
        "seed_sram_or_scheduler_admitted": False,
    }, "admission boundary drift")
    require(contract["directed_campaign"]["required_pass_line"] in (
        args.run / "sim.raw.log").read_text(), "PASS line drift")
    print("PASS M64 directed validator receipt_sha256={} simv_sha256={} tests=4096 outputs=4096 system_speedup_admitted=false".format(
        sha(receipt_path), receipt["simv_sha256"]))


if __name__ == "__main__":
    main()
