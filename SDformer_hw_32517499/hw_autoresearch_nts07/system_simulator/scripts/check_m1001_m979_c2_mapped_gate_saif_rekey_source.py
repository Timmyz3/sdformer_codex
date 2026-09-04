#!/usr/bin/env python3
"""M1001 static rekey checker and frozen M979 per-SAIF validator proxy."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
FROZEN_CHECKER = HERE / "check_m979_c2_mapped_gate_saif_source.py"
RUNNER = HW / "dc_handoff/scripts/run_m1005_m1001_c2_mapped_gate_saif_one_shot.sh"
CONTRACT = HW / "contracts/m1001_m979_c2_mapped_gate_saif_rekey_source_contract_r1_20260829.json"
TEST = HW / "system_simulator/tests/test_m1001_m979_c2_mapped_gate_saif_rekey_source.py"
M979_RECEIPT = HW / "reviews/m979_m974_c2_mapped_gate_saif_source_receipt_r1_20260829"
FROZEN = {
    HW / "dc_handoff/tb/tb_m979_c2_three_axis_mapped_gate_case_saif.sv":
        "cce12a93c4c8fd8d424fbf9f6354ba30e2870a05a7480fc7de26b3b29c87266c",
    HW / "dc_handoff/scripts/m979_c2_mapped_gate_per_case_saif.ucli.tcl":
        "846cd4a1b877803cce986b39cdf0a27ec87b59451ca7e6fc9141c999df85cdad",
    FROZEN_CHECKER:
        "409a6d996e95e4fc46d2ff3cf8e26fbe5e52594d1e7b1522db599811025382d5",
    HW / "system_simulator/tests/test_m979_c2_mapped_gate_saif_source.py":
        "aaff81432b28ac506142d4890096f379db74d380516892107f20e10a6fcf2461",
    HW / "contracts/m979_m974_c2_three_axis_mapped_gate_saif_source_contract_r1_20260829.json":
        "d2939e24e587b03680b7b4e0265a8fc8b3dbbea89759e2268e97b118fe32455c",
    HW / "dc_handoff/scripts/run_m993_m979_c2_mapped_gate_saif_one_shot.sh":
        "ba98f230cd676767c121760edf4025fbb71acbb7ddadfa4f695cee9acdf51ecc",
}
RECEIPT_ID = (
    "8992b243dfe8397efe66eff4e9ba70435522172e94dd79872de7e1b7139f48cd",
    "67455efdda9eaaaa0e223eea3e61d4dbe00024ac22cc69c4d5a49c50c09731a6",
    "da08f8c116e5ba28dbf839fb733e4dc0c0efec3847fb59339f98338c16401dd9",
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load_frozen():
    require(sha(FROZEN_CHECKER) == FROZEN[FROZEN_CHECKER], "M979 checker drift")
    spec = importlib.util.spec_from_file_location("m1001_frozen_m979_checker", FROZEN_CHECKER)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module
    spec.loader.exec_module(module); return module


M979 = load_frozen()


def verify_flat_receipt():
    require(sha(M979_RECEIPT / "review.json") == RECEIPT_ID[0] and
            sha(M979_RECEIPT / "SHA256SUMS") == RECEIPT_ID[1] and
            sha(M979_RECEIPT / "SHA256SUMS.seal.sha256") == RECEIPT_ID[2],
            "M979 receipt identity drift")
    # Pure-Python verification of the frozen flat receipt.
    for line in (M979_RECEIPT / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split(maxsplit=1)
        require(sha(M979_RECEIPT / name) == expected, "M979 receipt member drift")
    outer, name = (M979_RECEIPT / "SHA256SUMS.seal.sha256").read_text().split()
    require(name == "SHA256SUMS" and sha(M979_RECEIPT / name) == outer,
            "M979 receipt outer drift")


def validate_static(contract=CONTRACT):
    for path, expected in FROZEN.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                "frozen M979 source drift: " + str(path))
    verify_flat_receipt()
    value = json.loads(Path(contract).read_text())
    require(value.get("status") == "PASS_M1001_REKEY_SOURCE_ONLY__NO_EDA" and
            value.get("launch_now") is False, "M1001 contract drift")
    runner = RUNNER.read_text()
    for token in ("M1001", "M1002", "M1003", "M1004", "M1005",
                  "k1 k8 k1x8", "0 1 2 3 4", "case${case_id}.saif",
                  "M979_CASE", "M979_UCLI_SAIF", "ATTEMPT_ATOMIC_CONSUME"):
        require(token in runner, "M1001 runner missing: " + token)
    for stale in ("M990_", "M991_", "M992_", "M993_", "m990_", "m991_",
                  "m992_", "m993_"):
        require(stale not in runner, "conflicting old chain survived: " + stale)
    canonical = value["canonical"]
    require(canonical["result"].endswith("m1005_m1001_c2_three_axis_mapped_gate_saif_r1_20260829") and
            canonical["attempt"].endswith(".m1005_m1001_c2_three_axis_mapped_gate_saif_attempt_consumed"),
            "M1005 canonical identity drift")
    boundary = value["claim_boundary"]
    require(all(boundary[key] is False for key in
                ("vcs_executed", "saif_created", "pt_executed", "ptpx_executed",
                 "power", "energy", "headline")), "M1001 false execution claim")
    return {"schema": "m1001_m979_c2_saif_rekey_static_check_v1",
            "status": "PASS_M1001_STATIC_REKEY__NO_EDA",
            "runner_sha256": sha(RUNNER), "contract_sha256": sha(contract),
            "test_sha256": sha(TEST), "frozen_file_count": len(FROZEN),
            "axes": 3, "cases": 15, "m979_semantics_modified": False,
            "vcs_pt_ptpx_executed": False, "gpu_remote_used": False}


def main():
    p = argparse.ArgumentParser(); p.add_argument("--contract", type=Path, default=CONTRACT)
    p.add_argument("--saif", type=Path); p.add_argument("--axis", choices=sorted(M979.AXES))
    p.add_argument("--case", type=int, dest="case_id"); p.add_argument("--cycles", type=int)
    a = p.parse_args()
    if a.saif:
        require(a.axis is not None and a.case_id is not None and a.cycles is not None,
                "--saif requires --axis --case --cycles")
        value = M979.validate_saif(a.saif, a.axis, a.case_id, a.cycles)
        value["validation_authority"] = "M1001_FROZEN_M979_SEMANTICS"
    else:
        value = validate_static(a.contract)
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
