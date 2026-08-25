#!/usr/bin/env python3
"""Validate the independent M37-r8 VCS/source-intent-only admission."""

import argparse
import copy
import hashlib
import importlib.util
import json
import pathlib
import re
import stat
import sys


EXPECTED_STATUS = "PASS_EXACT_M37_R8_VCS_SOURCE_INTENT_ONLY"
EXPECTED_SCHEMA = "m37_r8_independent_vcs_source_intent_admission_v1"
VCS_DIR = pathlib.Path(
    "/home/zhumd/work/synopsys_date_dual/hw_autoresearch_nts07/dc_handoff/"
    "runs/m37_csd_reconstruct_t10_vcs_r8_20260822"
)
EXPECTED_EXTERNAL = {
    "receipt": "363fb61d2838b6379a065dd8eb23b6219441cfb8ed70164766f07d8469e95d97",
    "contract": "91155bebc0ac7526be6082130c4b05ece5e15d977fa05f64b055cf611ddcd214",
    "rtl": "ab7d73a6a82f8547437919813d6cf9496d0672fc23f46cfaec0c3d9be46c8cbd",
    "snapshot_provenance": "f7b88ceafe4447ad7dc1abb11751bead49d3170293ffec1ea6f521aac0c99f99",
    "snapshot_ledger": "01dc86fcda8ba3627e2de27fbab26866ca794b0e3e8da05d6fbd563cf72364a3",
    "input_manifest": "b81d30b3e25795abb53d3e151a9a5c4ec9d6a520e4269dedc889c694c8e092a7",
    "output_manifest": "4a272bd92ab6776d17b0a5d20d18bd259c264b721e7d7daec4156f0de7517d3a",
    "run_local_seal": "0528c4b311d70c4be88566a7112f36818c757a0c9a7cc482e479722da397c630",
    "compile_log": "e83804bca04af32d9f050b50700c9604119609ad68a46d0b034929bd8db50e98",
    "sim_log": "9853e34c2e6bdae0a10b436aaa4336eeb87af9ae35ac972ccee47b4d7aadded8",
    "vectors": "2d58455e5b9bbf4b15450649f6259a6216c3ff8dbcb1097e90439c3c067e1627",
    "source_audit": "c037891f997c35ac612a098aed8f8cf9f70e2cce3e3775b4cc8de2fdd61a4632",
    "runner_status": "ebaf9d12a2a06b42a8803729a5679a47330108ceda7e7d60b725f234039ccd5f",
    "independent_auditor": "6fcf221ac018e38283723b687852e1809941aabdbbfa031dd812da14113cc856",
}


class ValidationFailure(RuntimeError):
    pass


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValidationFailure(message)


def validate_payload(payload):
    require(payload.get("schema") == EXPECTED_SCHEMA, "schema drift")
    require(payload.get("status") == EXPECTED_STATUS, "status drift")
    anchors = payload["anchors"]
    for key in (
        "receipt", "contract", "rtl", "snapshot_provenance", "snapshot_ledger",
        "input_manifest", "output_manifest", "run_local_seal",
    ):
        require(anchors[key][1] == EXPECTED_EXTERNAL[key], "{} anchor drift".format(key))
    review = payload["review"]
    require(review == {
        "independent_of_r8_implementation": True,
        "score_0_to_100": 94,
        "p0": 0,
        "p1": 1,
        "p2": 3,
        "go": "STANDALONE_R8_VCS_AND_EXACT_SHA_BOUND_SOURCE_INTENT_ONLY",
        "nogo": "DC_STA_FORMALITY_PPA_POWER_ENERGY_SYSTEM_HEADLINE",
    }, "review decision drift")
    require(payload["admitted"] == {
        "standalone_r8_vcs_functional": True,
        "exact_sha_bound_source_intent": True,
        "physical_zero_multiplier": False,
        "dc": False,
        "sta": False,
        "formality": False,
        "ppa": False,
        "power": False,
        "energy": False,
        "system": False,
        "headline": False,
    }, "claim boundary drift")
    require(payload["independent_source_reaudit"]["dut_constant_uses_integer_multiplier_used_as_structure_proof"] is False, "constant DUT signal incorrectly used as proof")


def resolve_manifest_target(name, root):
    path = pathlib.Path(name)
    return path if path.is_absolute() else root / path


def verify_manifest_text(text, root, expected_count, substitutions=None):
    substitutions = substitutions or {}
    records = []
    for line in text.splitlines():
        match = re.match(r"^([0-9a-f]{64})  (.+)$", line)
        require(match is not None, "malformed manifest line")
        records.append((match.group(1), match.group(2)))
    require(len(records) == expected_count, "manifest entry-count drift")
    require(len({name for _, name in records}) == expected_count, "duplicate manifest target")
    for expected_sha, name in records:
        target = substitutions.get(name, resolve_manifest_target(name, root))
        require(target.is_file(), "manifest target missing: {}".format(target))
        require(sha256(target) == expected_sha, "manifest target SHA drift: {}".format(target))


def load_source_auditor(path):
    spec = importlib.util.spec_from_file_location("m37_source_auditor", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_external(payload, hw_root):
    receipt = hw_root.parent / payload["anchors"]["receipt"][0]
    contract = hw_root.parent / payload["anchors"]["contract"][0]
    rtl = hw_root.parent / payload["anchors"]["rtl"][0]
    snapshot_provenance = hw_root.parent / payload["anchors"]["snapshot_provenance"][0]
    snapshot_ledger = hw_root.parent / payload["anchors"]["snapshot_ledger"][0]
    auditor_path = hw_root.parent / payload["independent_source_reaudit"]["auditor"][0]
    for label, path in (
        ("receipt", receipt), ("contract", contract), ("rtl", rtl),
        ("snapshot_provenance", snapshot_provenance),
        ("snapshot_ledger", snapshot_ledger),
        ("independent_auditor", auditor_path),
    ):
        require(sha256(path) == EXPECTED_EXTERNAL[label], "{} external SHA drift".format(label))
    require(stat.S_IMODE(rtl.stat().st_mode) == 0o444, "immutable r8 snapshot mode drift")
    verify_manifest_text(snapshot_ledger.read_text(), rtl.parent, 2)
    require(
        payload["historical_manifest_resolution"]["required_snapshot_mode_octal"] == "0444",
        "snapshot mode contract drift",
    )
    receipt_payload = json.loads(receipt.read_text())
    require(receipt_payload["schema"] == "m37_output_receipt_v3", "receipt schema drift")
    require(pathlib.Path(receipt_payload["vcs_run"]["directory"]).resolve() == VCS_DIR.resolve(), "receipt VCS directory drift")

    input_manifest = VCS_DIR / "input_sha256.txt"
    output_manifest = VCS_DIR / "output_sha256.txt"
    local_seal = VCS_DIR / "run_local_seal.sha256"
    require(sha256(input_manifest) == EXPECTED_EXTERNAL["input_manifest"], "input manifest SHA drift")
    require(sha256(output_manifest) == EXPECTED_EXTERNAL["output_manifest"], "output manifest SHA drift")
    require(sha256(local_seal) == EXPECTED_EXTERNAL["run_local_seal"], "local seal SHA drift")
    verify_manifest_text(
        input_manifest.read_text(), hw_root, 8,
        {"rtl_m37/qfit_atlif_csd_reconstruct_t10.sv": rtl},
    )
    verify_manifest_text(output_manifest.read_text(), VCS_DIR, 5)
    verify_manifest_text(local_seal.read_text(), VCS_DIR, 3)

    files = {
        "compile_log": VCS_DIR / "compile.log",
        "sim_log": VCS_DIR / "sim.log",
        "vectors": VCS_DIR / "vectors.txt",
        "source_audit": VCS_DIR / "rtl_multiplier_intent_audit.txt",
        "runner_status": VCS_DIR / "runner_status.txt",
    }
    for label, path in files.items():
        require(sha256(path) == EXPECTED_EXTERNAL[label], "{} SHA drift".format(label))
    sim = files["sim_log"].read_text(errors="replace")
    require("Compiler version V-2023.12-SP1_Full64; Runtime version V-2023.12-SP1_Full64" in sim, "VCS version marker absent")
    require("M37_PASS total_tiles=245 nominal_tiles=96 dut_unique_signed_input_coefficient_product_pairs=65536 product_miters=117600 bit_miters=39200 arithmetic_issues=1225 no_data_multiplier=1" in sim, "M37 pass metric drift")
    cover_matches = [int(value) for value in re.findall(r", 2758 attempts, ([0-9]+) match$", sim, re.MULTILINE)]
    require(cover_matches == [220, 1271, 249, 117, 245, 571, 133, 210], "SVA cover vector drift")

    source_auditor = load_source_auditor(auditor_path)
    source_text = rtl.read_text()
    _, stars = source_auditor.audit_text(source_text)
    require(len(stars) == 44, "canonical star count drift")
    counterexamples = source_auditor.run_counterexamples(source_text)
    require(len(counterexamples) == 3 and all("result=REJECT" in item for item in counterexamples), "source forgery self-test drift")

    # These are in-memory adversarial tests.  They do not rely on the immutable
    # artifact SHA and therefore exercise the validator semantics themselves.
    for name, mutator in (
        ("forged_status", lambda forged: forged.update(status="PASS_DC")),
        ("forged_dc_claim", lambda forged: forged["admitted"].update(dc=True)),
        ("forged_receipt_sha", lambda forged: forged["anchors"]["receipt"].__setitem__(1, "0" * 64)),
    ):
        forged = copy.deepcopy(payload)
        mutator(forged)
        try:
            validate_payload(forged)
        except ValidationFailure:
            continue
        raise ValidationFailure("{} counterexample was accepted".format(name))
    forged_manifest = input_manifest.read_text().replace(input_manifest.read_text()[:64], "0" * 64, 1)
    try:
        verify_manifest_text(
            forged_manifest, hw_root, 8,
            {"rtl_m37/qfit_atlif_csd_reconstruct_t10.sv": rtl},
        )
    except ValidationFailure:
        pass
    else:
        raise ValidationFailure("forged manifest counterexample was accepted")
    forged_log = sim.replace("total_tiles=245", "total_tiles=999", 1).encode()
    require(hashlib.sha256(forged_log).hexdigest() != EXPECTED_EXTERNAL["sim_log"], "forged log counterexample was accepted")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("admission", type=pathlib.Path)
    parser.add_argument("--hw-root", type=pathlib.Path, default=pathlib.Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    payload = json.loads(args.admission.read_text())
    validate_payload(payload)
    validate_external(payload, args.hw_root.resolve())
    print("M37_R8_VCS_SOURCE_INTENT_ADMISSION_VALID=1 status={}".format(EXPECTED_STATUS))
    print("manifest_entries=input8/output5/local3 source_counterexamples=3 artifact_counterexamples=5")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValidationFailure, OSError, ValueError, KeyError) as error:
        print("M37_R8_VCS_SOURCE_INTENT_ADMISSION_VALID=0 detail={}".format(error), file=sys.stderr)
        raise SystemExit(1)
