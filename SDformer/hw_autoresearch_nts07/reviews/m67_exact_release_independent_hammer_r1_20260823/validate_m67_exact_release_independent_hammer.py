#!/usr/bin/env python3
"""Independent read-only audit for the frozen M67 exact VCS release.

The producer directory is never written.  Adversarial linked-resign and path
substitution candidates are constructed only in a TemporaryDirectory.
"""

from __future__ import print_function

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


HW = Path(__file__).resolve().parents[2]
RUN = HW / "dc_handoff/runs/m67_lookahead_pressure_exact_sha_vcs_r1_20260823"
CONTRACT = HW / "contracts/m67_lookahead_pressure_vcs_contract_r1_20260823.json"
VALIDATOR = HW / "dc_handoff/scripts/validate_m67_lookahead_pressure_vcs_r1.py"
BUILDER = HW / "dc_handoff/scripts/build_m67_lookahead_pressure_vcs_receipt_r1.py"
RUNNER = HW / "dc_handoff/scripts/run_vcs_m67_lookahead_pressure_exact_sha_r1.sh"
RECEIPT = RUN / "m67_lookahead_pressure_vcs_receipt_r1.json"
PY36 = Path("/usr/bin/python3.6")

EXPECTED_SHA = {
    "contract": "2ca2884b6bbec6e54eca878ea15aa70a4c819539d54bd9d67527fa4d3914634a",
    "validator": "b9784ac9c400e9843b5950495746f37cf8df744e7c63a941cf53405abcfca496",
    "builder": "ef0ccd200268be4a67ec96006f61f5d766c09f5b7ca031c0b9c66c778774d8f3",
    "runner": "4d60a8e2de639caab556fc753e57dd319b0da9998c95cf9e32dd3e791b1ca866",
    "output_manifest": "cd35fddbc19d35a6c6d508a117f2f5f4bddf58757f2870370623cc4db3f3b411",
    "output_check": "00e3b8cd058e25c528a4371d4b1af0be4ecae9922ffd58ee4b81e3f8917a06ef",
    "run_complete": "9b2f1beaa53e05f672f83a93c4f9e01fe395136d405a0d770b8d1da55e447e5d",
    "receipt": "28623dd9d8b589e4ebd7e6f0f1ee4c799de8543db9d8ceca0100d046bfca4703",
    "sim_log": "39c4a7f18c45b65b6845fb34689e51063eb14078948af653d9a1ab4060cec5f4",
    "ledger": "ccb48854d459496714389528b7619b6ca13a50d33070e8f80e7f79752fb51b04",
    "snapshot_manifest": "6b40b149d9255c9ef653d2f8676ab5e79842eed1259c6a6b5433ac8c153c96d6",
    "compile_log": "84742b79cc689401a7612aafca2bb484b6e5fdd91a666137b3168b8e101c53f4",
    "zero_rc": "9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa",
    "validator_log": "b2985d9ee7f5e86350da9a85f5c7303a01df1031fcaa62c413e5bdb1d2c72067",
    "simv": "888d839de03564d2ac2ad7d7d4d0bd17513cd12c238f28358027cbd831a09b7b",
}

FAILED_MARKERS = {
    "results/m67_pressure_dev_r1_20260823/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt":
        "46277d95b77d90902e865f4864c1d9e2c9ff816786f3d8caaaf956eb568aeb26",
    "results/m67_pressure_dev_r2_20260823/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt":
        "c9240b1d8931c88ccb22b6c875b35b917d671830b2558896f1257cf7997009bf",
    "results/m67_pressure_dev_r3_20260823/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt":
        "0f2431991e4122c2f8debb1e23cdb164549d45e7a711e2906300050f06f8c5c9",
    "results/m67_pressure_dev_r4_20260823/FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt":
        "9dd73bfdf27459b4593d23b2e6ab964254afb499a2626c8e9a566386ef59c5fc",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=strict_pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          ValueError("non-standard JSON constant: " + value)))


def validate_exact_roots():
    paths = {
        "contract": CONTRACT,
        "validator": VALIDATOR,
        "builder": BUILDER,
        "runner": RUNNER,
        "output_manifest": RUN / "output_manifest.sha256",
        "output_check": RUN / "output_check.raw.log",
        "run_complete": RUN / "RUN_COMPLETE.txt",
        "receipt": RECEIPT,
        "sim_log": RUN / "sim.raw.log",
        "ledger": RUN / "m67_handshake_ledger.log",
        "snapshot_manifest": RUN / "snapshot.sha256",
        "compile_log": RUN / "compile.raw.log",
        "validator_log": RUN / "validator.raw.log",
        "simv": RUN / "simv",
    }
    for name, path in paths.items():
        require(path.is_file() and not path.is_symlink(),
                "missing or symlinked exact root: " + name)
        require(sha256(path) == EXPECTED_SHA[name], "exact root drift: " + name)
    for name in ("compile.rc", "sim.rc"):
        require(sha256(RUN / name) == EXPECTED_SHA["zero_rc"], name + " drift")


def validate_modes_and_manifest():
    require(RUN.resolve() == RUN and RUN.is_dir() and not RUN.is_symlink(),
            "run is not canonical")
    regular = set()
    symlinks = []
    for path in RUN.rglob("*"):
        mode = stat.S_IMODE(path.lstat().st_mode)
        relative = path.relative_to(RUN).as_posix()
        if path.is_symlink():
            symlinks.append(relative)
        elif path.is_dir():
            require(mode == 0o555, "directory is not 0555: " + relative)
        elif path.is_file():
            require(mode == 0o444, "file is not 0444: " + relative)
            regular.add(relative)
        else:
            raise ValueError("unexpected filesystem member: " + relative)
    require(len(symlinks) == 19, "symlink count drift")
    require(not any(name.startswith("snapshot/") for name in symlinks),
            "snapshot contains symlink")

    manifest_path = RUN / "output_manifest.sha256"
    manifest_entries = {}
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        require(line.strip(), "blank output manifest line")
        expected, raw_name = line.split(None, 1)
        require(re.fullmatch(r"[0-9a-f]{64}", expected) is not None,
                "bad manifest digest")
        require(raw_name.startswith("./"), "non-relative manifest entry")
        name = raw_name[2:]
        require(name not in manifest_entries, "duplicate manifest path: " + name)
        require(".." not in Path(name).parts and not Path(name).is_absolute(),
                "unsafe manifest path: " + name)
        target = RUN / name
        require(target.is_file() and not target.is_symlink(),
                "manifest member missing/symlink: " + name)
        require(sha256(target) == expected, "manifest member drift: " + name)
        manifest_entries[name] = expected
    require(len(manifest_entries) == 109, "output manifest entry count drift")
    require(set(manifest_entries) ==
            regular - {"output_manifest.sha256", "output_check.raw.log"},
            "output manifest regular-file coverage drift")
    check_lines = (RUN / "output_check.raw.log").read_text(
        encoding="utf-8").splitlines()
    require(check_lines == ["./{}: OK".format(name)
                            for name in manifest_entries],
            "stored manifest check output drift")


def validate_semantics():
    contract = load_json(CONTRACT)
    receipt = load_json(RECEIPT)
    boundary = contract["claim_boundary"]
    require(receipt["claim_boundary"] == boundary, "receipt boundary drift")
    require(boundary["directed_pressure_vcs_sva_admitted"] is True,
            "scoped VCS claim absent")
    for name, admitted in boundary.items():
        if name != "directed_pressure_vcs_sva_admitted":
            require(admitted is False, "widened canonical claim: " + name)
    require(receipt["contract"]["sha256"] == EXPECTED_SHA["contract"],
            "receipt contract root drift")
    require(receipt["run_directory"] == str(RUN), "receipt path drift")
    require(receipt["run_artifact_sha256"] == {
        "compile.raw.log": EXPECTED_SHA["compile_log"],
        "compile.rc": EXPECTED_SHA["zero_rc"],
        "sim.raw.log": EXPECTED_SHA["sim_log"],
        "sim.rc": EXPECTED_SHA["zero_rc"],
        "m67_handshake_ledger.log": EXPECTED_SHA["ledger"],
        "snapshot.sha256": EXPECTED_SHA["snapshot_manifest"],
    }, "receipt artifact ledger drift")

    complete = {}
    for line in (RUN / "RUN_COMPLETE.txt").read_text(encoding="utf-8").splitlines():
        key, value = line.split("=", 1)
        require(key not in complete, "duplicate RUN_COMPLETE key")
        complete[key] = value
    require(complete == {
        "status": "PASS_EXACT_SHA_SYNOPSYS_VCS_PRESSURE_R1",
        "receipt_sha256": EXPECTED_SHA["receipt"],
        "tamper_attacks_rejected": "6",
        "system_speedup_admitted": "false",
        "headline_admitted": "false",
        "ppa_admitted": "false",
    }, "RUN_COMPLETE semantic drift")

    sim = (RUN / "sim.raw.log").read_text(encoding="utf-8")
    for marker in contract["required_marker_lines"]:
        require(sim.count(marker) == 1, "simulation marker drift")
    require(sim.count(contract["required_terminal_pass_line"]) == 1,
            "terminal PASS not unique")
    require(re.search(r"Assertion failure|failed at|Offending|^Error|^Fatal",
                      sim, re.I | re.M) is None,
            "failure signature in canonical simulation")
    require((RUN / "compile.rc").read_text().strip() == "0" and
            (RUN / "sim.rc").read_text().strip() == "0", "nonzero rc")


def validate_ledger():
    counts = {"C": 0, "L": 0, "R": 0, "O": 0, "END": 0}
    command_tags = []
    output_tags = []
    request_tags = []
    launch_counts = {1: 0, 2: 0, 3: 0, 4: 0}
    launch_slots = 0
    for line in (RUN / "m67_handshake_ledger.log").read_text(
            encoding="utf-8").splitlines():
        fields = line.split()
        require(fields and fields[0] in counts, "bad ledger record")
        kind = fields[0]
        counts[kind] += 1
        if kind == "C":
            command_tags.append(int(fields[3], 16))
        elif kind == "L":
            k = int(fields[2])
            require(k in launch_counts, "bad launch K")
            launch_counts[k] += 1
            launch_slots += k
        elif kind == "R":
            request_tags.append(int(fields[2], 16))
        elif kind == "O":
            output_tags.append(int(fields[2], 16))
        else:
            require(line == "END commands=73 outputs=73", "ledger END drift")
    require(counts == {"C": 73, "L": 30, "R": 56, "O": 73, "END": 1},
            "ledger conservation drift")
    require(command_tags == list(range(73)), "command tags not consecutive")
    require(output_tags == list(range(73)), "output tags not consecutive")
    require(request_tags == list(range(56)), "request tags not consecutive")
    require(launch_slots == 73, "launch slot conservation drift")
    require(launch_counts == {1: 11, 2: 5, 3: 4, 4: 10},
            "launch K partition drift")
    return {"record_counts": counts, "launch_slots": launch_slots,
            "launch_k_partition": launch_counts}


def invoke_validator(run, receipt):
    result = subprocess.run(
        [str(PY36), str(VALIDATOR), "--mode", "full", "--run", str(run),
         "--receipt", str(receipt)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        universal_newlines=True)
    return result.returncode, result.stdout.strip()


def linked_resign_attacks():
    canonical_rc, canonical_out = invoke_validator(RUN, RECEIPT)
    require(canonical_rc == 0 and canonical_out.startswith(
        "PASS M67 full validator tamper_rejected=6 "),
        "producer full validator rerun failed")
    receipt = load_json(RECEIPT)
    survived = []
    with tempfile.TemporaryDirectory(prefix="m67_exact_hammer_") as temp_name:
        temp = Path(temp_name)

        alias = temp / "run_alias"
        alias.symlink_to(RUN, target_is_directory=True)
        rc, _ = invoke_validator(alias, alias / RECEIPT.name)
        require(rc == 0, "expected run symlink alias survival changed")
        survived.append("run_directory_symlink_alias")

        widened = dict(receipt)
        widened["system_speedup_admitted"] = True
        widened["headline_admitted"] = True
        widened["claimed_speedup"] = 3.0
        widened_path = temp / "externally_resigned_widened_receipt.json"
        widened_path.write_text(json.dumps(widened, sort_keys=True) + "\n",
                                encoding="utf-8")
        rc, _ = invoke_validator(RUN, widened_path)
        require(rc == 0, "expected extra-claim receipt survival changed")
        survived.append("external_receipt_extra_claim_widening")

        fake_run = temp / "synthetic_run"
        (fake_run / "snapshot").mkdir(parents=True)
        artifact_names = ("compile.raw.log", "compile.rc", "sim.raw.log",
                          "sim.rc", "m67_handshake_ledger.log")
        for name in artifact_names:
            (fake_run / name).symlink_to(RUN / name)
        rc_sha = sha256(RUN / "compile.rc")
        fake_snapshot = fake_run / "snapshot.sha256"
        fake_snapshot.write_text(
            "".join("{}  ../compile.rc\n".format(rc_sha) for _ in range(9)),
            encoding="utf-8")
        synthetic = dict(widened)
        synthetic["run_directory"] = str(fake_run.resolve())
        synthetic["run_artifact_sha256"] = dict(
            receipt["run_artifact_sha256"])
        synthetic["run_artifact_sha256"]["snapshot.sha256"] = sha256(
            fake_snapshot)
        synthetic_path = temp / "synthetic_receipt.json"
        synthetic_path.write_text(json.dumps(synthetic, sort_keys=True) + "\n",
                                  encoding="utf-8")
        rc, _ = invoke_validator(fake_run, synthetic_path)
        require(rc == 0, "expected snapshot/path substitution survival changed")
        survived.append("synthetic_symlink_run_and_duplicate_parent_snapshot_entries")

    return {"canonical_validator_rc": canonical_rc,
            "canonical_terminal": canonical_out,
            "survived": survived}


def validate_failed_provenance():
    for relative, expected in FAILED_MARKERS.items():
        path = HW / relative
        require(path.is_file() and not path.is_symlink(),
                "old failed marker missing: " + relative)
        require(sha256(path) == expected, "old failed marker drift: " + relative)
        text = path.read_text(encoding="utf-8")
        require("FAILED" in text,
                "old failed marker semantics drift: " + relative)
    require(not (RUN / "FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt").exists(),
            "failed marker contaminates exact PASS release")


def main():
    validate_exact_roots()
    validate_modes_and_manifest()
    validate_semantics()
    ledger = validate_ledger()
    validate_failed_provenance()
    linked = linked_resign_attacks()
    result = {
        "status": "PASS_CURRENT_CANONICAL_BYTES_WITH_LINKED_RESIGN_P1_DISCLOSED",
        "producer_modified": False,
        "producer_full_validator_rerun": "PASS",
        "output_manifest_entries": 109,
        "all_directories_0555": True,
        "all_regular_files_0444": True,
        "producer_symlink_entries": 19,
        "ledger": ledger,
        "original_failed_markers_preserved": 4,
        "linked_resign": linked,
        "p0_count": 0,
        "p1_count": 1,
        "p2_count": 3,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print("FAIL M67 independent hammer: {}".format(error))
        raise SystemExit(1)
