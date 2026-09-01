#!/usr/bin/env python3
"""Different-author source-only hammer for M1701; never launches EDA."""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
RUNNER = HW / "dc_handoff/scripts/run_dc_m1701_m1695_c1_tool_entity_repair_exact_one_shot.sh"
TCL = HW / "dc_handoff/scripts/run_dc_m1695_m1665_c1_fastmin_hold_closure_candidate.tcl"
TEST = HW / "system_simulator/tests/test_m1701_m1695_c1_tool_entity_repair_source.py"
CONTRACT = HW / "contracts/m1701_m1695_c1_tool_entity_repair_source_contract_r1_20260901.json"
WITNESS = HW / "contracts/m1701_m1695_dc_shell_symlink_pre_attempt_failure_witness_r1_20260901.json"
AUTHOR = HW / "reviews/m1701_m1695_c1_tool_entity_repair_source_author_receipt_r1_20260901"
M1695 = HW / "dc_handoff/scripts/run_dc_m1695_m1665_c1_fastmin_hold_closure_exact_one_shot.sh"
M1697 = HW / "contracts/m1697_m1696_m1695_c1_fastmin_hold_closure_launch_release_r1_20260901.json"
M1703 = HW / "contracts/m1703_m1702_m1701_m1695_c1_tool_entity_repair_launch_release_r1_20260901.json"
RESULT = HW / "dc_handoff/runs/m1701_m1695_c1_tool_entity_repair_dc_r1_20260901"
ATTEMPT = HW / "dc_handoff/runs/.m1701_m1695_c1_tool_entity_repair_dc_attempt_consumed"
DC_ENTRY = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
DC_TARGET = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell")

EXPECTED = {
    "runner": "0a6f9a22bac945ac5757279e865fad6d26da4de7b1d134721668fba253fc6e15",
    "tcl": "cb05b053078c7ab9d084cddf5028802aeff52ef1a4aef6d1b026ba6da2f41ad8",
    "test": "5c899fbccf74d355e140fc0ebd227372dff4f4e0d77927d39109e8e2b543a4e9",
    "contract": "f83f1adcb50393196c3a270cd24fcf7baa7abed19ed88a5910990863f1a49588",
    "witness": "e94035700768769bd051ce2951bed338ad7830adc03d542b46be1f8e1f0dc5f4",
    "author_receipt": "77482016529e209d7e82f03f15afa83c6fe2ff08d5d63c704f07cd3333e37ba1",
    "author_manifest": "7f709fdfbd8a5102bd557487bfa57bd4702eac38976f8a4f8e1540d634d7ec0f",
    "author_outer": "da8f323b740fa8127726bf2f905320b5de2d5b44e8c206dc68c6947ede873411",
    "m1695": "f470eee1f4f68be76d4d680522efca4157472582e9f442721ef836bd5957ca5d",
    "m1697": "45fe5a6029a182a52a63fab47288eff982ce64a861c113767cdc3db00e3c3fbb",
    "target": "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
}


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON " + token)))
    need(type(value) is dict, "JSON root")
    return value


def verify_file_seal(path):
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [sha(path), path.name], "file seal")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name], "outer seal")


def verify_dir_seal(root, manifest_sha, outer_sha):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(root.is_dir() and not root.is_symlink(), "sealed root")
    need(sha(manifest) == manifest_sha and sha(outer) == outer_sha, "seal identity")
    need(outer.read_text().split() == [manifest_sha, "SHA256SUMS"], "outer")
    listed = set()
    for row in manifest.read_text().splitlines():
        fields = row.split(maxsplit=1)
        need(len(fields) == 2, "manifest syntax")
        digest, name = fields[0], fields[1].lstrip("*")
        path = root / name
        need(name not in listed and path.is_file() and not path.is_symlink()
             and stat.S_ISREG(path.lstat().st_mode) and sha(path) == digest,
             "manifest member")
        listed.add(name)
    actual = set(path.relative_to(root).as_posix() for path in root.rglob("*")
                 if path.is_file() and path.name not in
                 {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(actual == listed, "population closure")


def main():
    for path, key in ((RUNNER, "runner"), (TCL, "tcl"), (TEST, "test"),
                      (CONTRACT, "contract"), (WITNESS, "witness"),
                      (AUTHOR / "author_receipt.json", "author_receipt"),
                      (M1695, "m1695"), (M1697, "m1697")):
        need(path.is_file() and not path.is_symlink() and sha(path) == EXPECTED[key],
             "identity " + str(path))
    verify_file_seal(CONTRACT)
    verify_file_seal(WITNESS)
    verify_file_seal(M1697)
    verify_dir_seal(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])

    contract = strict_json(CONTRACT)
    witness = strict_json(WITNESS)
    author = strict_json(AUTHOR / "author_receipt.json")
    need(contract["status"] ==
         "SOURCE_ONLY_M1701_M1695_C1_TOOL_ENTITY_REPAIR__NO_EDA_AUTHORIZED",
         "contract status")
    need(author["status"] ==
         "PASS_M1701_M1695_C1_TOOL_ENTITY_REPAIR_SOURCE_AUTHOR_HANDOFF__NO_EDA",
         "author status")
    need(witness["status"] ==
         "PASS_PRE_ATTEMPT_FAILURE_WITNESS__M1695_SHA_EXACT_REJECTED_OFFICIAL_DC_SHELL_SYMLINK__NO_EDA",
         "witness status")
    need(contract["authorization"]["dc_runs_now"] == 0
         and contract["authorization"]["future_dc_runs_max"] == 1
         and not contract["authorization"]["retry"], "budget")

    # Live official tool entity and all six negative shape mutations.
    need(stat.S_ISLNK(DC_ENTRY.lstat().st_mode), "entry symlink")
    raw = os.readlink(str(DC_ENTRY))
    direct = str((DC_ENTRY.parent / raw).absolute())
    resolved = str(DC_ENTRY.resolve(strict=True))
    need(raw == "snps_shell", "raw link")
    need(direct == str(DC_TARGET) and resolved == str(DC_TARGET), "target path")
    need(stat.S_ISREG(DC_TARGET.lstat().st_mode)
         and not stat.S_ISLNK(DC_TARGET.lstat().st_mode), "target type")
    need(sha(DC_TARGET) == EXPECTED["target"], "target SHA")
    spec = importlib.util.spec_from_file_location("m1701_author_test", str(TEST))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    good = ("snps_shell", str(DC_TARGET), str(DC_TARGET), True, False,
            EXPECTED["target"])
    mutations = [
        ("../bin/snps_shell",) + good[1:],
        (good[0], "/tmp/other") + good[2:],
        good[:2] + ("/tmp/other",) + good[3:],
        good[:3] + (False,) + good[4:],
        good[:4] + (True,) + good[5:],
        good[:5] + ("0" * 64,),
    ]
    rejected = 0
    for row in mutations:
        try:
            module.validate_entity_shape(*row)
        except ValueError:
            rejected += 1
    need(rejected == 6, "entity mutations")

    text = RUNNER.read_text()
    tcl = TCL.read_text()
    syntax = subprocess.run(["/usr/bin/bash", "-n", str(RUNNER)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            universal_newlines=True, check=False)
    need(syntax.returncode == 0, "bash syntax")
    active = "\n".join(row.split("#", 1)[0] for row in text.splitlines())
    launch = '"${DC_SHELL}" -no_home_init -no_local_init -no_gui -f "${TCL}"'
    need(active.count(launch) == 1, "DC launch count")
    need(text.count("sha_official_direct_symlink_exact") == 2,
         "one entity validation call")
    authority = text.index('verify_file_seal "${CONTRACT}"')
    shared = text.index('exec 9>"${SHARED_QUEUE}"')
    attempt = text.index('mkdir -- "${ATTEMPT}"')
    prelaunch = text.index('same-UID DC collision immediately before launch')
    launch_at = text.index(launch)
    need(authority < shared < attempt < prelaunch < launch_at, "launch order")
    need('SHARED_QUEUE="/tmp/date_dual_synopsys_same_uid_eda_queue.lock"' in text,
         "shared queue")
    need(text.count('mkdir -- "${ATTEMPT}"') == 1 and "retry=false" in text,
         "one attempt")
    module.validate_tcl(tcl)
    need("read_verilog" not in module.commands(tcl)
         and "compile_ultra" not in module.commands(tcl), "TCL scope")
    for token in ("set_false_path", "set_multicycle_path", "set_min_delay",
                  "set_max_delay", "set_disable_timing", "set_case_analysis"):
        need(not re.search(r"(?m)^\s*" + token + r"\b", tcl), "timing exception")
    need(not os.path.lexists(RESULT) and not os.path.lexists(ATTEMPT)
         and not os.path.lexists(M1703), "future namespace")

    print(json.dumps({
        "schema": "m1702_m1701_c1_tool_entity_source_independent_hammer_r1_v1",
        "status": "PASS_M1702_M1701_M1695_C1_TOOL_ENTITY_REPAIR_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_DC_ATTEMPT",
        "score": 98,
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 1,
        "verified": {
            "exact_official_direct_symlink": True,
            "raw_direct_resolved_target_sha_bound": True,
            "entity_mutations_rejected": "6/6",
            "m1695_tcl_byte_identical": True,
            "single_dc_launch": True,
            "shared_queue_and_prelaunch_collision": True,
            "attempt_before_dc_no_retry": True,
            "m1703_release_required": True,
            "launch_adjacent_entity_recheck": False,
            "eda_executed": False,
            "attempt_created": False,
            "release_created": False
        }
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
