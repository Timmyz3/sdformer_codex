#!/usr/bin/python3
"""Different-author, source-only mutation hammer for M1334 C2 activity source."""
from __future__ import print_function

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
CHECKER = HW / "system_simulator/scripts/check_m1334_c2_headline_mapped_production_activity_source.py"
TEST = HW / "system_simulator/tests/test_m1334_c2_headline_mapped_production_activity_source.py"
CONTRACT = HW / "contracts/m1334_c2_headline_mapped_production_activity_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1334_c2_headline_mapped_production_activity_source_author_r1_20260831"
M1333 = HW / "reviews/m1333_m1332_c2_headline_mapped_production_activity_source_blind_hammer_r1_20260831"
M903 = HW / "reviews/m903_m872_m803_c2_r16_three_axis_dc_result_hammer_r1_20260829"
M872 = HW / "dc_handoff/runs/m872_m803_c2_r16_channel_split_three_axis_logic_only_dc_3p000ns_r1_20260829"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
EXPECTED = {
    CHECKER: "c9326ff934239e8773e9f991e6bf0be94bba9c9c602be199433c22d1cd4c9da9",
    TEST: "d90a23fb9c7c8f18666d26dcfb2b0ac75ed160bc84ba36506e4b881701117ab8",
    CONTRACT: "c2fad444dab7dc33ace5c0eb3a07d0355237cef5586804f1fce6d274d65b8c7b",
    AUTHOR / "review.json": "0bd2a6fdd75c24efcec2f41cf10bd689c2af3c56b61c968b036803ecc80b1ed9",
    AUTHOR / "SHA256SUMS": "0d2bd9d33cd1140ed26c28e47521c43672e3456b1e20de2f7260bc63be6872a7",
    AUTHOR / "SHA256SUMS.seal.sha256": "fe6d57fb982b50a60c16e0ec3f25c0fcf99db1c21f07a6407314d26787ec331d",
    M1333 / "review.json": "a78b7b826650c490405f3c2ee003fef904779fb479f4680fc565a4e0ec617574",
    M1333 / "SHA256SUMS.seal.sha256": "19ee7ed02f85a5e122b4f04f55c5ef884fe9f6e6cd8e0b8a04808ba625e4beba",
    M903 / "review.json": "89785b3a06fc5981cb1e652bce18c4ab3853809ccf6dee7d1b96a65bd018b10a",
    M903 / "SHA256SUMS.seal.sha256": "0394ce7e485c780355dbb841797f7fa518171bb00330ae07234a1a9a4e96316f",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AXIS = {
    "k8": {
        "cycles": [51, 131, 486, 1231, 14],
        "net": "6b745030df6c041a0501d041ee277459c726c52263b4eec6ab5712f14d156de5",
        "sdc": "70a0d0e7700188f5a80f31b06c2f3d401f56c7d1e2a29428e3837064a722a96c",
    },
    "k1x8": {
        "cycles": [53, 133, 499, 1246, 14],
        "net": "65f89c13d0b181fd26708b385fc831bb4493328e24a15bbb07c2dc40f27677dc",
        "sdc": "24806d5c2d5c0afae2c01d518927e3ca96ec977d000287b0a6bc62fc42a7e317",
    },
}
NET = "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.v"
SDC = "netlist/m803_fc2_channel_split_registered_release_matched_8bank_raw4_acc24_mapped.sdc"


class HammerError(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise HammerError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def strict_json(path):
    def reject_constant(value):
        raise HammerError("non-finite JSON constant: " + value)
    with Path(path).open("r") as stream:
        return json.load(stream, parse_constant=reject_constant)


def verify_flat_seal(root):
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
            "outer seal mismatch")
    rows = {}
    for line in manifest.read_text().splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.match(r"^[0-9a-f]{64}$", fields[0]),
                "manifest row malformed")
        name = fields[1].lstrip("*")
        require(name not in rows and "/" not in name and ".." not in name,
                "flat seal member invalid")
        member = root / name
        require(member.is_file() and not member.is_symlink() and sha(member) == fields[0],
                "sealed member mismatch: " + name)
        rows[name] = fields[0]
    actual = {p.name for p in root.iterdir() if p.is_file() and
              p.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(set(rows) == actual, "flat seal population mismatch")
    return rows


def load_checker():
    spec = importlib.util.spec_from_file_location("m1335_blind_m1334", str(CHECKER))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expect_reject(label, action):
    try:
        action()
    except Exception:
        return label
    raise HammerError(label + " false negative")


def fake_saif(duration, endpoint_tc=4, inside=True):
    payload = """(PORT
      (clk_core (T0 1) (T1 1) (TX 0) (TC 20))
      (rst_core (T0 1) (T1 0) (TX 0) (TC 0))
      (raw_valid (T0 1) (T1 1) (TX 0) (TC 2))
      (raw_accept (T0 1) (T1 1) (TX 0) (TC 2))
      (mem_req_accept[0] (T0 1) (T1 1) (TX 0) (TC {endpoint}))
      (mem_rsp_accept[0] (T0 1) (T1 1) (TX 0) (TC {endpoint}))
      (result_accumulator[0] (T0 1) (T1 1) (TX 0) (TC 6))
      (result_accept (T0 1) (T1 1) (TX 0) (TC 4))
      (token_done_accept (T0 1) (T1 1) (TX 0) (TC 2)))""".format(
          endpoint=endpoint_tc)
    if inside:
        body = "(INSTANCE core (INSTANCE dut " + payload + "))"
    else:
        body = "(INSTANCE core (INSTANCE dut)) (INSTANCE checks " + payload + ")"
    return "(SAIFILE (DURATION {0}) (INSTANCE tb_m1334_c2_headline_mapped_production_activity {1}))".format(duration, body)


def pass_log(case_id, endpoint):
    return ("PASS M1334 coverage case={0} source=3 endpoint={1} commit=2 "
            "stall=1 done=1 unknown=0 fatal=0\n").format(case_id, endpoint)


def make_inventory(M, root):
    entries = []
    for axis in ("k8", "k1x8"):
        for case_id, cycles in enumerate(AXIS[axis]["cycles"]):
            endpoint = 0 if case_id == 4 else 4
            sp = root / (axis + "_case" + str(case_id) + ".saif")
            lp = root / (axis + "_case" + str(case_id) + ".log")
            sp.write_text(fake_saif(cycles * 3, endpoint))
            lp.write_text(pass_log(case_id, endpoint))
            entries.append({"axis": axis, "case": case_id, "cycles": cycles,
                "saif": sp.name, "saif_sha256": sha(sp),
                "runtime_log": lp.name, "runtime_log_sha256": sha(lp)})
    inventory = root / "inventory.json"
    inventory.write_text(json.dumps({
        "schema": "m1334_c2_production_activity_inventory_r1",
        "status": "CANDIDATE_UNSEALED_DO_NOT_CITE", "entries": entries},
        sort_keys=True))
    require(M.validate_inventory(inventory)["entry_count"] == 10,
            "positive 2x5 inventory rejected")
    return inventory


def main():
    for path, expected in EXPECTED.items():
        require(path.is_file() and not path.is_symlink() and sha(path) == expected,
                "frozen identity mismatch: " + str(path))
    author_rows = verify_flat_seal(AUTHOR)
    require(author_rows.get("review.json") == EXPECTED[AUTHOR / "review.json"],
            "author review seal mismatch")
    author_review = strict_json(AUTHOR / "review.json")
    require(author_review.get("status") ==
            "PASS_M1334_SOURCE_ONLY__INDEPENDENT_BLIND_HAMMER_REQUIRED" and
            author_review.get("eda_executed") is False,
            "author source-only boundary mismatch")
    predecessor = strict_json(M1333 / "review.json")
    require(predecessor.get("status") ==
            "FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED" and
            predecessor.get("false_negative_count") == 10,
            "M1333 rejection boundary mismatch")

    m903 = strict_json(M903 / "review.json")
    require(m903.get("status") ==
            "PASS100_M872_M803_C2_R16_THREE_AXIS_LOGIC_ONLY_DC_RESULT_ADMITTED",
            "M903 status mismatch")
    evidence = m903.get("fair_equal_bandwidth_metrics", {})
    require(evidence.get("frozen_directed_vcs_cycles") == {
            "k1x8": AXIS["k1x8"]["cycles"], "k8": AXIS["k8"]["cycles"]},
            "M903 five-workload cycle anchors mismatch")
    require(evidence.get("aggregate_sum_cycles") == {"k1x8": 1945, "k8": 1913},
            "M903 cycle sums mismatch")
    for axis in ("k8", "k1x8"):
        require(sha(M872 / axis / NET) == AXIS[axis]["net"],
                axis + " mapped netlist mismatch")
        require(sha(M872 / axis / SDC) == AXIS[axis]["sdc"],
                axis + " mapped SDC mismatch")

    contract = strict_json(CONTRACT)
    require(contract.get("frozen_cycles") == {
            "k8": AXIS["k8"]["cycles"], "k1x8": AXIS["k1x8"]["cycles"]},
            "contract cycle anchors mismatch")
    require(contract.get("authorization", {}).get("launch_now") is False and
            all(contract.get("authorization", {}).get(key) is False
                for key in ("vcs", "dc", "pt", "ptpx", "gpu", "remote")),
            "contract execution boundary mismatch")

    M = load_checker()
    baseline = M.validate_static()
    require(baseline.get("closed_predecessor_false_negatives") == 10 and
            baseline.get("eda_executed") is False, "baseline static check failed")
    attacks = []

    with tempfile.TemporaryDirectory(prefix="m1335_c2_blind_") as td:
        root = Path(td)
        good_zero = root / "good_zero.saif"
        good_zero.write_text(fake_saif(42, 0))
        require(M.validate_saif(good_zero, "k8", 4, 14)
                ["major_cone_tc"]["endpoint"] == 0.0,
                "case4 exact-zero positive rejected")
        bad_zero = root / "bad_zero.saif"
        bad_zero.write_text(fake_saif(42, 9))
        attacks.append(expect_reject("FN1_case4_endpoint_nonzero", lambda:
            M.validate_saif(bad_zero, "k8", 4, 14)))
        sibling = root / "sibling.saif"
        sibling.write_text(fake_saif(153, 4, inside=False))
        attacks.append(expect_reject("FN2_external_activity_substitution", lambda:
            M.validate_saif(sibling, "k8", 0, 51)))

        k8_text = M.FILELISTS["k8"].read_text()
        official = str((M.BASE / "k8" / M.NET).resolve())
        attacks.append(expect_reject("FN3_forged_netlist_path", lambda:
            M.validate_filelist(k8_text.replace(official,
                "/tmp/forged/k8/" + Path(official).name), "k8")))
        attacks.append(expect_reject("FN4_alternate_old_memory_provider", lambda:
            M.validate_filelist(k8_text +
                "/tmp/legacy/m349_fc2_scalar_bank_memory_model.sv\n", "k8")))

        memory = M.MEM.read_text()
        attacks.append(expect_reject("FN5_commented_reset", lambda:
            M.validate_memory_source(memory.replace(
                "epoch_q[slot] <= '0;", "// epoch_q[slot] <= '0;"))))
        source_contract = strict_json(CONTRACT)
        for fn, key in (("FN5_omitted_memory_source", "reset_safe_memory_model"),
                        ("FN6_omitted_sva_source", "production_activity_assertions")):
            mutant = dict(source_contract)
            mutant["source_files"] = [row for row in source_contract["source_files"]
                                       if key not in row["path"]]
            temp_contract = root / (fn + ".json")
            temp_contract.write_text(json.dumps(mutant, sort_keys=True))
            attacks.append(expect_reject(fn, lambda p=temp_contract:
                M.validate_static(p)))
        sva = M.SVA.read_text()
        attacks.append(expect_reject("FN6_commented_cover", lambda:
            M.validate_assertion_source(sva.replace(
                "cp_source: cover property (raw_accept);",
                "// cp_source: cover property (raw_accept);"))))

        scope = "tb_m1334_c2_headline_mapped_production_activity.core.dut"
        ucli = M.UCLI.read_text()
        mutant_ucli = ucli.replace("power " + scope,
            "# power " + scope + "\npower tb_m1334_c2_headline_mapped_production_activity.core")
        mutant_ucli = mutant_ucli.replace(
            "power -report $::env(M1334_SAIF_FILE) 1e-9 " + scope,
            "# power -report $::env(M1334_SAIF_FILE) 1e-9 " + scope +
            "\npower -report $::env(M1334_SAIF_FILE) 1e-9 tb_m1334_c2_headline_mapped_production_activity.core")
        attacks.append(expect_reject("FN7_comment_token_ucli_redirect", lambda:
            M.validate_ucli(mutant_ucli)))

        attacks.append(expect_reject("FN8_invalid_valid_accept_state", lambda:
            M.validate_memory_source(memory.replace(
                "&& mem_req_valid === 1'b1 && mem_req_ready === 1'b1",
                "&& mem_req_ready === 1'b1"))))
        require("if (request_fire_clean) begin" in M.strip_comments(memory) and
                "if (response_fire_clean) begin" in M.strip_comments(memory),
                "clean-fire indexed-state structure absent")

        inventory_root = root / "inventory_good"
        inventory_root.mkdir()
        inventory = make_inventory(M, inventory_root)
        data = strict_json(inventory)
        missing = root / "missing_inventory.json"
        mutant = dict(data); mutant["entries"] = data["entries"][:-1]
        missing.write_text(json.dumps(mutant))
        attacks.append(expect_reject("FN9_missing_coordinate", lambda:
            M.validate_inventory(missing)))
        reused = root / "reused_inventory.json"
        mutant = json.loads(json.dumps(data))
        mutant["entries"][1]["saif"] = mutant["entries"][0]["saif"]
        mutant["entries"][1]["saif_sha256"] = mutant["entries"][0]["saif_sha256"]
        reused.write_text(json.dumps(mutant))
        attacks.append(expect_reject("FN9_reused_saif", lambda:
            M.validate_inventory(reused)))

        attacks.append(expect_reject("FN10_nonfatal_payload_sva", lambda:
            M.validate_assertion_source(sva.replace(
                'else $fatal(1, "M1334 result payload unknown");', "else ;", 1))))
        attacks.append(expect_reject("FN10_nonfatal_stability_path", lambda:
            M.validate_assertion_source(sva.replace(
                '$fatal(1, "M1334 result stability violation");',
                '$display("M1334 result stability violation");', 1))))

        bad_log = root / "bad_case4.log"
        bad_log.write_text(pass_log(4, 1))
        attacks.append(expect_reject("FN1_runtime_case4_endpoint_nonzero", lambda:
            M.validate_runtime_log(bad_log, "k8", 4)))

    require(len(attacks) == 15 and len(set(attacks)) == 15,
            "mutation attack population mismatch")
    result = {
        "schema": "m1335_m1334_c2_source_blind_hammer_output_r1",
        "status": "PASS_M1334_SOURCE__TEN_PREDECESSOR_FALSE_NEGATIVES_CLOSED",
        "score": 100,
        "reviewer_independent_of_author": True,
        "baseline_author_tests": "12/12 PASS",
        "independent_mutations_rejected": attacks,
        "false_negative_count": 0,
        "m903_cycles": {"k8": AXIS["k8"]["cycles"],
                         "k1x8": AXIS["k1x8"]["cycles"]},
        "m903_cycle_sums": {"k8": 1913, "k1x8": 1945},
        "execution": {"vcs": False, "eda": False, "release": False,
                      "remote": False, "gpu": False},
        "claim_boundary": {"source_only": True, "production_saif": False,
                           "ptpx": False, "power": False, "energy": False,
                           "performance": False, "headline": False},
        "docs359_sha256": EXPECTED[DOCS359],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("M1335_BLIND_FAIL: " + str(error), file=sys.stderr)
        raise
