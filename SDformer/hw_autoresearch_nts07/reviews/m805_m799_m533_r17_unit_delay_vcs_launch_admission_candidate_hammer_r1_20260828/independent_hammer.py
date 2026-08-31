#!/usr/bin/env python3
"""Read-only M805 candidate hammer for the M799/M533 R17 VCS package.

The only subprocesses permitted here are the exact pinned-Python source
closure tests and the runner-owned pre-mkdir stub.  The stub exits before any
VCS identity/license probe and before the prospective result mkdir.
"""

import copy
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_load(path):
    def hook(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise ValueError("duplicate JSON key: " + key)
            obj[key] = value
        return obj
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=hook)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def require_regular(path, expected):
    st = path.lstat()
    require(stat.S_ISREG(st.st_mode) and not path.is_symlink(), "not regular: " + str(path))
    require(sha(path) == expected, "SHA mismatch: " + str(path))


def verify_object_double_seal(path):
    manifest = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require_regular(path, manifest.read_text(encoding="utf-8").split()[0])
    require_regular(manifest, outer.read_text(encoding="utf-8").split()[0])
    return sha(manifest), sha(outer)


def verify_directory_double_seal(directory):
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink(), "manifest missing")
    require(outer.is_file() and not outer.is_symlink(), "outer seal missing")
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = raw.split(None, 1)
        name = name.lstrip("* ")
        require_regular(directory / name, digest)
    require_regular(manifest, outer.read_text(encoding="utf-8").split()[0])
    return sha(manifest), sha(outer)


RUNNER_REL = "dc_handoff/scripts/run_vcs_m799_m533_m528_dead_write_only_1rw_unit_delay_r17_exact_sha.sh"
SOURCE_REL = "contracts/m799_m533_m528_dead_write_only_1rw_unit_delay_source_only_contract_r1_20260828.json"
CANDIDATE_REL = "contracts/m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_admission_candidate_r1_20260828.json"
STATIC_DIR_REL = "reviews/m799_m533_r17_unit_delay_source_static_hammer_r1_20260828"
HANDOFF_DIR_REL = "reviews/m799_m533_r17_py36_dryrun_repair_source_author_handoff_r1_20260828"
M770_DIR_REL = "reviews/m770_m533_r13_vcs_home_failure_fresh_hammer_r1_20260828"
M782_DIR_REL = "reviews/m782_m533_r14_premkdir_launch_boundary_failure_hammer_r1_20260828"
M794_DIR_REL = "reviews/m794_m533_r15_premkdir_undefined_function_failure_hammer_r1_20260828"
M797_DIR_REL = "reviews/m797_m795_m533_r16_function_closure_fresh_hammer_r1_20260828"
R15_RELEASE_REL = "contracts/m784_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"
RESULT_REL = "results/m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_r17_20260828"
RELEASE_REL = "contracts/m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json"
M528_RESULT_REL = "results/m528_h67_single_port_same_ledger_recompute_r4_20260827/m528_h67_single_port_same_ledger_recompute_result_r1.json"


EXPECTED = {
    "runner": "4d1b0a940ee44013bf09b0b8b41197b31f43c44d83732da8710233e679a7e0fe",
    "source": "0fe4fe0b4531bf5bfa6d69919f8845f935b389c0cf9762e7a0bf2dec5eb8eae5",
    "candidate": "fc36f2f56cc48316e941b0840cb1803f74ec9c65e7d5272ede47178acfead865",
    "static": "abd9a611c312d01bc1aa04d74ca2d2fe80ca578733e752db0e926d69aea8a5dd",
    "handoff": "6ac274ba1179f3f8d5ce1d0be08a20f6ed991c93020db63dad64ac70dea15609",
    "m770": "caba813792a8df3b1b9b72a7ddb7ec053096acab6188645b9d3c59a2ca8c3192",
    "m782": "ff7498279990537c7e60f886d44a3a6ec919aeb39d2fe5a9294a049f9a79bf6b",
    "m794": "bc244f11943089794151b16d5bf6bf56b4708e4df69d4c6bb0ecbcd2efe0def8",
    "m797": "7f9b7d492bd29329e3982afd3553d6aa7a9ba4d186d6fa21dc0912e754251074",
    "r15_release": "6c3d4a1ffef609765a387f45bdf502510a1d0d9ded6df0b281f50668d689fd08",
    "top": "726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1",
    "macro": "8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783",
    "binding": "db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983",
    "sva": "b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b",
    "tb": "d194f91293cf7e533e099d8b36956fb00db16402340c8e6e678059cb9adb0fd2",
    "foundry_v": "8343acf01604cf0c6ac4757fd268a8f409401e0b80964ff671b030281ebb444d",
    "closure": "7daeb06f0dd8d3e18d077fc8ad115911e2a223491f913c5b5c4f0b570a1093a8",
    "whitelist": "7bc11a6c4b7ce568de9a934c8178114ec8401a8e01125722c7173b92e75061d6",
    "dryrun": "20136d66506042453d40ba4564f1340580c46666d5e206641c871fccefa2fa36",
    "python": "9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f",
    "m528_result": "778c8e1bed6a19852c14bc61e00761f798008d67042b7a74efbaaffdde4b3de1",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


AUTH = {"vcs_runs": 1, "simv_runs": 1, "iverilog_runs": 0,
        "verilator_runs": 0, "dc_runs": 0, "formality_runs": 0,
        "pt_runs": 0, "ptpx_runs": 0, "cpu_runs": 0, "gpu_runs": 0,
        "network_or_remote_jobs": 0}


def validate_candidate(candidate, root=HW):
    require(candidate.get("schema") == "m799_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_admission_candidate_v1", "candidate schema")
    require(candidate.get("status") == "ADDITIVE_R17_PYTHON36_DRYRUN_REPAIR_SOURCE_CANDIDATE_ONLY__FRESH_HAMMERS_AND_TRUE_RELEASE_REQUIRED", "candidate status")
    require(candidate.get("launch_now") is False, "candidate launch boundary")
    require(candidate.get("authorization") == AUTH, "candidate authorization")
    identity = candidate.get("identity", {})
    require(identity.get("runner_sha256") == EXPECTED["runner"], "candidate runner SHA")
    require(identity.get("source_contract_sha256") == EXPECTED["source"], "candidate source SHA")
    require(candidate.get("macro_model_mode") == "foundry_UNIT_DELAY_functional", "candidate macro mode")
    unique = candidate.get("unique_attempt", {})
    require(unique.get("result_path") == RESULT_REL, "candidate result identity")
    require(not (root / unique["result_path"]).exists(), "candidate result collision")
    claim = candidate.get("claim_boundary", {})
    require(claim.get("functional_vcs_only") is True, "functional-only boundary")
    for key in ("functional_vcs_verified", "timing_verified", "rtl_verified",
                "speedup", "ppa", "energy", "system_or_paper_headline", "paper_citable"):
        require(claim.get(key) is False, "candidate claim boundary: " + key)


def run_test(command):
    completed = subprocess.run(command, cwd=str(HW), universal_newlines=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=240)
    require(completed.returncode == 0, "test failed: " + " ".join(command) + "\n" + completed.stderr[-1000:])
    return json.loads(completed.stdout)


def main():
    runner = HW / RUNNER_REL
    source_path = HW / SOURCE_REL
    candidate_path = HW / CANDIDATE_REL
    static_dir = HW / STATIC_DIR_REL
    handoff_dir = HW / HANDOFF_DIR_REL
    chain_dirs = [HW / M770_DIR_REL, HW / M782_DIR_REL, HW / M794_DIR_REL,
                  HW / M797_DIR_REL, handoff_dir, static_dir]

    for path, digest in ((runner, EXPECTED["runner"]),
                         (source_path, EXPECTED["source"]),
                         (candidate_path, EXPECTED["candidate"])):
        require_regular(path, digest)
        verify_object_double_seal(path)
    for directory in chain_dirs:
        verify_directory_double_seal(directory)

    candidate = strict_load(candidate_path)
    source = strict_load(source_path)
    static = strict_load(static_dir / "review.json")
    handoff = strict_load(handoff_dir / "handoff.json")
    m770 = strict_load(HW / M770_DIR_REL / "review.json")
    m782 = strict_load(HW / M782_DIR_REL / "review.json")
    m794 = strict_load(HW / M794_DIR_REL / "review.json")
    m797 = strict_load(HW / M797_DIR_REL / "review.json")
    r15_release = strict_load(HW / R15_RELEASE_REL)
    m528 = strict_load(HW / M528_RESULT_REL)
    binding = strict_load(HW / "rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json")
    validate_candidate(candidate)

    require(sha(static_dir / "review.json") == EXPECTED["static"], "M801 review SHA")
    require(sha(handoff_dir / "handoff.json") == EXPECTED["handoff"], "handoff SHA")
    require(sha(HW / M770_DIR_REL / "review.json") == EXPECTED["m770"], "M770 SHA")
    require(sha(HW / M782_DIR_REL / "review.json") == EXPECTED["m782"], "M782 SHA")
    require(sha(HW / M794_DIR_REL / "review.json") == EXPECTED["m794"], "M794 SHA")
    require(sha(HW / M797_DIR_REL / "review.json") == EXPECTED["m797"], "M797 SHA")
    require_regular(HW / R15_RELEASE_REL, EXPECTED["r15_release"])
    verify_object_double_seal(HW / R15_RELEASE_REL)
    require_regular(HW / M528_RESULT_REL, EXPECTED["m528_result"])
    require_regular(HW / "docs/359_DATE终局冻结_20260813.md", EXPECTED["docs359"])

    require(static.get("verdict") == "PASS" and static.get("score_100") == 100, "M801 verdict")
    require([static.get(k) for k in ("p0_count", "p1_count", "p2_count")] == [0, 0, 0], "M801 severity")
    require(static.get("identity", {}).get("runner_sha256") == EXPECTED["runner"], "M801 runner binding")
    require(static.get("identity", {}).get("source_contract_sha256") == EXPECTED["source"], "M801 source binding")
    require(static.get("identity", {}).get("candidate_sha256") == EXPECTED["candidate"], "M801 candidate binding")
    require(static.get("decision", {}).get("vcs_launch_authorized_now") is False, "M801 launch boundary")
    require(handoff.get("decision", {}).get("author_release_now") is False, "handoff release boundary")

    require(m770.get("verdict") == "PASS" and m770.get("score_out_of_100") == 100, "M770 disposition")
    require(m770.get("decision", {}).get("r14_launch_authorized_now") is False, "M770 launch boundary")
    require(m782.get("verdict") == "PASS_FAILURE_AUDIT" and m782.get("score_out_of_100") == 100, "M782 disposition")
    require(m782.get("decision", {}).get("r14_release_status") == "PERMANENTLY_WITHDRAWN_DO_NOT_EXECUTE_DO_NOT_CITE", "R14 withdrawal")
    require(m794.get("verdict") == "PASS_FAILURE_AUDIT" and m794.get("score_out_of_100") == 100, "M794 disposition")
    require(m794.get("decision", {}).get("r15_release_permanently_withdrawn") is True, "R15 withdrawal")
    require(m794.get("decision", {}).get("r15_attempt_consumed") is False, "R15 attempt boundary")
    require(m797.get("verdict") == "FAIL_SOURCE_GATE" and m797.get("score_100") == 98, "M797 disposition")
    require(m797.get("decision", {}).get("launch_release_authorized") is False, "R16 release boundary")
    require(not (HW / "results/m795_m533_m528_dead_write_only_1rw_unit_delay_vcs_r16_20260828").exists(), "R16 result exists")
    require(not (HW / "contracts/m795_m533_m528_dead_write_only_1rw_unit_delay_vcs_launch_release_r1_20260828.json").exists(), "R16 release exists")
    require(not (HW / RESULT_REL).exists(), "R17 attempt/result exists")
    require(not (HW / RELEASE_REL).exists(), "R17 release exists")
    require(r15_release.get("launch_now") is True, "historical R15 release identity changed")

    frozen = source["frozen_sources"]
    frozen_paths = {
        "top_r2": "rtl_m528_dw1rw/m528_dead_write_only_1rw_product_capture_island_r2.sv",
        "macro_adapter": "rtl_m528_dw1rw/m528_dw1rw_parent_scratch_9x128_macro.sv",
        "macro_binding_plan": "rtl_m528_dw1rw/m528_dw1rw_macro_binding_plan_r1_20260827.json",
        "sva_r2": "verif_m528_dw1rw/m528_dead_write_only_1rw_product_capture_assertions_r2.sv",
        "tb_r7": "tb_m528_dw1rw/tb_m528_dead_write_only_1rw_product_capture_r7.sv",
    }
    for key, rel in frozen_paths.items():
        require(frozen[key]["path"] == rel, "frozen path: " + key)
        require_regular(HW / rel, frozen[key]["sha256"])
    require(frozen["top_r2"]["sha256"] == EXPECTED["top"], "top identity")
    require(frozen["macro_adapter"]["sha256"] == EXPECTED["macro"], "macro identity")
    require(frozen["macro_binding_plan"]["sha256"] == EXPECTED["binding"], "binding identity")
    require(frozen["sva_r2"]["sha256"] == EXPECTED["sva"], "SVA identity")
    require(frozen["tb_r7"]["sha256"] == EXPECTED["tb"], "TB identity")
    require_regular(Path(frozen["foundry_v"]["path"]), EXPECTED["foundry_v"])

    runner_text = runner.read_text(encoding="utf-8")
    sha_edges = re.findall(r"(?m)^\s*require_regular_sha\s+([0-9a-f]{64})\s+", runner_text)
    require(len(sha_edges) == 76, "runner SHA edge count")
    compile_order = ['"${FOUNDRY_SLOW_V}"', '"${MACRO_RTL}"', '"${TOP_RTL}"', '"${SVA}"', '"${TB}"']
    compile_line = '"${FOUNDRY_SLOW_V}" "${MACRO_RTL}" "${TOP_RTL}" "${SVA}" "${TB}"'
    require(compile_line in runner_text, "exact VCS file list/order")
    require(runner_text.count("+define+UNIT_DELAY") == 1, "UNIT_DELAY compile define count")
    require("+notimingcheck" not in runner_text and "+no_notifier" not in runner_text, "timing bypass")
    require(runner_text.count('[[ ! -e "${RESULT_DIR}" ]]') >= 3, "result absence rechecks")
    require('if mkdir -- "${RESULT_DIR}"; then' in runner_text, "atomic mkdir")
    sva_text = (HW / frozen_paths["sva_r2"]).read_text(encoding="utf-8")
    require("ap_read_xor_write: assert property (!(scratch_read && scratch_write));" in sva_text, "1RW nonoverlap SVA")
    require(binding.get("forbidden", []).count("concurrent read plus write") == 1, "binding nonoverlap")

    capacity = m528["capacity"]["m505_dead_write_only_1rw"]
    aggregate = m528["aggregate_cycles"]
    require(m528["claim_boundary"]["exact_cpu_cycle_recompute"] is True, "same-ledger class")
    require(m528["claim_boundary"]["rtl"] is False and m528["claim_boundary"]["vcs"] is False, "same-ledger boundary")
    require(capacity["macro_rounded_total_bytes"] == 213376 and capacity["budget_margin_bytes"] == 32384, "240KiB capacity")
    require(m528["capacity"]["budget_bytes"] == 245760, "240KiB budget")
    require(aggregate["m505_dead_write_only_1rw_cycles"] == 435293339, "same-ledger cycles")
    require(abs(aggregate["speedup_vs_m468_strong_zero"] - 1.7467534301047505) < 1e-15, "same-ledger speedup")
    top_text = (HW / frozen_paths["top_r2"]).read_text(encoding="utf-8")
    require("resident signed19 psum prior" in top_text and "issue_psum_prior" in top_text, "signed19 local psum boundary")
    require(m528["capacity"]["m505_dead_write_only_1rw"]["logical_items"]["psum"] == 116736, "Acc24-byte capacity charge")

    py = "/usr/libexec/platform-python3.6"
    require_regular(Path(py), EXPECTED["python"])
    closure = HW / "verif_m528_dw1rw/test_m799_r17_runner_function_closure.py"
    whitelist = HW / "verif_m528_dw1rw/m799_r17_external_command_whitelist.json"
    dryrun = HW / "verif_m528_dw1rw/test_m799_r17_runner_premkdir_dry_run.py"
    require_regular(closure, EXPECTED["closure"])
    require_regular(whitelist, EXPECTED["whitelist"])
    require_regular(dryrun, EXPECTED["dryrun"])
    outputs = {}
    outputs["positive"] = run_test([py, str(closure), str(runner), str(whitelist)])
    for mutation in ("delete-definition", "rename-definition", "inject-stale"):
        outputs[mutation] = run_test([py, str(closure), str(runner), str(whitelist),
                                      "--mutation", mutation, "--expect-fail"])
    outputs["dryrun"] = run_test([py, str(dryrun), str(runner)])
    require(outputs["positive"]["pass"] is True, "closure positive")
    require(len(outputs["positive"]["definitions"]) == 31, "definition count")
    require(len(outputs["positive"]["custom_calls"]) == 230, "call count")
    for mutation in ("delete-definition", "rename-definition", "inject-stale"):
        require(outputs[mutation]["observed_pass"] is False, mutation + " attack")
    require(outputs["dryrun"]["runner_rc"] == 86, "stub rc")
    require(outputs["dryrun"]["events"] == ["stub_collision_initial", "stub_cgroup",
            "stub_resource", "stub_collision_final", "live_probe_boundary_stop"], "stub events")
    require(all(value == 0 for value in outputs["dryrun"]["totals"].values()), "stub side effects")

    wrong_sha = copy.deepcopy(candidate)
    wrong_sha["identity"]["runner_sha256"] = "0" * 64
    try:
        validate_candidate(wrong_sha)
    except RuntimeError:
        wrong_sha_rejected = True
    else:
        wrong_sha_rejected = False
    require(wrong_sha_rejected, "wrong-SHA mutation accepted")

    with tempfile.TemporaryDirectory(prefix="m805_candidate_collision.") as raw:
        fake_root = Path(raw)
        collision = fake_root / RESULT_REL
        collision.mkdir(parents=True)
        try:
            validate_candidate(copy.deepcopy(candidate), root=fake_root)
        except RuntimeError:
            collision_rejected = True
        else:
            collision_rejected = False
    require(collision_rejected, "directory collision mutation accepted")

    try:
        json.loads('{"schema":"a","schema":"b"}', object_pairs_hook=lambda pairs: _duplicate_hook(pairs))
    except ValueError:
        duplicate_rejected = True
    else:
        duplicate_rejected = False
    require(duplicate_rejected, "duplicate-key mutation accepted")
    require(not (HW / RESULT_REL).exists() and not (HW / RELEASE_REL).exists(), "post-audit result/release side effect")

    print(json.dumps({
        "status": "PASS_M805_CANDIDATE_HAMMER_MECHANICAL_CHECKS",
        "candidate_sha256": EXPECTED["candidate"],
        "source_static_sha256": EXPECTED["static"],
        "pinned_python": "3.6.8",
        "closure": {"definitions": 31, "calls": 230, "positive": True,
                    "three_negative_attacks_rejected": True, "external_commands": 20},
        "premkdir_stub": {"rc": 86, "events": outputs["dryrun"]["events"],
                          "side_effect_totals": outputs["dryrun"]["totals"]},
        "contract_attacks": {"wrong_sha_rejected": wrong_sha_rejected,
                             "directory_collision_rejected": collision_rejected,
                             "duplicate_key_rejected": duplicate_rejected},
        "sha_edges": 76,
        "r17_result_absent": True,
        "r17_release_absent": True,
        "vcs_or_license_queries": 0,
        "docs359_sha256": EXPECTED["docs359"],
    }, indent=2, sort_keys=True))


def _duplicate_hook(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate JSON key: " + key)
        obj[key] = value
    return obj


if __name__ == "__main__":
    main()
