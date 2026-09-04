#!/usr/bin/env python3
"""Build the source-only Motion final-checkpoint hardware-rebind readiness map.

This program deliberately does not access the A800 host, deserialize a
checkpoint, inspect a GPU, run validation/capture, copy data, or launch EDA.
It inventories the local source primitives that can be reused after the M1171
selection receipt receives a different-author result hammer and fails closed
on the production entrypoints that do not yet exist for the new checkpoint.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat
import sys
from typing import Any


REPO = Path(__file__).resolve().parents[2]
HW = REPO / "hw_autoresearch_nts07"
OUTPUT = HW / "reviews/m1172_motion_final_rebind_execution_readiness_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

REMOTE_REPO = Path("/root/private_data/work/sdformer_codex/SDformer")
REMOTE_RUN = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/results/"
    "date_two_contribution_full30_20260826/c12_binary_motion_ttx"
)
REMOTE_CONFIG = REMOTE_REPO / (
    "neuron_experiments/H9_bipolar_self_attention/configs/generated/"
    "dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml"
)
REMOTE_CONFIG_SHA = "c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955"
REMOTE_EP29_CANDIDATE = REMOTE_RUN / "checkpoint_epoch29.pth"
REMOTE_BINDER_RESULT = HW / "results/m1171_motion_final_checkpoint_selection_rebind_binder_r4_20260830"

SOURCES = {
    "m1171_binder_launcher": (
        "hw_autoresearch_nts07/scripts/run_m1171_motion_final_checkpoint_binder_remote_one_shot_source.py",
        "ec3483ec484e3e61c7bb27530682b597837e375c2649403f9b27617b4b54c695",
    ),
    "m1167_binder": (
        "hw_autoresearch_nts07/scripts/build_m1167_motion_final_checkpoint_selection_rebind_binder_r3.py",
        "7ea88b861ad54f6029f2631766a7da21b3626054217d36c27c4509293ce35d89",
    ),
    "standard_valid": (
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/run_h9_standard_valid825_eval.py",
        "a9207ffb6366ac203db0f7809e3d4a154df06859df26b08ffe18a50d182ad0d9",
    ),
    "deploy_numeric_primitives": (
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/run_dsec_fullres_paper_w15_deploy_followup.py",
        "45b2c42e886f7df91f4b50b175168237a5ebc879c2ede2176c52717b6d3a65ed",
    ),
    "ordered_profiler": (
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py",
        "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684",
    ),
    "attention_qk_writer": (
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/h67_bit_trace.py",
        "75c9134061aa06c8050389cbaac0a80a7956911cda0f8ce7b4144ba40ab3f58e",
    ),
    "c1_capture_ep35_frozen": (
        "hw_autoresearch_nts07/system_simulator/scripts/trace_m40_bottleneck_packed_sources.py",
        "b02ac10fb95e68fa2871b74330d6f39d7d3d8cbfa6440990d43ec832e943bf19",
    ),
    "decoder_capture_ep35_frozen": (
        "neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m699_h67_ep35_multisequence_decoder_payload.py",
        "fdd88b0285c329ea13466093479b2dc52e9242a7312d4fdce14903cdef1a1769",
    ),
    "decoder_address_core": (
        "hw_autoresearch_nts07/system_simulator/scripts/build_m1105dr2_decoder_only_address_timed_source.py",
        "b2d8ef4139283de06b7e332429bdf752ad16122ffbeda0ff7d75bce6d816a5c4",
    ),
    "decoder_ep35_runner_frozen": (
        "hw_autoresearch_nts07/system_simulator/scripts/run_m1111dr2_m1105dr2_decoder_only_production_zero_arg.py",
        "1167258c228631b73ca1784ae57db19e8f0fbe709efa34f369585c508bc9d746",
    ),
    "c1_ep35_runner_frozen": (
        "hw_autoresearch_nts07/system_simulator/scripts/run_m1161ca_c1_production_real_replay_driver_one_shot_source.py",
        "d7ffb8dbab289e83fd8a32f4ed5244cd005a4b6d0785b586df932fd6a97ee20d",
    ),
    "c1_int8_export_ep35_frozen": (
        "hw_autoresearch_nts07/system_simulator/scripts/export_m41_h67_ep35_bottleneck_int8.py",
        "bc272c5e1449fb745fe200313f25e97c293ad971fa8856d9aad13dfc89785a5e",
    ),
    "attention_trace_audit": (
        "hw_autoresearch_nts07/scripts/audit_h67_bit_trace.py",
        "fb577011e72bb127cf4160aca5fb1902d627be4ea3de508cfeca8f091d1870f2",
    ),
    "rqtb_replay": (
        "hw_autoresearch_nts07/scripts/profile_h67_rqtb_temporal_pair_multicast.py",
        "61d5e79698ffa96cf41ef1341505bdc31a4cc7c4fa621aa8e33309f2c83327a2",
    ),
    "c1_cohort": (
        "hw_autoresearch_nts07/results/m36_h67_ep35_patch_embed_profile_s10_r1_20260822/sample_workload.csv",
        "bb45f8b5406e34835f05e1993692d8cba241c748471037d75fcfa1ec2478cffa",
    ),
    "decoder_cohort": (
        "hw_autoresearch_nts07/contracts/m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json",
        "43d3b024c1a78d8bc2422af3846c9a376a67bedbecb2ff7396a17bc51ec68fc7",
    ),
}


class ReadinessError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise ReadinessError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    value = path.lstat()
    require(stat.S_ISREG(value.st_mode) and not path.is_symlink(),
            f"{label} must be a non-symlink regular file")


def source_inventory() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for name, (relative, expected) in SOURCES.items():
        path = REPO / relative
        regular(path, name)
        observed = sha256(path)
        require(observed == expected, f"source/data identity drift: {name}")
        output[name] = {
            "path": relative,
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    return output


def matrix() -> dict[str, Any]:
    inventory = source_inventory()
    regular(DOCS359, "protected docs/359")
    require(sha256(DOCS359) == DOCS359_SHA, "protected docs/359 SHA drift")
    stages = [
        {
            "id": "R0",
            "m1125k": "E0",
            "state": "WAIT_M1171_RESULT_AND_DIFFERENT_AUTHOR_RESULT_HAMMER",
            "reusable": ["m1171_binder_launcher", "m1167_binder"],
            "missing": ["sealed M1171 production result", "different-author result hammer"],
            "may_execute_now": False,
        },
        {
            "id": "R1",
            "m1125k": "E0,E8-root",
            "state": "PRIMITIVES_EXIST__PRODUCTION_ENTRYPOINT_MISSING",
            "reusable": ["ordered_profiler:load_config/build_model/validate_h9_load_audit/h9_module_counts/threshold_training_semantics"],
            "missing": [
                "generic selected-checkpoint launcher",
                "ordered named-module topology plus parameter shape/dtype digest",
                "numeric signature covering precision/rank/T10/value classes",
                "model commit plus dirty-diff digest",
                "valid-list/preprocess identity manifest",
            ],
            "may_execute_now": False,
        },
        {
            "id": "R2A",
            "m1125k": "E1",
            "state": "STANDARD_VALID_EXISTS__DEPLOY_LAUNCHER_MISSING",
            "reusable": [
                "standard_valid (already consumed by R0 selection)",
                "deploy_numeric_primitives:make_deploy_configs/run_or_reuse_eval/protocol_from_profile",
            ],
            "missing": [
                "generic final-selected Motion dyadic plus hardware-order launcher",
                "derived-config identity seal and exact-load/topology recheck",
                "hardware-order scope remains attention-core numeric, not full-network RTL-exact",
            ],
            "may_execute_now": False,
        },
        {
            "id": "R2B",
            "m1125k": "E2",
            "state": "BLOCKED_SOURCE_GAP__NO_ONE_LOAD_UNIFIED_CAPTURE_ENTRYPOINT",
            "reusable": [
                "ordered_profiler for execution/operator/ATLIF/Linear/Conv2d/patch/stage activity",
                "attention_qk_writer for Q/K/gate payload",
                "c1_capture_ep35_frozen hook/writer design only",
                "decoder_capture_ep35_frozen hook/bitpack/theta design only",
            ],
            "missing": [
                "one model construction/load shared by the union of C1 S10 and decoder S3x10 cohorts",
                "checkpoint-parametric C1 writer (existing M40 rejects non-ep35 SHA/config)",
                "checkpoint-parametric decoder writer (existing M699 rejects non-ep35 SHA/config/host)",
                "direct BN taps and raw Q/K/V-or-architecture-absent manifest",
                "one atomic manifest joining all call order, sample and tensor identities",
                "race-free ownership against the legacy M511 GPU watcher",
            ],
            "may_execute_now": False,
        },
        {
            "id": "R3A",
            "m1125k": "E3",
            "state": "CORE_REUSABLE__EP29_LAUNCH_AUTHORITY_MISSING",
            "reusable": ["c1_ep35_runner_frozen algorithms and O(axes) sink"],
            "missing": ["new capture-bound schedule/digest authority and ep29 runner contract"],
            "depends_on": ["R2B", "R3D"],
            "may_execute_now": False,
        },
        {
            "id": "R3B",
            "m1125k": "E4",
            "state": "CORE_REUSABLE__EP29_PAYLOAD_BINDING_MISSING",
            "reusable": ["decoder_address_core mapper/arbitration model"],
            "missing": ["new 120-call payload authority, D1 theta identity/miter and fresh runner contract"],
            "depends_on": ["R2B", "R3D"],
            "may_execute_now": False,
        },
        {
            "id": "R3C",
            "m1125k": "E5,E6",
            "state": "ANALYZERS_REUSABLE_AFTER_CAPTURE",
            "reusable": ["attention_trace_audit", "rqtb_replay", "ordered profiler aggregate ledgers"],
            "missing": ["capture-bound manifests and direct FC/BN/patch fanout adapters"],
            "depends_on": ["R2B"],
            "may_execute_now": False,
        },
        {
            "id": "R3D",
            "m1125k": "E8",
            "state": "EP35_EXPORTER_FROZEN__EP29_EXPORTER_MISSING",
            "reusable": ["c1_int8_export_ep35_frozen arithmetic/range methodology only"],
            "missing": ["selected-checkpoint weight/bias export, overflow proof, encoding miter and fit receipt"],
            "depends_on": ["R1", "R2B"],
            "may_execute_now": False,
        },
        {
            "id": "R4A_R4B_R5_R6",
            "m1125k": "E7 and final join",
            "state": "WAIT_UPSTREAM",
            "missing": ["real-trace VCS/SAIF, name mapping, PTPX, decoder-complete Table A and final hammer"],
            "depends_on": ["R2A", "R3A", "R3B", "R3C", "R3D"],
            "may_execute_now": False,
        },
    ]
    return {
        "schema": "m1172_motion_final_rebind_execution_readiness_r1_v1",
        "status": "SOURCE_AUDIT_COMPLETE__WAIT_M1171_RESULT_HAMMER__R1_R2A_R2B_PRODUCTION_ENTRYPOINTS_MISSING",
        "score": 100,
        "remote_binding": {
            "repo": str(REMOTE_REPO),
            "run_dir": str(REMOTE_RUN),
            "training_config": str(REMOTE_CONFIG),
            "training_config_sha256": REMOTE_CONFIG_SHA,
            "ep29_candidate_path_not_final_until_binder": str(REMOTE_EP29_CANDIDATE),
            "selection_receipt": str(REMOTE_BINDER_RESULT),
            "selected_checkpoint_rule": "read exact path/SHA/size/mtime from hammered M1171 final_checkpoint_selection.json; never infer ep29 from presence",
        },
        "fixed_cohorts": {
            "c1": {
                "samples": 10,
                "manifest": inventory["c1_cohort"],
                "rule": "frozen first-ten zurich_city_09_a sample keys from M36",
            },
            "decoder": {
                "sequences": ["interlaken_01_a", "thun_01_b", "zurich_city_12_a"],
                "samples_per_sequence": 10,
                "selection": "lexicographic files then round(i*(N-1)/9), i=0..9",
                "manifest": inventory["decoder_cohort"],
            },
            "unified_union": {
                "expected_forward_samples": 40,
                "deduplicate_by_source_sha256": True,
                "order": "C1 cohort first, then decoder sequences in declared order",
            },
        },
        "source_inventory": inventory,
        "readiness_matrix": stages,
        "shortest_safe_dag": [
            "S0 wait for current valid825 completion",
            "S1 run exact M1171 binder once",
            "S2 different-author hammer M1171 small result",
            "S3 author plus hammer R1 signature, R2A generic deploy and R2B unified-capture sources without GPU",
            "S4 resolve legacy M511 watcher ownership; no check-then-race is accepted",
            "S5 run R1 then one-load R2B capture first; run dyadic/hardware-order valid only after capture releases GPU",
            "S6 fan out R3A/R3B/R3C/R3D on CPU with at most three workers",
            "S7 run real-trace VCS/SAIF then PTPX",
            "S8 build decoder-complete memory-inclusive Table A and independent final hammer",
        ],
        "gpu_collision_rule": {
            "status": "UNRESOLVED_P0",
            "reason": "legacy M511 watcher does not consume an M1172 shared lease, so a process-list check alone has a time-of-check/time-of-use race",
            "required_fix": "cancel/retire the legacy watcher or patch it and the final-rebind launcher to honor one shared flock before either may start model construction",
            "priority": "unified capture precedes any legacy M511 capture because it subsumes decoder payload and avoids another checkpoint load",
        },
        "execution": {
            "remote_access": False,
            "checkpoint_read_or_copy": False,
            "gpu": False,
            "capture": False,
            "cpu_replay": False,
            "eda": False,
            "docs359_modified": False,
        },
        "claim_boundary": {
            "source_readiness_only": True,
            "hardware_rebind_authorized": False,
            "checkpoint_selected": False,
            "accuracy": False,
            "hardware_speedup": False,
            "system_speedup": False,
            "energy": False,
            "paper_citable_result": False,
        },
    }


def write_result(result: dict[str, Any]) -> None:
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(),
            f"fresh output required: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "readiness.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# M1172 Motion final-checkpoint hardware-rebind readiness",
        "",
        f"Status: `{result['status']}`.",
        "",
        "| stage | E0-E8 | state | production now |",
        "|---|---|---|---:|",
    ]
    for row in result["readiness_matrix"]:
        lines.append(
            f"| {row['id']} | {row['m1125k']} | {row['state']} | "
            f"{'yes' if row['may_execute_now'] else 'no'} |"
        )
    lines.extend([
        "",
        "The shortest safe path is binder result hammer -> source closure -> one-load unified capture -> CPU fanout -> SAIF/PTPX -> Table A. The legacy M511 watcher must share a lock or be retired before GPU launch.",
        "",
        "This receipt is source/readiness evidence only; it selects no checkpoint and authorizes no remote, GPU, capture, replay or EDA action.",
    ])
    (OUTPUT / "readiness.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (OUTPUT / "RUN_COMPLETE.txt").write_text(
        "PASS_M1172_SOURCE_READINESS_AUDIT__WAIT_M1171_RESULT_HAMMER__NO_PRODUCTION\n",
        encoding="utf-8",
    )
    members = ["RUN_COMPLETE.txt", "readiness.json", "readiness.md"]
    manifest = OUTPUT / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha256(OUTPUT / name)}  {name}\n" for name in members),
        encoding="utf-8",
    )
    (OUTPUT / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")


def main() -> int:
    require(len(sys.argv) == 1, "M1172 accepts zero arguments")
    result = matrix()
    write_result(result)
    print("PASS_M1172_SOURCE_READINESS_AUDIT__WAIT_M1171_RESULT_HAMMER__NO_PRODUCTION")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
