#!/usr/bin/env python3
"""Independent read-only M1171 result hammer.

The program validates an offline snapshot of the remote M1171 output plus the
five small profile inputs, configuration and ranking.  It deliberately does
not connect to the remote host and cannot start GPU, training, replay or EDA.
"""
from __future__ import annotations

import argparse
import copy
import csv
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Callable


EPOCHS = (9, 14, 19, 24, 29)
PAYLOADS = {
    "RUN_COMPLETE.txt", "e0_e8_rebind_targets.json",
    "final_checkpoint_selection.json", "five_checkpoint_metrics.csv",
}
ALL_MEMBERS = PAYLOADS | {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
TOKEN = (
    "PASS_M1167_FINAL_CHECKPOINT_SELECTED_R3_CANONICAL_EPOCH_NAMES__"
    "INDEPENDENT_RESULT_HAMMER_REQUIRED__NO_HARDWARE_REBIND_AUTHORITY\n"
)
METRICS = (
    "AEE", "AAE", "AAE_Benchmark", "AEE_PE1", "AEE_PE2", "AEE_PE3",
    "AEE_outliers", "DSEC_Fl",
)
ZERO_AUDIT = (
    "missing_count", "unexpected_count", "overlay_missing_count",
    "overlay_unexpected_count",
)
CONFIG_PATH = (
    "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/"
    "H9_bipolar_self_attention/configs/generated/"
    "dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml"
)
RUN_PATH = (
    "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/"
    "H9_bipolar_self_attention/results/"
    "date_two_contribution_full30_20260826/c12_binary_motion_ttx"
)
EXPECTED_CLAIMS = {
    "checkpoint_copied": False,
    "eda_started_by_binder": False,
    "final_selection_bound": True,
    "gpu_started_by_binder": False,
    "hardware_rebind_authorized": False,
    "hardware_replay_complete": False,
    "hardware_speedup": False,
    "independent_hammer_required_before_hardware_rebind": True,
    "power_or_energy": False,
    "standard_valid825_bound": True,
    "system_speedup": False,
}


class HammerError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise HammerError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def no_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in out, f"duplicate JSON key: {key}")
        out[key] = value
    return out


def reject_constant(value: str) -> None:
    raise HammerError(f"non-finite JSON constant: {value}")


def strict_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=no_duplicate, parse_constant=reject_constant)


def typed_int(value: Any, expected: int, label: str) -> None:
    require(type(value) is int and value == expected,
            f"{label} must be exact non-bool int {expected}")


def finite(value: Any, label: str) -> float:
    require(type(value) in (int, float) and not isinstance(value, bool),
            f"{label} must be numeric")
    out = float(value)
    require(math.isfinite(out), f"{label} must be finite")
    return out


def reseal(root: Path) -> None:
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(
        f"{sha256(root / name)}  {name}\n" for name in sorted(PAYLOADS)
    ), encoding="utf-8")
    (root / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(manifest)}  SHA256SUMS\n", encoding="utf-8")


def verify_seal(root: Path) -> None:
    require(root.is_dir() and not root.is_symlink(), "result must be a real directory")
    observed = {p.name for p in root.iterdir()}
    require(observed == ALL_MEMBERS, f"exact member set mismatch: {sorted(observed)}")
    for item in root.iterdir():
        require(item.is_file() and not item.is_symlink(), f"non-regular member: {item.name}")
    outer = (root / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8").split()
    require(outer == [sha256(root / "SHA256SUMS"), "SHA256SUMS"], "outer seal mismatch")
    rows: dict[str, str] = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  ([A-Za-z0-9_.-]+)", line)
        require(match is not None, "manifest syntax mismatch")
        digest, name = match.groups()
        require(name in PAYLOADS and name not in rows, "manifest member mismatch")
        rows[name] = digest
    require(set(rows) == PAYLOADS, "manifest population mismatch")
    for name, digest in rows.items():
        require(sha256(root / name) == digest, f"payload digest mismatch: {name}")
    require((root / "RUN_COMPLETE.txt").read_text(encoding="utf-8") == TOKEN,
            "terminal token mismatch")


def expected_targets() -> list[dict[str, str]]:
    values = [
        ("E0", "final checkpoint and deployment identity", "BOUND_BY_THIS_RECEIPT",
         "independent hammer must verify selection/config/profile/checkpoint identity"),
        ("E1", "standard plus dyadic/quantized/hardware-order valid825",
         "STANDARD_VALID825_BOUND__DEPLOYMENT_NUMERICS_INVALIDATED",
         "run dyadic/quantized and RTL-exact accuracy without retuning valid825"),
        ("E2", "unified ordered full-network capture", "INVALIDATED_RECAPTURE_REQUIRED",
         "single selected-checkpoint load; fixed C1 and decoder cohorts"),
        ("E3", "C1 four-Conv ledger and official baseline replay", "INVALIDATED_REPLAY_REQUIRED",
         "selected-checkpoint 51.84M source-row same-ledger replay"),
        ("E4", "decoder D0-D3 payload, numeric miter and address cycles", "INVALIDATED_REPLAY_REQUIRED",
         "decoder-complete selected-checkpoint payload and address replay"),
        ("E5", "ATLIF/FC/patch/BN activity and traffic", "INVALIDATED_REPLAY_REQUIRED",
         "derive activity only from sealed E2 capture"),
        ("E6", "attention/RQTB NPZ, exact miter and Amdahl", "INVALIDATED_REPLAY_REQUIRED",
         "selected-checkpoint Q/K capture and Fixed-RQTB replay"),
        ("E7", "real-trace SAIF/PTPX and decoder-complete Table A", "INVALIDATED_REPLAY_REQUIRED",
         "E2-E6 complete, then real-trace VCS/SAIF/PTPX and same-resource join"),
        ("E8", "weight/range/compression re-admission", "INVALIDATED_REPLAY_REQUIRED",
         "selected-checkpoint export, overflow proof, encoding miter and fit"),
    ]
    return [{"id": i, "target": t, "state_after_selection": s, "next_gate": n}
            for i, t, s, n in values]


def validate_bundle(root: Path, fixture: Path, readback: dict[str, Any]) -> dict[str, Any]:
    verify_seal(root)
    selection = strict_json(root / "final_checkpoint_selection.json")
    require(set(selection) == {
        "claim_boundary", "configuration", "e0_e8_invalidation_and_rebind_targets",
        "five_checkpoint_metric_table", "ranking", "run", "schema", "selected",
        "selection_rule", "source_hardening", "status",
    }, "selection top-level schema mismatch")
    require(selection["schema"] == "m1167_motion_final_checkpoint_selection_rebind_binder_r3_v1",
            "schema mismatch")
    require(selection["status"] ==
            "READY_FINAL_CHECKPOINT_SELECTION_R3_CANONICAL_EPOCH_NAMES__HARDWARE_REBIND_NOT_AUTHORIZED",
            "status mismatch")
    require(selection["claim_boundary"] == EXPECTED_CLAIMS, "claim boundary mismatch")
    require(selection["run"] == {
        "absolute_path": RUN_PATH,
        "label": "date_two_contribution_full30_20260826/c12_binary_motion_ttx",
        "predeclared_epochs": list(EPOCHS),
    }, "run identity mismatch")
    require(selection["selection_rule"] == {
        "all_five_profiles_required": True,
        "primary": "minimum exact standard-valid825 AEE",
        "ranking_mode": "aee",
        "tie_break": "lowest epoch",
        "valid825_reuse_for_retuning_forbidden": True,
    }, "selection rule mismatch")

    config = selection["configuration"]
    rr_config = readback["configuration"]
    require(config == {"absolute_path": CONFIG_PATH, **rr_config}, "configuration identity mismatch")
    require(sha256(fixture / "dsec_fullres_w15_two_contrib_c12_binary_motion_ttx_nb0ep29_ft30_20260826.yml")
            == rr_config["sha256"], "downloaded configuration digest mismatch")

    ranking = selection["ranking"]
    rr_ranking = readback["ranking"]
    require(ranking == {
        "absolute_path": f"{RUN_PATH}/profile_ranking_valid825.md",
        **rr_ranking, "ranking_mode": "aee", "ordered_epochs": [29, 24, 19, 14, 9],
    }, "ranking identity mismatch")
    ranking_path = fixture / "profile_ranking_valid825.md"
    require(sha256(ranking_path) == rr_ranking["sha256"], "downloaded ranking digest mismatch")
    declarations = re.findall(r"^Ranking mode: `([^`]+)`\.$",
                              ranking_path.read_text(encoding="utf-8"), re.MULTILINE)
    require(declarations == ["aee"], "ranking declaration mismatch")
    ranking_rows = [(int(a), int(b)) for a, b in re.findall(
        r"^\|\s*(\d+)\s*\|\s*(\d+)\s*\|", ranking_path.read_text(encoding="utf-8"), re.MULTILINE)]
    require(ranking_rows == [(1, 29), (2, 24), (3, 19), (4, 14), (5, 9)],
            "ranking rows mismatch")

    rows = selection["five_checkpoint_metric_table"]
    require(type(rows) is list and [row.get("epoch") for row in rows] == list(EPOCHS),
            "five-row epoch population/order mismatch")
    profile_rows: dict[int, dict[str, Any]] = {}
    for row in rows:
        epoch = row["epoch"]
        rr = readback["epochs"][str(epoch)]
        checkpoint = row["checkpoint"]
        checkpoint_path = f"{RUN_PATH}/checkpoint_epoch{epoch}.pth"
        require(checkpoint == {
            "absolute_path": checkpoint_path,
            "size_bytes": rr["checkpoint_size_bytes"],
            "mtime_ns": rr["checkpoint_mtime_ns"],
            "sha256": rr["checkpoint_sha256"],
        }, f"epoch{epoch} checkpoint identity mismatch")
        profile_path = fixture / f"spike_profile_epoch{epoch}.json"
        require(sha256(profile_path) == rr["profile_sha256"],
                f"epoch{epoch} downloaded profile digest mismatch")
        profile = strict_json(profile_path)
        profile_rows[epoch] = profile
        typed_int(profile.get("samples"), 825, f"epoch{epoch} samples")
        typed_int(profile.get("total_spikes"), int(profile["total_spikes"]),
                  f"epoch{epoch} total_spikes")
        require(profile["total_spikes"] > 0, f"epoch{epoch} spikes must be positive")
        identity = profile.get("artifact_identity")
        require(identity == {
            "config_path": CONFIG_PATH,
            "config_sha256": rr_config["sha256"],
            "checkpoint_path": checkpoint_path,
            "checkpoint_size": rr["checkpoint_size_bytes"],
            "checkpoint_mtime_ns": rr["checkpoint_mtime_ns"],
            "checkpoint_sha256": rr["checkpoint_sha256"],
        }, f"epoch{epoch} profile artifact identity mismatch")
        audit = profile.get("checkpoint_load_audit")
        require(type(audit) is dict and audit.get("checkpoint") == checkpoint_path,
                f"epoch{epoch} load audit mismatch")
        for key in ZERO_AUDIT:
            typed_int(audit.get(key), 0, f"epoch{epoch} {key}")
        typed_int(audit.get("checkpoint_overlay_keys"), 210,
                  f"epoch{epoch} checkpoint_overlay_keys")
        typed_int(audit.get("model_overlay_keys"), 210, f"epoch{epoch} model_overlay_keys")
        require(profile.get("module_counts") == {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
                f"epoch{epoch} module counts mismatch")
        receipt_profile = row["profile"]
        require(receipt_profile == {
            "absolute_path": f"{RUN_PATH}/standard_valid825/epoch{epoch}/spike_profile.json",
            "size_bytes": rr["profile_size_bytes"], "mtime_ns": rr["profile_mtime_ns"],
            "sha256": rr["profile_sha256"], "samples": 825,
            "artifact_identity_exact": True, "load_missing_count": 0,
            "load_unexpected_count": 0, "overlay_missing_count": 0,
            "overlay_unexpected_count": 0,
            "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
        }, f"epoch{epoch} receipt profile identity mismatch")
        require(row["accuracy_metrics"] == {key: str(Decimal(str(profile["metrics"][key])))
                                             for key in METRICS},
                f"epoch{epoch} accuracy metric mismatch")
        activity = row["activity"]
        require(activity["total_spikes"] == profile["total_spikes"],
                f"epoch{epoch} total spike mismatch")
        for key, source_key in (("global_firing_rate", "global_firing_rate"),
                                ("dense_flops", "dense_flops"),
                                ("effective_flops", "effective_flops"),
                                ("spike_energy_proxy_uj", "energy_uj")):
            require(finite(activity[key], key) == finite(profile[source_key], source_key),
                    f"epoch{epoch} activity mismatch: {key}")
        expected_sparsity = 1.0 - float(profile["effective_flops"]) / float(profile["dense_flops"])
        require(activity["effective_sparsity"] == expected_sparsity,
                f"epoch{epoch} effective sparsity mismatch")
        require(activity["energy_scope"] == "spike_activity_proxy_not_hardware_energy",
                f"epoch{epoch} energy scope mismatch")

    expected_order = sorted(EPOCHS,
        key=lambda e: (Decimal(profile_rows[e]["metrics"]["AEE"]), e))
    require(expected_order == [29, 24, 19, 14, 9], "independent AEE ordering mismatch")
    selected_row = next(row for row in rows if row["epoch"] == expected_order[0])
    require(selection["selected"] == {
        "epoch": 29, "checkpoint": selected_row["checkpoint"],
        "accuracy_metrics": selected_row["accuracy_metrics"],
        "activity": selected_row["activity"],
        "profile_sha256": selected_row["profile"]["sha256"],
    }, "selected checkpoint mismatch")

    targets = expected_targets()
    require(selection["e0_e8_invalidation_and_rebind_targets"] == targets,
            "embedded E0-E8 policy mismatch")
    require(strict_json(root / "e0_e8_rebind_targets.json") == targets,
            "detached E0-E8 policy mismatch")

    hard = selection["source_hardening"]
    require(hard["revision"] == "r3" and
            hard["sealed_r1_dependency_sha256"] == readback["source_sha256"][
                "build_m1163_motion_final_checkpoint_selection_rebind_binder.py"] and
            hard["sealed_r2_dependency_sha256"] == readback["source_sha256"][
                "build_m1166_motion_final_checkpoint_selection_rebind_binder_r2.py"],
            "sealed source chain mismatch")
    require(hard["canonical_epoch_entry_names"] == [f"epoch{e}" for e in EPOCHS] and
            hard["raw_entry_name_set_must_be_exact"] is True,
            "canonical epoch hardening mismatch")

    with (root / "five_checkpoint_metrics.csv").open(encoding="utf-8", newline="") as stream:
        csv_rows = list(csv.DictReader(stream))
    require(len(csv_rows) == 5 and [int(row["epoch"]) for row in csv_rows] == list(EPOCHS),
            "CSV population mismatch")
    for csv_row, json_row in zip(csv_rows, rows):
        epoch = json_row["epoch"]
        require(csv_row["checkpoint_sha256"] == json_row["checkpoint"]["sha256"] and
                csv_row["profile_sha256"] == json_row["profile"]["sha256"] and
                csv_row["AEE"] == json_row["accuracy_metrics"]["AEE"] and
                csv_row["total_spikes"] == str(json_row["activity"]["total_spikes"]),
                f"epoch{epoch} CSV/JSON mismatch")

    require(readback["attempt"] == {
        "marker_name": ".m1171_motion_final_checkpoint_selection_rebind_binder_r4_attempt_consumed",
        "interpreter": "/opt/conda/envs/sdformerflow/bin/python",
        "python_version": "3.10.20", "automatic_retry": False,
        "command_sha256": "3d69fe85b25983b9510cd0bb8eb15abd5f5a7b636ec6810b239a1e3502b6bb22",
    }, "attempt marker readback mismatch")
    expected_sources = {
        "build_m1163_motion_final_checkpoint_selection_rebind_binder.py":
            "50d22cb0f7d656c79eeb99894cb85c975441f16fd46d7df55c37ff34976aaf32",
        "build_m1166_motion_final_checkpoint_selection_rebind_binder_r2.py":
            "2171da4909fc1844c1323ca5138ccc1232fdad61d3b00446709a144461d7472c",
        "build_m1167_motion_final_checkpoint_selection_rebind_binder_r3.py":
            "7ea88b861ad54f6029f2631766a7da21b3626054217d36c27c4509293ce35d89",
        "run_m1171_motion_final_checkpoint_binder_remote_one_shot_source.py":
            "ec3483ec484e3e61c7bb27530682b597837e375c2649403f9b27617b4b54c695",
    }
    require(readback["source_sha256"] == expected_sources, "remote source SHA readback mismatch")
    require(readback["docs359_sha256"] ==
            "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
            "remote docs/359 SHA mismatch")
    return {
        "selected_epoch": 29,
        "selected_checkpoint_sha256": selection["selected"]["checkpoint"]["sha256"],
        "selected_aee": selection["selected"]["accuracy_metrics"]["AEE"],
        "selected_aae": selection["selected"]["accuracy_metrics"]["AAE"],
        "selected_aae_benchmark": selection["selected"]["accuracy_metrics"]["AAE_Benchmark"],
        "selected_total_spikes": selection["selected"]["activity"]["total_spikes"],
        "selected_firing_rate": selection["selected"]["activity"]["global_firing_rate"],
        "selected_energy_proxy_uj": selection["selected"]["activity"]["spike_energy_proxy_uj"],
    }


def attack(root: Path, fixture: Path, readback: dict[str, Any],
           mutate: Callable[[Path, Path, dict[str, Any]], None]) -> bool:
    with tempfile.TemporaryDirectory(prefix="m1175_attack_") as name:
        work = Path(name)
        import shutil
        result_copy = work / "result"
        fixture_copy = work / "fixture"
        shutil.copytree(root, result_copy)
        shutil.copytree(fixture, fixture_copy)
        rr = copy.deepcopy(readback)
        mutate(result_copy, fixture_copy, rr)
        try:
            validate_bundle(result_copy, fixture_copy, rr)
        except (HammerError, json.JSONDecodeError, UnicodeError, KeyError, ValueError):
            return True
        return False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--fixture-dir", type=Path, required=True)
    parser.add_argument("--readback", type=Path, required=True)
    args = parser.parse_args()
    readback = strict_json(args.readback)
    baseline = validate_bundle(args.result_dir, args.fixture_dir, readback)

    def json_mutator(fn: Callable[[dict[str, Any]], None]) -> Callable[[Path, Path, dict[str, Any]], None]:
        def wrapped(root: Path, fixture: Path, rr: dict[str, Any]) -> None:
            path = root / "final_checkpoint_selection.json"
            value = strict_json(path)
            fn(value)
            path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            reseal(root)
        return wrapped

    mutations: dict[str, Callable[[Path, Path, dict[str, Any]], None]] = {
        "extra_result_member": lambda r, f, d: (r / "EXTRA").write_text("x", encoding="utf-8"),
        "terminal_token": lambda r, f, d: (r / "RUN_COMPLETE.txt").write_text("PASS\n", encoding="utf-8"),
        "payload_without_reseal": lambda r, f, d: (r / "five_checkpoint_metrics.csv").write_text("bad\n", encoding="utf-8"),
        "selected_epoch_resealed": json_mutator(lambda v: v["selected"].__setitem__("epoch", 24)),
        "authorize_hardware_resealed": json_mutator(
            lambda v: v["claim_boundary"].__setitem__("hardware_rebind_authorized", True)),
        "selected_aee_resealed": json_mutator(
            lambda v: v["selected"]["accuracy_metrics"].__setitem__("AEE", "1.0")),
        "checkpoint_sha_resealed": json_mutator(
            lambda v: v["five_checkpoint_metric_table"][4]["checkpoint"].__setitem__("sha256", "0" * 64)),
        "e0e8_policy_resealed": json_mutator(
            lambda v: v["e0_e8_invalidation_and_rebind_targets"][2].__setitem__("state_after_selection", "READY")),
        "csv_aee_resealed": lambda r, f, d: (
            (r / "five_checkpoint_metrics.csv").write_text(
                (r / "five_checkpoint_metrics.csv").read_text(encoding="utf-8").replace(
                    "1.209876834190253", "1.0"), encoding="utf-8"), reseal(r)),
        "typed_zero_bool_profile": lambda r, f, d: (
            (f / "spike_profile_epoch29.json").write_text(
                (f / "spike_profile_epoch29.json").read_text(encoding="utf-8").replace(
                    '"missing_count": 0', '"missing_count": false', 1), encoding="utf-8"),
            d["epochs"]["29"].__setitem__("profile_sha256", sha256(f / "spike_profile_epoch29.json"))),
        "ranking_reorder": lambda r, f, d: (
            (f / "profile_ranking_valid825.md").write_text(
                (f / "profile_ranking_valid825.md").read_text(encoding="utf-8").replace(
                    "| 1 | 29 |", "| 1 | 24 |", 1), encoding="utf-8"),
            d["ranking"].__setitem__("sha256", sha256(f / "profile_ranking_valid825.md"))),
        "remote_source_sha": lambda r, f, d: d["source_sha256"].__setitem__(
            "build_m1167_motion_final_checkpoint_selection_rebind_binder_r3.py", "0" * 64),
        "attempt_retry_true": lambda r, f, d: d["attempt"].__setitem__("automatic_retry", True),
        "docs359_sha": lambda r, f, d: d.__setitem__("docs359_sha256", "0" * 64),
    }
    outcomes = {name: attack(args.result_dir, args.fixture_dir, readback, mutate)
                for name, mutate in mutations.items()}
    require(all(outcomes.values()), "one or more mutations were not rejected")
    print(json.dumps({
        "schema": "m1175_m1171_result_hammer_output_r1_v1",
        "status": "PASS_M1175_M1171_INDEPENDENT_RESULT_HAMMER",
        "baseline": baseline,
        "attacks": {"count": len(outcomes), "all_rejected": True, "outcomes": outcomes},
        "authorization": {
            "E0_selection_identity": "ADMITTED_AFTER_THIS_HAMMER",
            "E1_standard_valid825": "BOUND_FROM_M1171",
            "E1_deployment_numerics": "RECAPTURE_REQUIRED",
            "E2_to_E8": "RECAPTURE_OR_REPLAY_REQUIRED_AS_DECLARED",
            "hardware_speedup": False,
            "system_speedup": False,
            "power_or_energy": False,
        },
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
