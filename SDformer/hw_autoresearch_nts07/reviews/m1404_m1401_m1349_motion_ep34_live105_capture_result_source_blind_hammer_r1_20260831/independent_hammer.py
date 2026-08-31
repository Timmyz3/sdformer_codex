#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh different-author fixture hammer for M1401. No canonical access."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Callable
import zlib

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/hammer_m1401_m1349_motion_ep34_live105_capture_result_source.py"
SOURCE_SHA = "f55642429fe097fdb5c5fd860592d4b04652fc47c85526eb756dc005125e8a22"
TEST = HW / "tests/test_hammer_m1401_m1349_motion_ep34_live105_capture_result_source.py"
TEST_SHA = "b3d0b7c075a9d54e6a679642cfb7bfa29874e51d02110127730798bc6b388192"
CONTRACT = HW / "contracts/m1401_m1349_motion_ep34_live105_capture_result_hammer_source_contract_r1_20260831.json"
CONTRACT_SHA = "e62539703a45b6e16f03d0ccf92222c0aaa545cc5421dc2128a208bef2acf6d5"
AUTHOR = HW / "reviews/m1401_m1349_motion_ep34_live105_capture_result_hammer_source_author_r1_20260831"
AUTHOR_REVIEW_SHA = "6091ee792fae929d9a57fce5b42ef7fe66863d65bc9cf81f4121a14e937a90bd"
AUTHOR_MANIFEST_SHA = "4bf2b67bee3fc1b797b87e1b7c509cfcf107dbb82b9f601d3ee8c0da1dacc225"
AUTHOR_OUTER_SHA = "21638dae993135fdb3c4f278e965091197b34771fc0d60867d4feb47a9489c27"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load():
    assert digest(SOURCE) == SOURCE_SHA
    spec = importlib.util.spec_from_file_location("m1404_blind_m1401", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load()
CHECKS: list[str] = []
ATTACKS = 0
FALSE_NEGATIVES = 0


def passed(name: str) -> None:
    CHECKS.append(name)


def reject(name: str, call: Callable[[], Any]) -> None:
    global ATTACKS, FALSE_NEGATIVES
    ATTACKS += 1
    try:
        call()
    except Exception:
        passed(name)
        return
    FALSE_NEGATIVES += 1
    raise AssertionError("accepted attack: " + name)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_seal(root: Path) -> tuple[str, str]:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    for path in (manifest, outer):
        if os.path.lexists(path): path.unlink()
    members = sorted(path for path in root.rglob("*") if path.is_file() and not path.is_symlink())
    manifest.write_text("".join(
        f"{digest(path)}  {path.relative_to(root).as_posix()}\n" for path in members),
        encoding="utf-8")
    outer.write_text(f"{digest(manifest)}  SHA256SUMS\n", encoding="ascii")
    return digest(manifest), digest(outer)


def manifest_value() -> dict[str, Any]:
    return {
        "schema": "m1343_motion_ep34_live105_unified_hardware_capture_r1_v1",
        "status": "CAPTURE_COMPLETE__FRESH_M1343_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
        "identity": {
            "checkpoint_load_audit": {"missing_count": 0, "unexpected_count": 0},
            "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "selection": {"selected": {
                "candidate_id": "resume_ep34", "epoch": 34,
                "checkpoint": {"sha256": M.CHECKPOINT_SHA256},
                "configuration": {"sha256": M.CONFIG_SHA256},
                "profile": {"sha256": M.PROFILE_SHA256, "samples": 825,
                            "module_counts": {"ATLIFTernaryPSN": 105,
                                              "ShiftmaxAttention": 12}},
            }},
        },
        "m1227_runtime_contract": {"final_selection_identity": {
            "epoch": 34, "checkpoint_sha256": M.CHECKPOINT_SHA256,
            "config_sha256": M.CONFIG_SHA256, "profile_sha256": M.PROFILE_SHA256,
            "selection_sha256": M.SELECTION_SHA256,
        }},
        "m1343_runtime_contract": {
            "static_modules": 259, "static_atlif": 105,
            "live_modules_per_sample": 259, "live_atlif": 105,
            "dead_sn_v": [], "dead_calls_per_sample": 0,
            "atlif_names_sha256": M.ATLIF_NAMES_SHA256,
            "ordered_records": 10360, "attention_records": 480,
            "payload_files": 640,
        },
        "cohort": {"samples": M.BASE.OLD.expected_cohort()},
        "claim_boundary": {
            "capture_only": True, "accuracy": False, "cycles": False,
            "speedup": False, "system_speedup": False, "energy": False,
            "rtl": False, "ppa": False, "fresh_result_hammer_required": True,
        },
    }


def admission_value() -> dict[str, Any]:
    return {
        "schema": "m1343_final_capture_admission_r1_v1", "status": "PASS",
        "ordered": 10360, "attention": 480, "payload_files": 640,
        "execution": 7360, "operator_rows": 79, "atlif_live_rows": 105,
        "atlif_static": 105, "dead_sn_v": [],
        "atlif_names_sha256": M.ATLIF_NAMES_SHA256,
        "claim_boundary": {"capture_only": True, "paper_result": False,
                           "cycles": False, "speedup": False, "energy": False,
                           "ppa": False},
    }


def names_by_category() -> dict[str, list[str]]:
    output = {}
    for category, count in M.EXPECTED_COUNTS.items():
        if category == "c1_conv3x3": values = list(M.M1349.R1.C1_TARGETS)
        elif category == "decoder_convtranspose": values = list(M.M1349.R1.DECODER_TARGETS)
        elif category == "atlif": values = list(M.M1349.EXPECTED_ATLIF_NAMES)
        else: values = [f"{category}.unit_{index:03d}" for index in range(count)]
        assert len(values) == count
        output[category] = values
    return output


def retained_record(root: Path, sample: int, ordinal: int, name: str) -> tuple[dict, dict]:
    value = np.array([1.0 if ordinal & 1 else -1.0], dtype="<f4")
    raw = value.tobytes()
    stem = f"s{sample:02d}_o{ordinal:05d}_{hashlib.sha256(name.encode()).hexdigest()[:12]}"
    compressed_rel = f"payloads/{stem}.fp32.zlib"
    support_rel = f"payloads/{stem}.support_sign.le.bitpack"
    (root / compressed_rel).write_bytes(zlib.compress(raw))
    support = bytes([1 if value[0] > 0 else 0, 1 if value[0] < 0 else 0])
    (root / support_rel).write_bytes(support)
    return ({"dtype": "torch.float32", "elements": 1, "bytes": 4,
             "active": 1, "positive": int(value[0] > 0),
             "negative": int(value[0] < 0), "nonfinite": 0},
            {"retained": True, "raw_fp32_sha256": hashlib.sha256(raw).hexdigest(),
             "compressed_fp32": compressed_rel,
             "compressed_sha256": digest(root / compressed_rel),
             "support_sign": support_rel,
             "support_sign_sha256": digest(root / support_rel),
             "positive_plane_bytes": 1, "negative_plane_bytes": 1})


def build_ordered(root: Path) -> list[dict[str, Any]]:
    names = names_by_category()
    ordered = []
    for sample in range(40):
        ordinal = 0
        for category in M.EXPECTED_COUNTS:
            for name in names[category]:
                meta: dict[str, Any] = {}; payload: dict[str, Any] = {}
                if category in {"c1_conv3x3", "decoder_convtranspose"}:
                    meta, payload = retained_record(root, sample, ordinal, name)
                    ordinal += 1
                ordered.append({"sample_id": sample, "category": category,
                                "name": name, "input": meta, "payload": payload})
    assert len(ordered) == 10360
    return ordered


def build_attention(root: Path) -> None:
    records = []
    for sample in range(40):
        for name in M.M1349.R1.ATTENTION_ALIASES:
            safe = name.replace(".", "_").replace("/", "_")
            relative = f"attention_qk/sample{sample}_{safe}.npz"
            np.savez_compressed(root / relative,
                q_shape=np.array([2, 1, 1, 1, 1], dtype=np.int32),
                k_shape=np.array([2, 1, 1, 1, 1], dtype=np.int32),
                q_bits_packed=np.array([1], dtype=np.uint8),
                k_bits_packed=np.array([2], dtype=np.uint8),
                gate_q17=np.array([[[0, 256]]], dtype=np.uint16))
            records.append({"sample_id": sample, "name": name,
                "file": Path(relative).name, "sha256": digest(root / relative),
                "windows_captured": 1, "heads": 1, "spatial_tokens": 1,
                "temporal_tokens": 2, "lanes": 1, "q_active_bits": 1,
                "k_active_bits": 1, "gate_nonzero": 1,
                "gate_min": 0, "gate_max": 256})
    write_json(root / "attention_qk/manifest.json", {"records": records})


def build_snapshots(root: Path) -> None:
    names = ["unified_ordered_sample.jsonl", "execution_sample.json",
             "operator_runtime_cumulative.json", "atlif_activity_cumulative.json"]
    for sample in range(40):
        directory = root / f"forensic_samples/sample_{sample:02d}"
        directory.mkdir(parents=True)
        files = {}
        for name in names:
            (directory / name).write_text("{}\n", encoding="utf-8")
            files[name] = digest(directory / name)
        write_json(directory / "snapshot_manifest.json", {
            "sample_id": sample,
            "call_audit": {"status": "PASS", "records": 259,
                           "live_modules_per_sample": 259},
            "files": files,
        })


def build_full_fixture(parent: Path) -> Path:
    root = parent / "result"
    (root / "payloads").mkdir(parents=True)
    (root / "attention_qk").mkdir()
    ordered = build_ordered(root)
    (root / "unified_ordered_records.jsonl").write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in ordered), encoding="utf-8")
    build_attention(root)
    build_snapshots(root)
    write_json(root / "manifest.json", manifest_value())
    write_json(root / "m1343_admission.json", admission_value())
    write_json(root / "execution_trace.json", [{} for _ in range(7360)])
    write_json(root / "operator_runtime.json",
               [{"name": f"operator.{i:03d}", "calls": 40} for i in range(79)])
    write_json(root / "atlif_activity.json",
               [{"name": name, "calls": 40} for name in M.M1349.EXPECTED_ATLIF_NAMES])
    (root / "RUN_COMPLETE.txt").write_text(
        "PASS_M1174_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM\n")
    write_seal(root)
    return root


def raw_fixture(parent: Path) -> tuple[Path, dict, dict[str, str]]:
    root = parent / "raw"; (root / "payloads").mkdir(parents=True)
    meta, payload = retained_record(root, 0, 0, M.M1349.R1.C1_TARGETS[0])
    row = {"input": meta, "payload": payload}
    rows = {payload["compressed_fp32"]: payload["compressed_sha256"],
            payload["support_sign"]: payload["support_sign_sha256"]}
    return root, row, rows


def attention_fixture(path: Path, extra=False, bad_tail=False) -> tuple[Path, dict]:
    values = {
        "q_shape": np.array([2, 1, 1, 1, 7], dtype=np.int32),
        "k_shape": np.array([2, 1, 1, 1, 7], dtype=np.int32),
        "q_bits_packed": np.array([0x55, 0x15 | (0xC0 if bad_tail else 0)], dtype=np.uint8),
        "k_bits_packed": np.array([0x2A, 0x2A], dtype=np.uint8),
        "gate_q17": np.array([[[0, 256]]], dtype=np.uint16),
    }
    if extra: values["invented"] = np.array([1], dtype=np.uint8)
    np.savez_compressed(path, **values)
    return path, {"windows_captured": 1, "heads": 1, "spatial_tokens": 1,
                  "temporal_tokens": 2, "lanes": 7}


def main() -> int:
    assert digest(TEST) == TEST_SHA and digest(CONTRACT) == CONTRACT_SHA
    assert not os.path.lexists(M.CANONICAL_RESULT)
    test_run = subprocess.run([str(PYTHON), str(TEST)], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
                              env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert test_run.returncode == 0 and "Ran 12 tests" in test_run.stdout and "OK" in test_run.stdout
    passed("author_12_tests_replayed")
    self_run = subprocess.run([str(PYTHON), str(SOURCE), "--source-self-check"], cwd=ROOT,
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              check=False, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    assert self_run.returncode == 0 and self_run.stdout.count(M.PASS_TOKEN) == 1
    passed("source_self_check_replayed")
    M.validate_source_policy(); passed("exact_contract_positive")
    author_rows, author_seal = M.BASE.verify_recursive_seal(AUTHOR)
    assert author_rows["review.json"] == AUTHOR_REVIEW_SHA
    assert author_seal == {"manifest_sha256": AUTHOR_MANIFEST_SHA,
                           "outer_file_sha256": AUTHOR_OUTER_SHA}
    passed("author_recursive_seal_exact")

    with tempfile.TemporaryDirectory(prefix="m1404_blind_") as raw:
        parent = Path(raw)
        root = build_full_fixture(parent)
        result = M.validate_result(root)
        assert result["population"] == {"ordered": 10360, "retained": 320,
            "attention": 480, "payload": 640, "execution": 7360,
            "operator": 79, "atlif": 105}
        passed("full_fixture_exact_result_positive")

        rows = [M.BASE.strict_text(line) for line in
                (root / "unified_ordered_records.jsonl").read_text().splitlines()]
        assert M.validate_ordered(rows)["ordered_rows"] == 10360
        passed("ordered_40x259_and_live105_positive")
        mutant = copy.deepcopy(rows); mutant[259], mutant[260] = mutant[260], mutant[259]
        reject("ordered_cross_sample_sequence_attack", lambda: M.validate_ordered(mutant))
        mutant = copy.deepcopy(rows); mutant[4]["name"] += ".rename"
        reject("ordered_live105_identity_attack", lambda: M.validate_ordered(mutant))

        M.validate_admission(admission_value()); passed("admission_exact_positive")
        admission = admission_value(); admission["extra"] = True
        reject("admission_extra_key_attack", lambda: M.validate_admission(admission))
        manifest = manifest_value(); M.validate_manifest(manifest); passed("manifest_identity_positive")
        manifest["identity"]["selection"]["selected"]["checkpoint"]["sha256"] = "0" * 64
        reject("manifest_checkpoint_attack", lambda: M.validate_manifest(manifest))

        seal_rows, _seal = M.BASE.verify_recursive_seal(root)
        assert len(seal_rows) > 1200; passed("recursive_seal_positive")
        (root / "unsealed.injection").write_text("x")
        reject("recursive_unsealed_member_attack", lambda: M.BASE.verify_recursive_seal(root))
        (root / "unsealed.injection").unlink()
        os.symlink("manifest.json", root / "symlink.injection")
        reject("recursive_symlink_attack", lambda: M.BASE.verify_recursive_seal(root))
        (root / "symlink.injection").unlink()

        raw_root, row, raw_rows = raw_fixture(parent)
        M.M1338.validate_one_retained_payload(raw_root, raw_rows, row)
        passed("retained_zlib_support_positive")
        compressed = raw_root / row["payload"]["compressed_fp32"]
        compressed.write_bytes(compressed.read_bytes() + b"TRAIL")
        row["payload"]["compressed_sha256"] = digest(compressed)
        raw_rows[row["payload"]["compressed_fp32"]] = digest(compressed)
        reject("retained_zlib_trailing_attack",
               lambda: M.M1338.validate_one_retained_payload(raw_root, raw_rows, row))
        raw_root2, row2, raw_rows2 = raw_fixture(parent / "second")
        support = raw_root2 / row2["payload"]["support_sign"]
        support.write_bytes(b"\x01\x01")
        row2["payload"]["support_sign_sha256"] = digest(support)
        raw_rows2[row2["payload"]["support_sign"]] = digest(support)
        reject("retained_support_overlap_attack",
               lambda: M.M1338.validate_one_retained_payload(raw_root2, raw_rows2, row2))

        attn, attn_row = attention_fixture(parent / "attention_ok.npz")
        M.M1338.validate_attention_npz(attn, attn_row); passed("attention_exact_npz_positive")
        bad, bad_row = attention_fixture(parent / "attention_extra.npz", extra=True)
        reject("attention_extra_member_attack", lambda: M.M1338.validate_attention_npz(bad, bad_row))
        bad, bad_row = attention_fixture(parent / "attention_tail.npz", bad_tail=True)
        reject("attention_nonzero_tail_attack", lambda: M.M1338.validate_attention_npz(bad, bad_row))

        payload_files = M.M1349.R1.validate_payload_population(root)
        assert len(payload_files) == 640; passed("payload_640_exact_positive")
        victim = payload_files[0]; held = victim.read_bytes(); victim.unlink()
        reject("payload_missing_member_attack", lambda: M.M1349.R1.validate_payload_population(root))
        victim.write_bytes(held)
        (root / "payloads/invented").write_bytes(b"")
        reject("payload_extra_member_attack", lambda: M.M1349.R1.validate_payload_population(root))
        (root / "payloads/invented").unlink()

        M.M1349.validate_snapshot_population_live105(root)
        passed("forensic_40_snapshots_positive")
        snap = root / "forensic_samples/sample_00/snapshot_manifest.json"
        value = json.loads(snap.read_text()); value["call_audit"]["records"] = 247
        write_json(snap, value)
        reject("forensic_old247_boundary_attack",
               lambda: M.M1349.validate_snapshot_population_live105(root))
        value["call_audit"]["records"] = 259; value["files"].pop("execution_sample.json")
        write_json(snap, value)
        reject("forensic_missing_snapshot_member_attack",
               lambda: M.M1349.validate_snapshot_population_live105(root))

    assert not os.path.lexists(M.CANONICAL_RESULT)
    passed("canonical_absent_before_and_after")
    assert FALSE_NEGATIVES == 0
    output = {
        "schema": "m1404_m1401_m1349_live105_result_source_blind_hammer_output_r1_v1",
        "status": "PASS",
        "checks_passed": len(CHECKS),
        "attack_count": ATTACKS,
        "false_negative_count": FALSE_NEGATIVES,
        "checks": CHECKS,
        "execution": {"capture": False, "gpu": False, "remote": False,
                      "eda": False, "canonical_result_created": False},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
