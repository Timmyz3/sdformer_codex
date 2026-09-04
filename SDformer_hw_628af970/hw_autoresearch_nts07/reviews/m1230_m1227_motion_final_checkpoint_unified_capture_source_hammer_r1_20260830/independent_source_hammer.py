#!/usr/bin/env python3
"""Independent, local-only M1230 hammer of the immutable M1227 source."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest import mock


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ROOT = HW.parent
SOURCE = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1227_motion_final_checkpoint_unified_hardware_r1.py"
)
TEST = HW / "tests/test_m1227_motion_final_checkpoint_unified_capture_source.py"
CONTRACT = HW / "contracts/m1227_motion_final_checkpoint_unified_capture_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1227_motion_final_checkpoint_unified_capture_source_author_r1_20260830"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1228_BINDER = HW / "scripts/build_m1228_motion_cross_run_final_checkpoint_rebind_binder_source.py"
M1228_CONTRACT = HW / "contracts/m1228_motion_cross_run_final_checkpoint_rebind_binder_source_contract_r1_20260830.json"
EXPECTED = {
    SOURCE: "11826d81c257bb0a14def4ab620be6c3971e4eea4175d6701e88de055140116b",
    TEST: "ba6a9c6eb9e8125db235d5ff1c6634167bf44e447517104a48efca5453df944e",
    CONTRACT: "d4adc675affb468e19d329b234bfe98b0e90f364c775c76e181358db5d0a19ef",
    AUTHOR / "SHA256SUMS": "81348789cd013a6a92032f7ee59c32a51794da7ab4b2e044d8bf6c4a71f41c55",
    AUTHOR / "SHA256SUMS.seal.sha256": "2b4bc6c81c9e749d5b23154b68e0b12e1ef5a40b1fc658427bd0d4bd285d5891",
    DOCS359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M1228_BINDER: "9b2b43b4d36ed64741cbb39db0d9f5d75eb7bec09b00f4e496f3d52ce3ae5efe",
    M1228_CONTRACT: "ea94c4832dfe235a0fe3ab5c6a034ac9c98dff0611f2b753ec97bcae682389df",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


for path, digest in EXPECTED.items():
    assert path.is_file(), path
    assert sha(path) == digest, (path, sha(path))


spec = importlib.util.spec_from_file_location("m1230_m1227_immutable", SOURCE)
assert spec and spec.loader
M = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = M
spec.loader.exec_module(M)


def verify_author_package() -> None:
    rows = M.verify_double_seal(
        AUTHOR,
        EXPECTED[AUTHOR / "SHA256SUMS"],
        EXPECTED[AUTHOR / "SHA256SUMS.seal.sha256"],
    )
    assert rows["review.json"] == "95130baefd87cd946dd15d7e8b66437770608b9af7d050f82ec289addcf86b6e"
    review = M.strict_json(AUTHOR / "review.json")
    assert review["source"]["sha256"] == EXPECTED[SOURCE]
    assert review["contract"]["sha256"] == EXPECTED[CONTRACT]
    assert review["test"]["sha256"] == EXPECTED[TEST]
    assert review["authorization"]["production_release"] is False


def static_inventory() -> dict[str, list[str]]:
    result = {}
    for category, count in M.EXPECTED_STATIC_COUNTS.items():
        if category == "atlif":
            result[category] = list(M.DEAD_SN_V) + [f"live.atlif.{index}" for index in range(93)]
        else:
            result[category] = [f"{category}.{index}" for index in range(count)]
    return result


def live_records(live: dict[str, list[str]], samples) -> list[dict[str, object]]:
    return [
        {"global_sample_id": sample, "category": category, "name": name}
        for sample in samples
        for category, names in live.items()
        for name in names
    ]


class ProfilerFixture:
    def __init__(self) -> None:
        self.execution_records = [{"sample_id": 0, "kind": "fixture"}]
        self.operator_records = {"op": {"name": "op", "calls": 1}}
        self.atlif_records = {"live": {"calls": 1}}


def identity(path: Path) -> dict[str, object]:
    state = path.stat()
    return {
        "absolute_path": str(path.resolve()),
        "size_bytes": state.st_size,
        "mtime_ns": state.st_mtime_ns,
        "sha256": sha(path),
    }


def sealed_selection_root(shape: str) -> tuple[tempfile.TemporaryDirectory, dict[str, object]]:
    temp = tempfile.TemporaryDirectory(prefix=".m1230_selection_attack.", dir=HW / "results")
    root = Path(temp.name)
    checkpoint = root / "checkpoint_epoch32.pth"
    config_a = root / "selected_config.yml"
    config_b = root / "top_level_config.yml"
    checkpoint.write_bytes(b"checkpoint-m1230")
    config_a.write_text("identity: selected\n", encoding="utf-8")
    config_b.write_text("identity: conflicting-top-level\n", encoding="utf-8")
    value = {
        "schema": "m1228_motion_cross_run_final_checkpoint_rebind_binder_source_r1_v1",
        "status": "READY_CROSS_RUN_SELECTION__INDEPENDENT_RESULT_HAMMER_REQUIRED__HARDWARE_REBIND_NOT_AUTHORIZED",
        "selected": {
            "epoch": 32,
            "checkpoint": identity(checkpoint),
            "configuration": identity(config_a),
        },
    }
    if shape == "mixed":
        value["configuration"] = identity(config_b)
    selection = root / "final_checkpoint_selection.json"
    selection.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    M.write_double_seal(root)
    entry = {
        "result_path": str(root.relative_to(ROOT)),
        "manifest_sha256": sha(root / "SHA256SUMS"),
        "outer_file_sha256": sha(root / "SHA256SUMS.seal.sha256"),
        "selection_member": selection.name,
        "selection_sha256": sha(selection),
        "selection_schema": value["schema"],
    }
    return temp, entry


verify_author_package()
m1224 = M.validate_m1224()
assert m1224["root_cause"]["arithmetic"]["static_inventory_modules"] == 259
assert m1224["root_cause"]["arithmetic"]["static_atlif_modules"] == 105
assert m1224["root_cause"]["arithmetic"]["runtime_live_unified_hook_modules_per_sample"] == 247
assert m1224["root_cause"]["arithmetic"]["runtime_live_atlif_modules"] == 93
assert m1224["root_cause"]["arithmetic"]["runtime_dead_atlif_modules"] == 12

lazy = subprocess.check_output([
    sys.executable, "-c",
    "import importlib.util,sys;"
    f"p={str(SOURCE)!r};"
    "s=importlib.util.spec_from_file_location('m1230_lazy',p);"
    "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);"
    "print(int('torch' in sys.modules),int('numpy' in sys.modules),"
    "int('m1227_sealed_m1174' in sys.modules))",
], cwd=ROOT, text=True).strip()
assert lazy == "0 0 0"
namespaces = (M.CANONICAL_RESULT, M.CANONICAL_ATTEMPT, M.CANONICAL_LOG)
assert len(set(namespaces)) == 3
assert all(not os.path.lexists(str(path)) for path in namespaces)

author_tests = subprocess.run(
    [sys.executable, "-m", "unittest", "-q",
     "hw_autoresearch_nts07.tests.test_m1227_motion_final_checkpoint_unified_capture_source"],
    cwd=ROOT,
    text=True,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    check=False,
)
assert author_tests.returncode == 0, author_tests.stdout

static = static_inventory()
live = M.expected_live_inventory(static)
assert sum(len(names) for names in static.values()) == 259
assert len(static["atlif"]) == 105
assert sum(len(names) for names in live.values()) == 247
assert len(live["atlif"]) == 93
assert len(M.DEAD_SN_V) == 12 and not (set(M.DEAD_SN_V) & set(live["atlif"]))

exact = live_records(live, range(40))
audit = M.audit_call_matrix(exact, live, range(40))
assert audit["status"] == "PASS" and audit["records"] == 9880
mutations = {
    "missing_live": exact[:-1],
    "duplicate_live": exact + exact[:1],
    "dead_sn_v_fires": exact + [{
        "global_sample_id": 0, "category": "atlif", "name": M.DEAD_SN_V[0]
    }],
    "unexpected_sample": [dict(exact[0], global_sample_id=40)] + exact[1:],
    "wrong_category": [dict(exact[0], category="atlif")] + exact[1:],
}
for name, rows in mutations.items():
    failed = M.audit_call_matrix(rows, live, range(40))
    assert failed["status"] == "FAIL", name

attention = [
    {"sample_id": sample, "name": name}
    for sample in range(40) for name in M.ATTENTION_ALIASES
]
assert M.audit_attention_population(attention)["records"] == 480
for rows in (attention[:-1], attention + attention[:1]):
    try:
        M.audit_attention_population(rows)
    except M.M1227Error:
        pass
    else:
        raise AssertionError("attention mutation accepted")

with tempfile.TemporaryDirectory(prefix="m1230_payload_") as name:
    staging = Path(name)
    payloads = staging / "payloads"
    payloads.mkdir()
    hashes = [hashlib.sha256(item.encode()).hexdigest()[:12]
              for item in M.C1_TARGETS + M.DECODER_TARGETS]
    for sample in range(40):
        for order, name_hash in enumerate(hashes):
            for suffix in ("fp32.zlib", "support_sign.le.bitpack"):
                (payloads / f"s{sample:02d}_o{order:05d}_{name_hash}.{suffix}").write_bytes(b"x")
    assert len(M.validate_payload_population(staging)) == 640
    (payloads / "extra.bin").write_bytes(b"attack")
    try:
        M.validate_payload_population(staging)
    except M.M1227Error:
        pass
    else:
        raise AssertionError("extra payload accepted")

one_sample = live_records(live, [0])
one_audit = M.audit_call_matrix(one_sample, live, [0])
with tempfile.TemporaryDirectory(prefix="m1230_snapshot_") as name:
    staging = Path(name)
    final = M.atomic_sample_snapshot(staging, 0, one_sample, ProfilerFixture(), one_audit)
    snapshot = M.strict_json(final / "snapshot_manifest.json")
    assert snapshot["status"] == "SAMPLE_COMPLETE__FORENSIC_ONLY__NOT_CANONICAL"
    assert snapshot["claim_boundary"] == {
        "forensic_only": True, "canonical": False, "paper_result": False
    }
    assert not (final / "RUN_COMPLETE.txt").exists()
    assert not (staging / "manifest.json").exists()

with tempfile.TemporaryDirectory(prefix="m1230_snapshot_interrupt_") as name:
    staging = Path(name)
    with mock.patch.object(M.os, "replace", side_effect=OSError("injected-before-publish")):
        try:
            M.atomic_sample_snapshot(staging, 0, one_sample, ProfilerFixture(), one_audit)
        except OSError as error:
            assert str(error) == "injected-before-publish"
        else:
            raise AssertionError("atomic rename interruption did not fail")
    forensic_root = staging / "forensic_samples"
    assert not (forensic_root / "sample_00").exists()
    assert list(forensic_root.iterdir()) == []

# Independent P0-1: the exact M1228 result shape stores configuration below
# selected.  M1227 instead indexes selection["configuration"], so it cannot
# consume the final binder it is meant to follow.
temp, entry = sealed_selection_root("m1228")
try:
    try:
        M.validate_final_selection(entry)
    except Exception as error:  # exact failure type is evidence
        m1228_shape_rejected = type(error).__name__
    else:
        raise AssertionError("M1228 shape unexpectedly accepted")
finally:
    temp.cleanup()
assert m1228_shape_rejected == "KeyError"

# Independent P0-2: adding a conflicting top-level configuration makes the
# same sealed result pass.  The returned capture binding silently pairs the
# selected checkpoint with that top-level config and ignores
# selected.configuration.  Therefore checkpoint/config pair integrity is not
# proven even though every file and the result are double sealed.
temp, entry = sealed_selection_root("mixed")
try:
    mixed = M.validate_final_selection(entry)
    selection = mixed["selection"]
    selected_config_sha = selection["selected"]["configuration"]["sha256"]
    top_config_sha = selection["configuration"]["sha256"]
    mixed_identity_accepted = (
        selected_config_sha != top_config_sha and
        mixed["identity"]["config_sha256"] == top_config_sha
    )
finally:
    temp.cleanup()
assert mixed_identity_accepted is True

print(json.dumps({
    "schema": "m1230_m1227_independent_hammer_execution_r1_v1",
    "status": "FAIL_P0_FINAL_SELECTION_PAIR_BINDING",
    "author_tests": 15,
    "author_tests_pass": True,
    "m1224_bound": True,
    "static_modules": 259,
    "static_atlif": 105,
    "live_modules_per_sample": 247,
    "live_atlif_per_sample": 93,
    "dead_sn_v": 12,
    "ordered_records": 9880,
    "attention_records": 480,
    "payload_files": 640,
    "atomic_snapshot_forensic_not_canonical": True,
    "atomic_snapshot_pre_publish_failure_leaves_no_partial": True,
    "lazy_import_heavy_modules": False,
    "fresh_namespaces_absent_and_distinct": True,
    "independent_mutations_rejected": sorted(mutations),
    "p0": {
        "m1228_shape_rejected_with": m1228_shape_rejected,
        "mixed_checkpoint_config_pair_accepted": mixed_identity_accepted,
    },
    "release_authoring": False,
    "source_modified": False,
    "gpu": False,
    "remote": False,
}, sort_keys=True))
