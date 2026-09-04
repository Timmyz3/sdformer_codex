#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author, source-only blind hammer for final sealed M1323."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
OUT = Path(__file__).resolve().parent
SOURCE = HW / "system_simulator/scripts/build_m1323_ep34_decoder_capture_adapter_source.py"
TEST = HW / "system_simulator/tests/test_m1323_ep34_decoder_capture_adapter_source.py"
CONTRACT = HW / "contracts/m1323_ep34_decoder_capture_adapter_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1323_ep34_decoder_capture_adapter_source_author_r1_20260831"
M1322 = HW / "reviews/m1322_m1321_ep34_decoder_adapter_source_hammer_r1_20260831"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

EXPECTED = {
    "source": "0481e39372ffe19cd3cff8d5053c9eae8326de4fb5ac61bd9e42527a3ad3a12a",
    "test": "c29980f357ea0e0a9b2e11650239b706f6c4e18892b4975925db164a72439487",
    "contract": "e4df50fed6068b0f384693044705b30f595d41d70dce78e738cb36a98e24cecc",
    "author_review": "022fc0ddc5e6de5907f4033a08d968e76db5a903c544687dca52538059f6c1d9",
    "author_manifest": "83cd60889bff6f8211ddd3819233f5eb267c7fb25d81d0af8a36767f60215702",
    "author_outer": "e010a86648b93aecb4614d1a12f67be9d9cb4d47961b941e64840481d5f2c28b",
    "m1322_review": "c8fa3f9a80812af3f3cdd4cb439dd5ad110538ff8a86e746e1a5420a106bb717",
    "m1322_manifest": "ee45fa6d7ddc75316d9212f4ef3972277524a89d371d1f0efbd937b6cda8319c",
    "m1322_outer": "d07a2391c667a2b91b2f0f90c4451e0203d2ff8d890fd1e9713e68b2f8b46048",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


class HammerError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise HammerError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_manifest(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("  ", 1)
        require(len(parts) == 2 and len(parts[0]) == 64, "malformed manifest row")
        relative = PurePosixPath(parts[1])
        require(relative.parts and not relative.is_absolute() and ".." not in relative.parts,
                "unsafe manifest path")
        require(parts[1] not in rows, "duplicate manifest path")
        rows[parts[1]] = parts[0]
    return rows


def verify_seal(root: Path, manifest_sha: str, outer_sha: str) -> dict[str, str]:
    require(root.is_dir() and not root.is_symlink(), "sealed root missing/symlink")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    require(sha256(manifest) == manifest_sha and sha256(outer) == outer_sha,
            "recursive seal SHA drift")
    require(outer.read_text(encoding="utf-8") == manifest_sha + "  SHA256SUMS\n",
            "outer seal content drift")
    rows = parse_manifest(manifest)
    actual = sorted(path.relative_to(root).as_posix() for path in root.rglob("*")
                    if path.is_file() and path.name not in {
                        "SHA256SUMS", "SHA256SUMS.seal.sha256"})
    require(sorted(rows) == actual, "sealed member population drift")
    for relative, digest in rows.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink() and sha256(path) == digest,
                "sealed member drift: " + relative)
    return rows


def load_source():
    spec = importlib.util.spec_from_file_location("m1324_sealed_m1323", SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load M1323")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def expect_reject(module, function, label: str) -> str:
    try:
        function()
    except module.M1323Error as error:
        return str(error)
    except Exception as error:
        raise HammerError(label + " raised wrong exception " + type(error).__name__) from error
    raise HammerError(label + " was accepted")


def input_stats(module, shape: list[int]) -> dict[str, Any]:
    elements = module.math.prod(shape)
    return {
        "shape": shape, "stride": [1] * len(shape), "dtype": "torch.float32",
        "elements": elements, "bytes": elements * 4, "active": 0,
        "positive": 0, "negative": 0, "nonfinite": 0,
    }


def retained_payload(module, sample: int, order: int, name: str) -> dict[str, Any]:
    stem = "s{:02d}_o{:05d}_{}".format(
        sample, order, hashlib.sha256(name.encode()).hexdigest()[:12])
    return {
        "retained": True, "raw_fp32_sha256": "1" * 64,
        "compressed_fp32": "payloads/{}.fp32.zlib".format(stem),
        "compressed_sha256": "2" * 64,
        "support_sign": "payloads/{}.support_sign.le.bitpack".format(stem),
        "support_sign_sha256": "3" * 64,
        "positive_plane_bytes": 1, "negative_plane_bytes": 1,
    }


def build_rows(module):
    inventory = module.frozen_inventory_names()
    cohort = module.expected_cohort()
    live = [(category, name) for category, names in inventory.items() for name in names]
    require(len(live) == 247 and len(set(live)) == 247, "independent inventory malformed")
    rows = []
    for sample in range(40):
        for category, name in live:
            order = len(rows)
            shape = ([1] if category != "decoder_convtranspose" else
                     list(module.SHAPES[module.MODULES.index(name)]))
            identity = cohort[sample]
            retained = category in {"c1_conv3x3", "decoder_convtranspose"}
            rows.append({
                "global_order": order, "global_sample_id": sample,
                "cohort": identity["cohort"], "sequence": identity["sequence"],
                "sample_key": identity["sample_key"],
                "source_sha256": identity["source_sha256"],
                "category": category, "name": name, "input": input_stats(module, shape),
                "payload": (retained_payload(module, sample, order, name) if retained
                            else dict(module.NONRETAINED_PAYLOAD)),
            })
    require(len(rows) == 9880, "independent rows not 9880")
    return rows, inventory, cohort


def weights(module):
    checkpoint = "a" * 64
    rows = []
    for ordinal, shape in enumerate(module.WEIGHT_SHAPES):
        rows.append({
            "module_ordinal": ordinal, "module": module.MODULES[ordinal],
            "checkpoint_sha256": checkpoint,
            "weight": {
                "shape": list(shape), "dtype": "torch.float32",
                "layout": "C_ORDER_CONTIGUOUS", "byte_order": "little",
                "content_bytes": module.math.prod(shape) * 4,
                "content_sha256": ("%x" % (ordinal + 1)) * 64,
            }, "bias": None,
        })
    return rows, checkpoint


def main() -> int:
    for label, path in (("source", SOURCE), ("test", TEST), ("contract", CONTRACT),
                        ("docs359", DOCS359)):
        require(path.is_file() and not path.is_symlink(), label + " missing/symlink")
        require(sha256(path) == EXPECTED[label], label + " SHA drift")
    author_rows = verify_seal(AUTHOR, EXPECTED["author_manifest"], EXPECTED["author_outer"])
    require(author_rows.get("review.json") == EXPECTED["author_review"],
            "author review member mismatch")
    failed_rows = verify_seal(M1322, EXPECTED["m1322_manifest"], EXPECTED["m1322_outer"])
    require(failed_rows.get("review.json") == EXPECTED["m1322_review"],
            "M1322 failure review member mismatch")

    baseline = subprocess.run([str(PYTHON), "-I", str(TEST)], cwd=ROOT, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    require(baseline.returncode == 0 and "Ran 9 tests" in baseline.stdout and
            baseline.stdout.rstrip().endswith("OK"), "author tests not 9/9 PASS")
    module = load_source()
    rows, inventory, cohort = build_rows(module)

    selected, identity = module.decoder_rows_from_ordered(rows, inventory, cohort)
    require(len(selected) == 120 and selected[0]["global_sample_id"] == 10 and
            selected[-1]["global_sample_id"] == 39 and
            identity["ordered_rows"] == 9880 and
            identity["unique_retained_payload_pairs"] == 320 and
            identity["all_sample_sequences_equal"] is True,
            "positive full-stream projection failed")
    checks = ["positive_9880_40x247_120_calls_320_payload_pairs"]

    attack = copy.deepcopy(rows)
    attack[10 * 247 + 1]["global_order"] = attack[10 * 247]["global_order"]
    expect_reject(module, lambda: module.decoder_rows_from_ordered(attack, inventory, cohort),
                  "duplicate selected global_order")
    checks.append("duplicate_selected_global_order_rejected")

    attack = copy.deepcopy(rows)
    attack[301] = copy.deepcopy(attack[300]); attack[301]["global_order"] = 301
    expect_reject(module, lambda: module.decoder_rows_from_ordered(attack, inventory, cohort),
                  "duplicate ignored row")
    checks.append("duplicate_ignored_row_rejected")

    attack = copy.deepcopy(rows); attack[500]["global_order"] = 499
    expect_reject(module, lambda: module.decoder_rows_from_ordered(attack, inventory, cohort),
                  "noncontiguous global order")
    attack = copy.deepcopy(rows); attack[500]["global_order"] = True
    expect_reject(module, lambda: module.decoder_rows_from_ordered(attack, inventory, cohort),
                  "boolean global order")
    attack = copy.deepcopy(rows); attack[247]["global_sample_id"] = True
    expect_reject(module, lambda: module.decoder_rows_from_ordered(attack, inventory, cohort),
                  "boolean sample id")
    checks.extend(["noncontiguous_order_rejected", "bool_order_rejected",
                   "bool_sample_rejected"])

    expect_reject(module, lambda: module.decoder_rows_from_ordered(rows[:-1], inventory, cohort),
                  "missing final row")
    over = rows + [copy.deepcopy(rows[-1])]; over[-1]["global_order"] = 9880
    expect_reject(module, lambda: module.decoder_rows_from_ordered(over, inventory, cohort),
                  "extra row")
    checks.extend(["9879_row_stream_rejected", "9881_row_stream_rejected"])

    attack = copy.deepcopy(rows)
    retained = [index for index, row in enumerate(attack) if row["payload"]["retained"] is True]
    source_index, victim_index = retained[0], retained[8]
    attack[victim_index]["payload"]["compressed_fp32"] = \
        attack[source_index]["payload"]["compressed_fp32"]
    attack[victim_index]["payload"]["support_sign"] = \
        attack[source_index]["payload"]["support_sign"]
    expect_reject(module, lambda: module.decoder_rows_from_ordered(attack, inventory, cohort),
                  "cross-call retained payload alias")
    checks.append("cross_call_payload_alias_rejected")

    attack = copy.deepcopy(rows)
    first_retained = retained[0]
    attack[first_retained]["payload"]["compressed_fp32"] = \
        attack[first_retained]["payload"]["compressed_fp32"].replace("s00_", "s01_", 1)
    expect_reject(module, lambda: module.decoder_rows_from_ordered(attack, inventory, cohort),
                  "wrong sample in payload stem")
    checks.append("payload_stem_sample_binding_rejected")

    attack = copy.deepcopy(rows)
    a, b = 255, 256
    first, second = copy.deepcopy(attack[a]), copy.deepcopy(attack[b])
    first["global_order"], second["global_order"] = b, a
    attack[a], attack[b] = second, first
    expect_reject(module, lambda: module.decoder_rows_from_ordered(attack, inventory, cohort),
                  "same-population module sequence permutation")
    checks.append("module_sequence_permutation_rejected")

    weight_rows, checkpoint = weights(module)
    require(len(module.validate_weight_identities(weight_rows, checkpoint)) == 4,
            "weight positive control failed")
    attack_weights = copy.deepcopy(weight_rows); attack_weights[1]["module_ordinal"] = True
    expect_reject(module, lambda: module.validate_weight_identities(
        attack_weights, checkpoint), "boolean weight ordinal")
    expect_reject(module, lambda: module.audit_two_plane_payload(
        Path("absent"), Path("absent"), (1,), True), "boolean payload ordinal")
    checks.extend(["weight_identity_positive", "bool_weight_ordinal_rejected",
                   "bool_payload_ordinal_rejected"])

    expect_reject(module, lambda: module.main([]), "default CLI")
    cli = subprocess.run([str(PYTHON), "-I", str(SOURCE), "--production-replay"],
                         cwd=ROOT, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, check=False)
    require(cli.returncode != 0 and "unrecognized arguments" in cli.stdout,
            "production CLI unexpectedly present")
    checks.extend(["default_cli_inert", "production_cli_absent"])

    source_text = SOURCE.read_text(encoding="utf-8")
    for key in ("capture_result_hammered", "normalized_payload_written",
                "production_replay", "cycles", "traffic", "speedup",
                "system_speedup", "energy", "ppa", "table_a"):
        require(('"%s": False' % key) in source_text, "claim promoted: " + key)
    checks.append("claim_boundary_fail_closed")

    output = {
        "schema": "m1324_m1323_ep34_decoder_adapter_source_hammer_r1_v1",
        "status": "PASS_M1324_M1323_SOURCE_HAMMER__ACTUAL_RESULT_SUCCESSOR_ALLOWED",
        "source_authority": {
            "source_path": str(SOURCE.relative_to(ROOT)), "source_sha256": sha256(SOURCE),
            "test_path": str(TEST.relative_to(ROOT)), "test_sha256": sha256(TEST),
            "contract_path": str(CONTRACT.relative_to(ROOT)), "contract_sha256": sha256(CONTRACT),
            "author_review_sha256": sha256(AUTHOR / "review.json"),
            "author_manifest_sha256": sha256(AUTHOR / "SHA256SUMS"),
            "author_outer_file_sha256": sha256(AUTHOR / "SHA256SUMS.seal.sha256"),
            "m1322_failure_review_sha256": sha256(M1322 / "review.json"),
            "docs359_sha256": sha256(DOCS359),
        },
        "independence": {"different_author": True},
        "author_tests": {"count": 9, "passed": True, "output": baseline.stdout},
        "blind_checks": checks,
        "m1322_findings": {
            "duplicate_selected_global_order_closed": True,
            "duplicate_ignored_row_closed": True,
            "boolean_weight_ordinal_closed": True,
            "full_9880_order_continuity_closed": True,
            "cross_call_payload_alias_closed": True,
        },
        "authorization": {
            "source_audit_citable": True,
            "actual_result_successor_authoring": True,
            "production_replay": False, "remote_access": False, "gpu": False,
        },
        "claim_boundary": {
            "source_only": True, "actual_capture_read": False,
            "actual_result_hammer_bound": False, "normalized_payload_written": False,
            "production_replay": False, "cycles": False, "traffic": False,
            "speedup": False, "system_speedup": False, "energy": False,
            "ppa": False, "table_a": False, "paper_citable_performance": False,
        },
    }
    (OUT / "hammer_output.json").write_text(
        json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8")
    (OUT / "author_test_output.txt").write_text(baseline.stdout, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
