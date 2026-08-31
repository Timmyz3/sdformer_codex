#!/opt/conda/envs/sdformerflow/bin/python
"""M1177/M1174-r2 hardened one-load Motion hardware capture.

This additive source keeps the r1 capture implementation as a SHA-pinned
substrate, but replaces every release-critical boundary: M1175 admission,
M1177 source-hammer consumption, canonical GPU ownership, fixed forty-source
cohort, complete module inventory/call coverage, complete 40x12 Q/K/gate
payload coverage, and recursive result sealing.  The checked-in r2 source
contract is source-only and cannot launch production.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
R1_PATH = Path(__file__).with_name(
    "capture_m1174_motion_checkpoint_parametric_unified_hardware.py")
R1_SHA256 = "b476fad6885be23aa63a6b5d8e690fb3e213421074270cbb25e8ec00c202080a"
SOURCE_CONTRACT = HW / (
    "contracts/m1177_motion_checkpoint_parametric_unified_capture_source_contract_r2_20260830.json")
CANONICAL_LEASE = HW / "results/gpu_profile_lease.lock"
M1175 = HW / "reviews/m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_20260830"
M1175_SCHEMA = "m1175_m1171_motion_final_checkpoint_binder_result_hammer_r1_v1"
M1175_REVIEW_SHA256 = "8b83690b8b1130d2335bb118d35645ae4d172740966ab69c6fcea9bc8b5d307b"
M1175_MANIFEST_SHA256 = "2a4481491d3d12bcba17263260a87e6511e523b4b410e18f3c7fecada07ab247"
M1175_OUTER_FILE_SHA256 = "17a306168fc3c39b86e869f2213a6592c677203bed52913bf4e6fff29390199e"
M1171_REMOTE_MANIFEST_SHA256 = "e09939b65a171dfbd7b990a26a630c919b0e8dd766759de20539743b02d78beb"
M1171_REMOTE_OUTER_FILE_SHA256 = "cdce21739ca41ceb6896747d7bb0ed06aff516341392a0f790159cfec3c03447"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_SELECTION = {
    "epoch": 29,
    "checkpoint_sha256": "2144dfd628cd928bfb768b92d4fa097b720db112c32d930b9f3cd85c6217286a",
    "checkpoint_size_bytes": 225504447,
    "checkpoint_mtime_ns": 1788057827000000000,
    "configuration_sha256": "c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955",
}
C1_TARGETS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
DECODER_TARGETS = tuple(
    "sttmultires_unet.decoders.{}.deconv.0".format(index) for index in range(4))
ATTENTION_ALIASES = tuple(
    "S{}.B{}.attn".format(stage, block)
    for stage, blocks in enumerate((2, 2, 6, 2)) for block in range(blocks))
CATEGORIES = frozenset({
    "c1_conv3x3", "decoder_convtranspose", "atlif", "fc1", "fc2",
    "patch_embed", "batch_norm", "qkv", "attention",
})


class R2Error(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise R2Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise R2Error("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "{} must be a non-symlink regular file: {}".format(label, path))


def directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise R2Error("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISDIR(mode) and not path.is_symlink(),
            "{} must be a non-symlink directory: {}".format(label, path))


def strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise R2Error("non-standard JSON token: " + token)
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def safe_member(root: Path, name: str) -> Path:
    relative = Path(name)
    require(name == relative.as_posix() and not relative.is_absolute() and
            ".." not in relative.parts and relative.parts,
            "unsafe sealed relative path: " + name)
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        require(os.path.lexists(cursor) and not cursor.is_symlink(),
                "missing/symlink sealed component: " + str(cursor))
    regular(cursor, "sealed member")
    return cursor


def canonical_write_double_seal(root: Path) -> None:
    directory(root, "result root")
    excluded = {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    members: list[Path] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "result tree contains symlink: " + str(path))
        relative = path.relative_to(root)
        if path.is_file() and relative.as_posix() not in excluded:
            members.append(relative)
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(
        sha256(root / item), item.as_posix()) for item in members), encoding="utf-8")
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")
    canonical_verify_double_seal(root)


def canonical_verify_double_seal(
    root: Path, expected_manifest_sha256: str | None = None,
    expected_outer_file_sha256: str | None = None,
) -> dict[str, str]:
    directory(root, "sealed root")
    manifest, outer = root / "SHA256SUMS", root / "SHA256SUMS.seal.sha256"
    regular(manifest, "manifest")
    regular(outer, "outer seal")
    if expected_manifest_sha256 is not None:
        require(sha256(manifest) == expected_manifest_sha256, "manifest SHA mismatch")
    if expected_outer_file_sha256 is not None:
        require(sha256(outer) == expected_outer_file_sha256, "outer file SHA mismatch")
    require(outer.read_text(encoding="utf-8").split() == [sha256(manifest), "SHA256SUMS"],
            "outer seal content mismatch")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "malformed manifest row")
        name = fields[1].lstrip("*")
        require(name not in rows, "duplicate sealed member: " + name)
        member = safe_member(root, name)
        require(sha256(member) == fields[0], "sealed payload mismatch: " + name)
        rows[name] = fields[0]
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.relative_to(root).as_posix() not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "sealed recursive population mismatch")
    return rows


def load_r1() -> Any:
    regular(R1_PATH, "sealed r1 substrate")
    require(sha256(R1_PATH) == R1_SHA256, "r1 substrate SHA drift")
    spec = importlib.util.spec_from_file_location("m1177_sealed_m1174r1", R1_PATH)
    require(spec is not None and spec.loader is not None, "cannot import r1 substrate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


R1 = load_r1()


def inventory_digest(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()


def frozen_inventory(policy: dict[str, Any]) -> dict[str, list[str]]:
    authorities = policy["inventory_authorities"]
    for entry in authorities.values():
        path = ROOT / entry["path"]
        regular(path, "inventory authority")
        require(sha256(path) == entry["sha256"], "inventory authority SHA drift")
    with (ROOT / authorities["operator_runtime"]["path"]).open(
            newline="", encoding="utf-8") as stream:
        operators = list(csv.DictReader(stream))
    fc1 = sorted({row["name"] for row in operators if row["name"].endswith(".mlp.fc1")})
    fc2 = sorted({row["name"] for row in operators if row["name"].endswith(".mlp.fc2")})
    qkv = sorted({row["name"] for row in operators if
                  row["name"].endswith(".attn.linear_q") or
                  row["name"].endswith(".attn.linear_k")})
    patch = sorted({row["name"] for row in operators if ".patch_embed." in row["name"]})
    bn: set[str] = set()
    for line in (ROOT / authorities["dependency_events"]["path"]).read_text(
            encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("module_type") in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}:
            bn.add(row["name"])
    result = {
        "c1_conv3x3": list(C1_TARGETS),
        "decoder_convtranspose": list(DECODER_TARGETS),
        "fc1": fc1, "fc2": fc2, "qkv": qkv, "patch_embed": patch,
        "batch_norm": sorted(bn),
        "attention": sorted({name.rsplit(".", 1)[0] for name in qkv}),
    }
    for category, expected in policy["expected_inventory"].items():
        if category == "atlif":
            continue
        require(len(result[category]) == expected["modules"] and
                inventory_digest(result[category]) == expected["names_sha256"],
                "frozen inventory drift: " + category)
    return result


def validate_m1175() -> dict[str, Any]:
    rows = canonical_verify_double_seal(
        M1175, M1175_MANIFEST_SHA256, M1175_OUTER_FILE_SHA256)
    require(rows.get("review.json") == M1175_REVIEW_SHA256,
            "M1175 review member SHA mismatch")
    review = strict_json(M1175 / "review.json")
    require(review.get("schema") == M1175_SCHEMA and review.get("status") == "PASS",
            "M1175 semantic admission mismatch")
    require(review.get("remote_result_manifest_sha256") == M1171_REMOTE_MANIFEST_SHA256 and
            review.get("remote_result_outer_file_sha256") == M1171_REMOTE_OUTER_FILE_SHA256,
            "M1175 does not bind exact M1171 result")
    selection = review.get("selection", {})
    require(all(selection.get(key) == value for key, value in EXPECTED_SELECTION.items()),
            "M1175 ep29 identity mismatch")
    require(review.get("authorization_after_hammer", {}).get(
        "E2_unified_ordered_capture") == "WORK_MAY_START_RESULT_NOT_ADMITTED",
        "M1175 E2 authorization mismatch")
    return review


def validate_r2_hammer(contract: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    entry = contract["inputs"]["m1177_source_hammer"]
    path = ROOT / entry["path"]
    require(path.is_relative_to(HW / "reviews"), "source hammer must be under reviews")
    rows = canonical_verify_double_seal(
        path, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("review.json") == entry["review_sha256"],
            "source hammer review member SHA mismatch")
    review = strict_json(path / "review.json")
    require(review.get("schema") ==
            "m1178_m1177_motion_unified_capture_source_hammer_r1_v1" and
            review.get("status") == "PASS", "source hammer semantic admission mismatch")
    require(review.get("source_sha256") == sha256(Path(__file__).resolve()) and
            review.get("contract_sha256") == sha256(SOURCE_CONTRACT) and
            review.get("test_sha256") == policy["test_sha256"],
            "source hammer does not bind exact r2 artifacts")
    require(review.get("authorization", {}).get("production_release") is True,
            "source hammer does not authorize release authoring")
    return review


def validate_fixed_samples(contract: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    expected = policy["frozen_samples"]
    observed = contract["cohort"]["samples"]
    require(observed == expected, "launch cohort differs from source-frozen forty rows")
    require(len(observed) == 40 and [row["global_sample_id"] for row in observed] == list(range(40)),
            "forty-sample order/population mismatch")
    require([row["cohort"] for row in observed[:10]] == ["c1"] * 10 and
            [row["sequence"] for row in observed[:10]] == ["zurich_city_09_a"] * 10,
            "C1 cohort label/sequence mismatch")
    expected_decoder = [sequence for sequence in R1.SEQUENCES for _ in range(10)]
    require([row["cohort"] for row in observed[10:]] == ["decoder"] * 30 and
            [row["sequence"] for row in observed[10:]] == expected_decoder,
            "decoder cohort label/sequence/order mismatch")
    require(len({row["path"] for row in observed}) == 40 and
            len({row["sha256"] for row in observed}) == 40 and
            len({row["sample_key"] for row in observed}) == 40,
            "frozen sources must have unique path/SHA/sample key")
    verified: list[dict[str, Any]] = []
    for row in observed:
        path = ROOT / row["path"]
        regular(path, "frozen source")
        require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"] and
                path.name == row["sample_key"], "frozen source identity mismatch")
        verified.append({**row, "resolved_path": str(path)})
    return verified


class StrictWriter(R1.UnifiedHookWriter):
    EXPECTED: dict[str, list[str]] = {}

    def _category(self, name: str, module: Any) -> str | None:
        if name in C1_TARGETS:
            return "c1_conv3x3"
        if name in DECODER_TARGETS:
            return "decoder_convtranspose"
        if module.__class__.__name__ == "ATLIFTernaryPSN":
            return "atlif"
        for category in ("fc1", "fc2", "patch_embed", "batch_norm", "qkv", "attention"):
            if name in self.EXPECTED[category]:
                return category
        return None

    def attach(self, model: Any) -> None:
        named = dict(model.named_modules())
        atlifs = sorted(name for name, module in named.items()
                        if module.__class__.__name__ == "ATLIFTernaryPSN")
        require(len(atlifs) == 105, "ATLIF topology must contain exactly 105 modules")
        self.EXPECTED = {**self.EXPECTED, "atlif": atlifs}
        for category, names in self.EXPECTED.items():
            require(names and all(name in named for name in names),
                    "missing expected module inventory: " + category)
        super().attach(model)
        require({category: sorted(names) for category, names in self.module_inventory.items()} ==
                {category: sorted(names) for category, names in self.EXPECTED.items()},
                "attached module inventory differs from exact expected inventory")
        self._r2_attached = True

    def close(self) -> None:
        if getattr(self, "_r2_attached", False):
            expected_calls = {
                category: {name: 40 for name in names}
                for category, names in self.EXPECTED.items()
            }
            observed = {category: {name: 0 for name in names}
                        for category, names in self.EXPECTED.items()}
            for row in self.records:
                require(row["category"] in observed and row["name"] in observed[row["category"]],
                        "unexpected runtime module record")
                observed[row["category"]][row["name"]] += 1
            require(observed == expected_calls, "per-module runtime call coverage is not exactly 40")
        super().close()


class StrictAttentionWriter:
    """Factory mixed with the sealed AttentionBitTraceWriter at launch time."""
    EXPECTED_ALIASES = ATTENTION_ALIASES

    def _assert_complete(self) -> None:
        require(len(self.records) == 40 * 12, "attention record count must be 40x12")
        population = {(int(row["sample_id"]), str(row["name"])) for row in self.records}
        require(population == {(sample, name) for sample in range(40)
                               for name in self.EXPECTED_ALIASES},
                "attention sample/block Cartesian coverage mismatch")
        for row in self.records:
            require(int(row.get("windows_captured", 0)) > 0 and
                    int(row.get("q_active_bits", -1)) >= 0 and
                    int(row.get("k_active_bits", -1)) >= 0 and
                    int(row.get("gate_nonzero", -1)) >= 0,
                    "attention Q/K/gate record is partial")
            path = Path(row["file"])
            regular(path, "attention NPZ payload")
            require(sha256(path) == row["sha256"], "attention payload SHA mismatch")
            import numpy as np
            with np.load(path, allow_pickle=False) as payload:
                require({"q_bits_packed", "k_bits_packed", "gate_q17"} <= set(payload.files) and
                        payload["q_bits_packed"].size > 0 and
                        payload["k_bits_packed"].size > 0 and
                        payload["gate_q17"].size > 0,
                        "attention NPZ lacks Q/K/gate payload")


def make_strict_attention_writer(base: type) -> type:
    class Concrete(StrictAttentionWriter, base):
        def write_manifest(self) -> None:
            # Base implementation writes after every record.  Avoid the
            # completeness property until the external final publication read.
            path_property = base.manifest_path
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stages = sorted({int(record["name"].split(".")[0][1:])
                             for record in self.records})
            payload = {
                "schema_version": 1, "sample_limit": self.sample_limit,
                "windows_per_call": self.windows_per_call,
                "first_block_only": self.first_block_only,
                "run_context": self.run_context, "records": self.records,
                "coverage": {"stages": stages, "stage_count": len(stages),
                             "record_count": len(self.records)},
            }
            path_property.fget(self).write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        @property
        def manifest_path(self) -> Path:
            self._assert_complete()
            return self.output_dir / "manifest.json"
    return Concrete


def validate_launch_contract(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    require(contract.get("schema") ==
            "m1177_motion_checkpoint_parametric_unified_capture_launch_r1_v1" and
            contract.get("status") ==
            "M1175_AND_M1178_BOUND__ONE_M1177_GPU_RUN_AUTHORIZED",
            "source-only or unhammered contract cannot launch")
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") ==
            "m1177_motion_checkpoint_parametric_unified_capture_source_contract_r2_v1" and
            policy.get("status") ==
            "SOURCE_ONLY__R2_HAMMER_AND_RELEASE_REQUIRED__NO_GPU",
            "canonical source policy mismatch")
    require(contract["contract_path"] == str(contract_path.relative_to(ROOT)),
            "launch contract path mismatch")
    require(contract["inputs"]["launcher"]["sha256"] == sha256(Path(__file__).resolve()) and
            contract["inputs"]["source_contract"]["path"] == str(SOURCE_CONTRACT.relative_to(ROOT)) and
            contract["inputs"]["source_contract"]["sha256"] == sha256(SOURCE_CONTRACT),
            "launch source/source-contract identity mismatch")
    require(contract["gpu_ownership"]["lease_path"] == str(CANONICAL_LEASE.relative_to(ROOT)),
            "launch contract cannot redirect canonical GPU lease")
    m1175 = validate_m1175()
    hammer = validate_r2_hammer(contract, policy)
    verified = validate_fixed_samples(contract, policy)
    require(contract["r1_compatible_binding"]["cohort"]["samples"] ==
            contract["cohort"]["samples"],
            "r1 substrate cohort differs from r2 frozen cohort")
    binding = R1.validate_launch_contract(contract["r1_compatible_binding"], contract_path)
    require(binding["selection"]["selected"]["epoch"] == 29,
            "r1 substrate binding selected wrong epoch")
    return {**binding, "m1175": m1175, "m1177_source_hammer": hammer,
            "verified_samples": verified, "policy": policy}


def run_capture(contract: dict[str, Any], binding: dict[str, Any]) -> Path:
    inventory = frozen_inventory(binding["policy"])
    StrictWriter.EXPECTED = inventory
    R1.UnifiedHookWriter = StrictWriter
    R1.write_double_seal = canonical_write_double_seal
    R1.verify_double_seal = canonical_verify_double_seal

    original_load_source = R1.load_source
    def strict_load(name: str, path: Path, expected_sha: str) -> Any:
        module = original_load_source(name, path, expected_sha)
        if name == "m1174_bit_writer":
            module.AttentionBitTraceWriter = make_strict_attention_writer(
                module.AttentionBitTraceWriter)
        return module
    R1.load_source = strict_load
    try:
        return R1.run_capture(contract["r1_compatible_binding"], binding)
    finally:
        R1.load_source = original_load_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    require(contract_path.is_relative_to(ROOT), "launch contract must be inside repository")
    contract = strict_json(contract_path)
    binding = validate_launch_contract(contract, contract_path)
    attempt = ROOT / contract["one_shot"]["attempt_marker"]
    require(attempt.is_relative_to(HW / "results") and not os.path.lexists(attempt),
            "fresh canonical attempt marker required")
    # Literal canonical lease: no contract-derived path reaches this call.
    with R1.exclusive_gpu_lease(CANONICAL_LEASE):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(attempt, flags, 0o400)
        try:
            os.write(descriptor, b"M1177_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        output = run_capture(contract, binding)
    canonical_verify_double_seal(output)
    print("PASS_M1177_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
