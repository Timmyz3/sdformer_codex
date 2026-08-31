#!/usr/bin/env python3
"""M700: H67 decoder polyphase replay on unmodified official Prosperity.

This additive runner is intentionally inert unless a later, independently
sealed static review explicitly authorizes execution.  Its exact headline
population is D0/D2/D3 only.  M686 proves that D1 has an exact scaled-binary
mask but that theta-folded FP32 execution is not bit exact; D1 is therefore a
separate opportunity diagnostic and exact decoder-complete cycles stay null.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.util
import io
import json
import math
import multiprocessing as mp
import os
import re
import stat
import subprocess
import sys
import tempfile
import types
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = (
    ROOT / "contracts" /
    "m700_h67_ep35_decoder_official_prosperity_iso_workload_contract_r2_20260828.json"
)
DEFAULT_OUT = (
    ROOT / "results" /
    "m700_h67_ep35_decoder_official_prosperity_dev_r2_20260828"
)
SINGLE_WRITER_LOCK = (
    ROOT / "results" /
    ".m700_h67_ep35_decoder_official_prosperity_dev_r2.single_writer.lock"
)

N_TILE = 128
M_TILE = 256
K_TILE = 16
MEM_IF_WIDTH = 1024
PHASE_ORDER = (3, 2, 1, 0)
EXACT_MODULES = (0, 2, 3)
DIAGNOSTIC_MODULES = (1,)
COUNTER_FIELDS = (
    "total_cycles", "compute_cycles", "raw_issue_cycles",
    "raw_preprocess_cycles", "preprocess_stall_cycles",
    "memory_stall_cycles", "num_ops", "dram_reads", "dram_writes",
    "g_act_reads", "g_act_writes", "g_wgt_reads", "g_wgt_writes",
    "g_psum_reads", "g_psum_writes",
)

_MAPPER = None
_FC = None
_SIMULATOR = None
_ACCELERATOR = None
_SIM_MODULE = None
_ACCELS: dict[bool, Any] = {}
_PAYLOAD_ROOT: Path | None = None
_WEIGHT_IDENTITIES: dict[str, Any] = {}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: Any) -> str:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject(value: str) -> None:
        raise RuntimeError("M700 non-standard JSON constant: " + value)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "M700 duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )


def safe_member(value: str) -> PurePosixPath:
    require(isinstance(value, str), "M700 member name must be a string")
    member = PurePosixPath(value)
    require(
        member.parts and not member.is_absolute() and
        member.parts[0] not in ("", ".") and ".." not in member.parts,
        "M700 unsafe member: " + value,
    )
    return member


def real_directory(path: Path, label: str) -> Path:
    path = Path(path)
    require(path.is_absolute() and path != Path("/") and ".." not in path.parts,
            label + " must be a non-root absolute path")
    cursor = Path(path.anchor)
    for part in path.parts[1:]:
        cursor /= part
        observed = os.lstat(str(cursor))
        require(stat.S_ISDIR(observed.st_mode) and
                not stat.S_ISLNK(observed.st_mode),
                label + " contains a symlink or non-directory")
    require(cursor.resolve(strict=True) == cursor,
            label + " lexical/resolved identity differs")
    return cursor


def trusted_file(root: Path, member: str | PurePosixPath, label: str) -> Path:
    root = real_directory(root, label + " root")
    member = safe_member(str(member))
    cursor = root
    for index, part in enumerate(member.parts):
        cursor /= part
        observed = os.lstat(str(cursor))
        require(not stat.S_ISLNK(observed.st_mode),
                label + " contains a symlink")
        if index + 1 == len(member.parts):
            require(stat.S_ISREG(observed.st_mode),
                    label + " leaf is not a regular file")
        else:
            require(stat.S_ISDIR(observed.st_mode),
                    label + " parent is not a directory")
    require(cursor.resolve(strict=True).is_relative_to(root),
            label + " resolves outside root")
    return cursor


def sealed_members(directory: Path) -> list[Path]:
    excluded = {"SHA256SUMS", "SHA256SUMS.seal.sha256"}
    return [
        path.relative_to(directory)
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.relative_to(directory).as_posix()
        not in excluded
    ]


def write_double_seal(directory: Path) -> None:
    directory = real_directory(Path(directory).resolve(), "M700 output")
    seal = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(not seal.exists() and not outer.exists(),
            "M700 refuses to overwrite a seal")
    seal.write_text("".join(
        f"{sha256_file(directory / member)}  {member.as_posix()}\n"
        for member in sealed_members(directory)
    ), encoding="utf-8")
    outer.write_text(
        f"{sha256_file(seal)}  SHA256SUMS\n", encoding="utf-8"
    )


def verify_double_seal(
    directory: Path,
    *,
    expected_manifest_file_sha256: str | None = None,
    expected_outer_file_sha256: str | None = None,
) -> dict[str, str]:
    directory = real_directory(Path(directory).resolve(), "M700 sealed package")
    seal = trusted_file(directory, "SHA256SUMS", "M700 inner seal")
    outer = trusted_file(
        directory, "SHA256SUMS.seal.sha256", "M700 outer seal"
    )
    if expected_manifest_file_sha256 is not None:
        require(sha256_file(seal) == expected_manifest_file_sha256,
                "M700 sealed-manifest file SHA mismatch")
    if expected_outer_file_sha256 is not None:
        require(sha256_file(outer) == expected_outer_file_sha256,
                "M700 outer-seal file SHA mismatch")
    outer_tokens = outer.read_text(encoding="utf-8").strip().split("  ", 1)
    require(len(outer_tokens) == 2 and outer_tokens[1] == "SHA256SUMS" and
            outer_tokens[0] == sha256_file(seal),
            "M700 outer seal content mismatch")
    listed: set[str] = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        tokens = line.split("  ", 1)
        require(len(tokens) == 2 and len(tokens[0]) == 64,
                "M700 malformed seal line")
        member = safe_member(tokens[1]).as_posix()
        require(member not in listed and member not in
                ("SHA256SUMS", "SHA256SUMS.seal.sha256"),
                "M700 duplicate/recursive sealed member")
        path = trusted_file(directory, member, "M700 sealed member")
        require(sha256_file(path) == tokens[0],
                "M700 sealed member SHA mismatch: " + member)
        listed.add(member)
    actual = {member.as_posix() for member in sealed_members(directory)}
    require(actual == listed, "M700 sealed population mismatch")
    return {
        "manifest_file_sha256": sha256_file(seal),
        "outer_seal_file_sha256": sha256_file(outer),
    }


def product(values: list[int]) -> int:
    result = 1
    for value in values:
        require(isinstance(value, int) and not isinstance(value, bool) and
                value > 0, "M700 invalid positive shape dimension")
        result *= value
    return result


def popcount_file(path: Path, elements: int) -> tuple[int, int]:
    table = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)
    raw = np.fromfile(path, dtype=np.uint8)
    expected_bytes = (elements + 7) // 8
    require(raw.size == expected_bytes, "M700 packed-byte length mismatch")
    used = elements & 7
    if used:
        require((int(raw[-1]) & (~((1 << used) - 1) & 0xff)) == 0,
                "M700 nonzero packed tail")
    ones = int(table[raw].sum(dtype=np.uint64))
    return ones, int(raw.size)


def git_text(repo: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), *args], text=True
    ).strip()


def verify_frozen_file(entry: dict[str, Any], label: str) -> dict[str, Any]:
    raw = Path(entry["path"])
    if raw.is_absolute():
        require(".." not in raw.parts and "." not in raw.parts,
                label + " absolute path has forbidden lexical components")
        root = real_directory(raw.parent, label + " parent")
        path = trusted_file(root, raw.name, label)
    else:
        member = safe_member(raw.as_posix())
        path = trusted_file(ROOT, member, label)
    observed = sha256_file(path)
    require(observed == entry["sha256"], label + " SHA mismatch")
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": observed}


def validate_m692(contract: dict[str, Any]) -> dict[str, Any]:
    frozen = contract["frozen_inputs"]
    raw_directory = Path(frozen["m692_review_directory"]["path"])
    if raw_directory.is_absolute():
        directory = raw_directory
    else:
        directory = ROOT / safe_member(raw_directory.as_posix())
    seal = verify_double_seal(
        directory,
        expected_manifest_file_sha256=frozen["m692_review_directory"]["manifest_file_sha256"],
        expected_outer_file_sha256=frozen["m692_review_directory"]["outer_seal_file_sha256"],
    )
    review_path = trusted_file(
        directory, frozen["m692_review_directory"]["review_member"],
        "M692 review",
    )
    review = strict_json(review_path)
    expected = contract["m692_admission"]
    require(review.get("status") == expected["status"],
            "M700 M692 status mismatch")
    require(review.get("severity") == {"p0": 0, "p1": 0, "p2": 0},
            "M700 M692 severity mismatch")
    require(review.get("go") is True,
            "M700 M692 does not explicitly GO")
    require(sha256_file(review_path) ==
            frozen["m692_review_directory"]["review_sha256"],
            "M700 M692 review SHA mismatch")
    return {"directory": str(directory), "review_sha256": sha256_file(review_path), **seal}


def validate_payload(contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    package_entry = contract["frozen_inputs"]["m686_package"]
    raw_package = Path(package_entry["path"])
    if raw_package.is_absolute():
        package = raw_package
    else:
        package = ROOT / safe_member(raw_package.as_posix())
    package = real_directory(package, "M686 package")
    top_seal = verify_double_seal(
        package,
        expected_manifest_file_sha256=package_entry["manifest_file_sha256"],
        expected_outer_file_sha256=package_entry["outer_seal_file_sha256"],
    )
    for subdir in ("runtime_receipt", "weights"):
        expected = package_entry["nested_seals"][subdir]
        verify_double_seal(
            package / subdir,
            expected_manifest_file_sha256=expected["manifest_file_sha256"],
            expected_outer_file_sha256=expected["outer_seal_file_sha256"],
        )
    manifest_path = trusted_file(package, "manifest.json", "M686 manifest")
    require(sha256_file(manifest_path) == package_entry["manifest_sha256"],
            "M700 M686 manifest SHA mismatch")
    require(sha256_file(trusted_file(package, "RUN_COMPLETE.txt", "M686 completion")) ==
            package_entry["run_complete_sha256"],
            "M700 M686 completion SHA mismatch")
    manifest = strict_json(manifest_path)
    require(manifest.get("schema") ==
            "m660_h67_ep35_layer_static_decoder_payload_v1",
            "M700 M686 schema mismatch")
    require(manifest.get("status") ==
            "PASS_S10_ALL4_SCALED_BINARY__D1_FOLDED_WEIGHT_MITER_NONEXACT",
            "M700 unexpected M686 status")
    require(manifest.get("packing") == {
        "values": [0, 1], "bit_order": "little",
        "order": "C_ORDER_FLAT",
        "whole_call_contiguous_copy_allowed": False,
    }, "M700 M686 packing drift")
    require(manifest.get("layer_static_route_table") == {
        "d0": "EXACT_BINARY_BITPACK",
        "d1": "EXACT_SCALED_BINARY_BITPACK",
        "d2": "EXACT_BINARY_BITPACK",
        "d3": "EXACT_BINARY_BITPACK",
    }, "M700 M686 route-table drift")
    d1 = manifest.get("d1_dual_result_decision", {})
    require(d1 == contract["d1_policy"]["required_decision"],
            "M700 M686 D1 decision drift")
    require(manifest.get("population") == contract["m686_population"],
            "M700 M686 population drift")

    modules = {int(row["module_index"]): row for row in contract["decoder_modules"]}
    exact_records = manifest.get("d0_d2_d3_binary_records")
    diagnostic_records = manifest.get("d1_records")
    require(isinstance(exact_records, list) and len(exact_records) == 30,
            "M700 exact record population mismatch")
    require(isinstance(diagnostic_records, list) and len(diagnostic_records) == 10,
            "M700 D1 diagnostic population mismatch")
    observed: set[tuple[int, int]] = set()
    normalized_exact: list[dict[str, Any]] = []
    normalized_diag: list[dict[str, Any]] = []
    for row in exact_records + diagnostic_records:
        sample = row.get("sample_id")
        module = row.get("module_index")
        require(isinstance(sample, int) and 0 <= sample < 10 and
                isinstance(module, int) and module in modules,
                "M700 invalid sample/module identity")
        require((sample, module) not in observed,
                "M700 duplicate sample/module record")
        observed.add((sample, module))
        expected = modules[module]
        require(row.get("name") == expected["name"] and
                row.get("input_shape") == expected["input_shape"],
                "M700 module name/shape drift")
        if module in EXACT_MODULES:
            require(row.get("route") == "EXACT_BINARY_BITPACK",
                    "M700 exact-subset route drift")
            identity = row.get("input")
            expected_ones = identity.get("one_count")
            destination = normalized_exact
            role = "EXACT_BINARY_OFFICIAL_SUBSET"
        else:
            require(module == 1 and row.get("route") ==
                    "EXACT_SCALED_BINARY_BITPACK",
                    "M700 D1 diagnostic route drift")
            identity = row.get("theta_binary_candidate")
            require(identity.get("theta_gate_pass") is True and
                    identity.get("other_finite_count") == 0 and
                    identity.get("nonfinite_count") == 0 and
                    row.get("folded_weight_miter", {}).get("bit_exact") is False,
                    "M700 D1 diagnostic identity drift")
            expected_ones = identity.get("theta_count")
            destination = normalized_diag
            role = "SCALED_BINARY_OPPORTUNITY_DIAGNOSTIC_ONLY"
        elements = identity.get("elements")
        require(isinstance(elements, int) and
                elements == product(row["input_shape"]),
                "M700 payload element arithmetic mismatch")
        path = trusted_file(package, row["relative_path"], "M686 bitpack")
        require(sha256_file(path) == identity.get("packed_sha256"),
                "M700 payload SHA mismatch")
        ones, packed_bytes = popcount_file(path, elements)
        require(ones == expected_ones and packed_bytes == identity.get("packed_bytes"),
                "M700 payload popcount/bytes mismatch")
        destination.append({
            "sample_id": sample,
            "sample_key": row["sample_key"],
            "sequence_key": row["sequence_key"],
            "module_index": module,
            "module": row["name"],
            "route": row["route"],
            "admission_role": role,
            "relative_path": row["relative_path"],
            "path": str(path),
            "shape": row["input_shape"],
            "elements": elements,
            "active_elements": ones,
            "packed_bytes": packed_bytes,
            "packed_sha256": identity["packed_sha256"],
        })
    require(observed == {(s, m) for s in range(10) for m in range(4)},
            "M700 40-cell lattice mismatch")

    weights = manifest.get("weight_payloads", {})
    for module, expected in modules.items():
        identity = weights.get(str(module))
        require(identity is not None and identity.get("shape") ==
                expected["weight_shape"] and
                identity.get("layout") == "C_ORDER_CONTIGUOUS" and
                identity.get("dtype") == "torch.float32" and
                identity.get("byte_order") == "little",
                "M700 weight identity drift")
        path = trusted_file(package, identity["relative_path"], "M686 weight")
        require(sha256_file(path) == identity["content_sha256"] and
                path.stat().st_size == identity["content_bytes"],
                "M700 weight payload mismatch")
    folded = weights.get("d1_folded_theta", {})
    require(folded.get("role") == "DIAGNOSTIC_CANDIDATE_NOT_ADMITTED" and
            folded.get("deployment_admitted") is False,
            "M700 D1 folded candidate was silently admitted")
    return {
        "package": str(package), "manifest_sha256": sha256_file(manifest_path),
        **top_seal,
    }, sorted(normalized_exact, key=lambda x: (x["sample_id"], x["module_index"])), sorted(normalized_diag, key=lambda x: x["sample_id"])


def verify_official_repo(contract: dict[str, Any]) -> dict[str, Any]:
    entry = contract["frozen_inputs"]["official_prosperity_repo"]
    repo = real_directory(Path(entry["path"]).resolve(), "official Prosperity repo")
    commit = git_text(repo, "rev-parse", "HEAD")
    dirty = git_text(repo, "status", "--porcelain", "--untracked-files=all")
    require(commit == entry["commit"], "M700 official commit mismatch")
    require(not dirty, "M700 official repository is dirty")
    files = {}
    for member, expected in entry["files"].items():
        path = trusted_file(repo, member, "official source")
        actual = sha256_file(path)
        require(actual == expected, "M700 official source SHA mismatch: " + member)
        files[member] = actual
    return {"path": str(repo), "commit": commit, "clean": True, "files": files}


def preflight(contract: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    require(contract.get("schema") ==
            "m700_h67_ep35_decoder_official_prosperity_iso_workload_contract_v2",
            "M700 contract schema mismatch")
    checked = {
        "m692": validate_m692(contract),
        "official_prosperity_repo": verify_official_repo(contract),
    }
    for label in (
        "m689_review", "m689_manifest", "m689_outer_seal",
        "m686_contract", "m686_producer", "m511_contract",
        "m672_mapper", "m672_test", "m670_mapper", "m670_test",
        "m677_review", "m677_manifest", "m677_outer_seal",
        "m618_reference_runner", "m618_reference_contract",
        "m619_review", "m619_manifest", "m619_outer_seal", "docs359",
        "m693_r1_runner", "m693_r1_contract", "m693_r1_test",
        "m697_review", "m697_manifest", "m697_outer_seal",
    ):
        checked[label] = verify_frozen_file(
            contract["frozen_inputs"][label], label
        )
    payload_identity, exact_records, diagnostic_records = validate_payload(contract)
    checked["m686_package"] = payload_identity
    checked["contract"] = {
        "path": str(CONTRACT), "sha256": sha256_file(CONTRACT)
    }
    checked["runner"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": sha256_file(Path(__file__).resolve()),
    }
    return checked, exact_records, diagnostic_records


def load_mapper(contract: dict[str, Any]) -> Any:
    path = trusted_file(
        ROOT, contract["frozen_inputs"]["m672_mapper"]["path"],
        "M700 mapper import",
    )
    require(sha256_file(path) ==
            contract["frozen_inputs"]["m672_mapper"]["sha256"],
            "M700 mapper drift before import")
    spec = importlib.util.spec_from_file_location("m700_frozen_m672", str(path))
    require(spec is not None and spec.loader is not None,
            "M700 cannot construct mapper import")
    module = importlib.util.module_from_spec(spec)
    old_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = old_dont_write
    require(sha256_file(path) ==
            contract["frozen_inputs"]["m672_mapper"]["sha256"],
            "M700 mapper drift across import")
    return module


def load_official_api(repo: Path) -> tuple[Any, Any, Any, Any]:
    names = ("utils", "configs", "accelerator", "networks", "baselines",
             "energy", "simulator", "prosparsity_engine")
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    saved_path = list(sys.path)
    old_dont_write = sys.dont_write_bytecode
    sys.path[:] = [str(repo / "simulator")] + [
        item for item in sys.path if item != str(repo / "simulator")
    ]
    sys.dont_write_bytecode = True
    try:
        sys.modules["prosparsity_engine"] = types.ModuleType("prosparsity_engine")
        accelerator_module = importlib.import_module("accelerator")
        networks_module = importlib.import_module("networks")
        simulator_module = importlib.import_module("simulator")
        return (
            accelerator_module.Accelerator,
            networks_module.FC,
            simulator_module.Simulator,
            simulator_module,
        )
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path[:] = saved_path
        sys.dont_write_bytecode = old_dont_write


def worker_init(contract_path: str) -> None:
    global _MAPPER, _FC, _SIMULATOR, _ACCELERATOR, _SIM_MODULE, _ACCELS
    global _PAYLOAD_ROOT, _WEIGHT_IDENTITIES
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    contract = strict_json(Path(contract_path))
    _MAPPER = load_mapper(contract)
    repo = Path(contract["frozen_inputs"]["official_prosperity_repo"]["path"])
    _ACCELERATOR, _FC, _SIMULATOR, _SIM_MODULE = load_official_api(repo)
    _ACCELS = {
        product_mode: _ACCELERATOR(
            type="Prosperity", adder_array_size=N_TILE, LIF_array_size=32,
            tile_size_M=M_TILE, tile_size_K=K_TILE,
            product_sparsity=product_mode, dense=False, issue_type=2,
            mem_if_width=MEM_IF_WIDTH,
        )
        for product_mode in (False, True)
    }
    _PAYLOAD_ROOT = real_directory(
        ROOT / safe_member(contract["frozen_inputs"]["m686_package"]["path"]),
        "M700 worker payload root",
    )
    manifest = strict_json(_PAYLOAD_ROOT / "manifest.json")
    _WEIGHT_IDENTITIES = manifest["weight_payloads"]


def materialize_phase(
    record: dict[str, Any], bank: int
) -> tuple[torch.Tensor, dict[str, int]]:
    require(_MAPPER is not None and _PAYLOAD_ROOT is not None,
            "M700 worker mapper is not initialized")
    chunks = []
    valid_slots_per_time = 0
    materialized_entries_per_time = 0
    active_tap_events = 0
    for tile in _MAPPER.iter_polyphase_tiles(
        record["path"], record["shape"], tile_m=M_TILE, phases=[bank],
        trusted_root=_PAYLOAD_ROOT,
    ):
        require(tile["phase_bank"] == bank and tile["values"].dtype == np.uint8,
                "M700 mapper phase/dtype drift")
        chunks.append(tile["values"])
        valid_slots_per_time += int(tile["valid"].sum(dtype=np.int64))
        materialized_entries_per_time += int(tile["valid"].size)
        active_tap_events += int(tile["values"].sum(dtype=np.int64))
    require(chunks, "M700 mapper returned no phase tiles")
    values = np.concatenate(chunks, axis=1)
    t_dim, _batch, channels, height, width = record["shape"]
    expected_k = len(_MAPPER.M514_PHASE_TAPS[bank]) * channels
    require(values.shape == (t_dim, height * width, expected_k) and
            np.all(np.logical_or(values == 0, values == 1)),
            "M700 materialized phase identity drift")
    return torch.from_numpy(values.copy()), {
        "valid_tap_slots_per_time": valid_slots_per_time,
        "valid_tap_slots_all_time": valid_slots_per_time * t_dim,
        "materialized_entries_per_time": materialized_entries_per_time,
        "materialized_entries_all_time": materialized_entries_per_time * t_dim,
        "structural_padding_zero_entries_all_time": (
            materialized_entries_per_time - valid_slots_per_time
        ) * t_dim,
        "active_tap_events": active_tap_events,
    }


def phase_weight_identity(record: dict[str, Any], bank: int) -> dict[str, Any]:
    require(_MAPPER is not None and _PAYLOAD_ROOT is not None,
            "M700 worker mapper is not initialized")
    key = str(record["module_index"])
    if record["module_index"] == 1:
        key = "d1_folded_theta"
    identity = _WEIGHT_IDENTITIES[key]
    path = trusted_file(
        _PAYLOAD_ROOT, identity["relative_path"], "M700 phase weight consume"
    )
    weight = np.memmap(path, dtype="<f4", mode="r", shape=tuple(identity["shape"]))
    matrix = np.ascontiguousarray(_MAPPER.phase_weight_matrix(weight, bank))
    expected = (len(_MAPPER.M514_PHASE_TAPS[bank]) * record["shape"][2],
                identity["shape"][1])
    require(matrix.shape == expected, "M700 phase weight shape drift")
    return {
        "source_relative_path": identity["relative_path"],
        "source_sha256": identity["content_sha256"],
        "source_role": identity.get("role", "FROZEN_ORIGINAL_WEIGHT"),
        "phase_shape": list(matrix.shape),
        "phase_sha256": hashlib.sha256(matrix.tobytes(order="C")).hexdigest(),
        "official_weight_values_consumed": False,
        "official_weight_bits": 8,
    }


def parse_official_stdout(raw: str) -> dict[str, int]:
    result = {}
    for field, pattern in {
        "raw_issue_cycles": r"^compute cycles:\s+([0-9]+)\s*$",
        "raw_preprocess_cycles": r"^preprocess cycles:\s+([0-9]+)\s*$",
    }.items():
        match = re.search(pattern, raw, flags=re.MULTILINE)
        require(match is not None, "M700 official stdout lacks " + field)
        result[field] = int(match.group(1))
    return result


def run_official(
    activation: torch.Tensor,
    *,
    module_index: int,
    bank: int,
    product_mode: bool,
    output_dim: int,
) -> dict[str, int]:
    require(_FC is not None and _SIMULATOR is not None and _ACCELERATOR is not None,
            "M700 official API is not initialized")
    t_dim, sequence, k_dim = activation.shape
    name = f"h67_decoder_d{module_index}_phase{bank}_polyphase"
    require(not name.endswith("_fc"), "M700 operator name triggers Conv2d billing")
    op = _FC(name, k_dim, output_dim, sequence, 1, t_dim)
    op.activation_tensor.sparse_map = activation
    accelerator = _ACCELS[product_mode]
    if hasattr(_SIM_MODULE, "clear_global_stats"):
        _SIM_MODULE.clear_global_stats()
    sim = _SIMULATOR(
        accelerator=accelerator, network=[op],
        benchmark_name="h67_ep35_decoder_polyphase_s10", use_cuda=False,
    )
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        stats = sim.run_fc(
            op, spike_stored_in_buffer=False, weight_stored_in_buffer=False
        )
    parsed = parse_official_stdout(captured.getvalue())
    return {
        "total_cycles": int(stats.total_cycles),
        "compute_cycles": int(stats.compute_cycles),
        **parsed,
        "preprocess_stall_cycles": int(stats.preprocess_stall_cycles),
        "memory_stall_cycles": int(stats.mem_stall_cycles),
        "num_ops": int(stats.num_ops),
        "dram_reads": int(stats.reads["dram"]),
        "dram_writes": int(stats.writes["dram"]),
        "g_act_reads": int(stats.reads["g_act"]),
        "g_act_writes": int(stats.writes["g_act"]),
        "g_wgt_reads": int(stats.reads["g_wgt"]),
        "g_wgt_writes": int(stats.writes["g_wgt"]),
        "g_psum_reads": int(stats.reads["g_psum"]),
        "g_psum_writes": int(stats.writes["g_psum"]),
    }


def expand_exact_n128(
    one_tile: dict[str, int], *, m_dim: int, k_dim: int, n_dim: int
) -> dict[str, int]:
    require(n_dim > 0 and n_dim % N_TILE == 0,
            "M700 exact N128 expansion requires integral N tiles")
    n_tiles = n_dim // N_TILE
    expanded = {
        field: one_tile[field] * n_tiles
        for field in COUNTER_FIELDS
        if field not in ("total_cycles", "memory_stall_cycles")
    }
    initial_bits = min(K_TILE, k_dim) * N_TILE * 8
    initial_bits += min(K_TILE, k_dim) * min(M_TILE, m_dim)
    middle_bits = expanded["dram_reads"] + expanded["dram_writes"] - initial_bits
    require(middle_bits >= 0, "M700 N expansion has negative middle transfer")
    initial_latency = initial_bits // MEM_IF_WIDTH
    middle_latency = middle_bits // MEM_IF_WIDTH
    expanded["memory_stall_cycles"] = initial_latency + max(
        0, middle_latency - expanded["compute_cycles"]
    )
    expanded["total_cycles"] = (
        expanded["compute_cycles"] + expanded["memory_stall_cycles"]
    )
    return expanded


def derived(counters: dict[str, int], output_dim: int) -> dict[str, Any]:
    return {
        **counters,
        "dram_bits": counters["dram_reads"] + counters["dram_writes"],
        "global_buffer_bits": sum(counters[field] for field in (
            "g_act_reads", "g_act_writes", "g_wgt_reads", "g_wgt_writes",
            "g_psum_reads", "g_psum_writes",
        )),
        "support_nnz": counters["num_ops"] // output_dim,
        "support_nnz_divisible_by_output_N": counters["num_ops"] % output_dim == 0,
    }


def worker_run(record: dict[str, Any]) -> dict[str, Any]:
    phases = []
    module = record["module_index"]
    output_dim = _WEIGHT_IDENTITIES[str(module)]["shape"][1]
    for bank in PHASE_ORDER:
        activation, support_accounting = materialize_phase(record, bank)
        t_dim, sequence, k_dim = activation.shape
        m_dim = t_dim * sequence
        modes = {}
        d0_miters = []
        for product_mode in (False, True):
            label = "product" if product_mode else "bit"
            direct = run_official(
                activation, module_index=module, bank=bank,
                product_mode=product_mode, output_dim=output_dim,
            )
            modes[label] = derived(direct, output_dim)
            if module == 0:
                primitive = run_official(
                    activation, module_index=module, bank=bank,
                    product_mode=product_mode, output_dim=N_TILE,
                )
                expanded = expand_exact_n128(
                    primitive, m_dim=m_dim, k_dim=k_dim, n_dim=output_dim
                )
                mismatches = {
                    field: {"direct": direct[field], "expanded": expanded[field]}
                    for field in COUNTER_FIELDS if direct[field] != expanded[field]
                }
                require(not mismatches, "M700 D0 direct-vs-N128x3 miter failed")
                d0_miters.append({"mode": label, "mismatches": mismatches, "pass": True})
        require(modes["bit"]["support_nnz"] ==
                support_accounting["active_tap_events"],
                "M700 bit-mode support NNZ disagrees with exact mapped activity")
        phases.append({
            "phase_bank": bank,
            "shape": {"T": t_dim, "S": sequence, "M": m_dim,
                      "K": k_dim, "N": output_dim},
            "tiles": {
                "M": math.ceil(m_dim / M_TILE),
                "K": math.ceil(k_dim / K_TILE),
                "N": math.ceil(output_dim / N_TILE),
                "M_padding_rows": math.ceil(m_dim / M_TILE) * M_TILE - m_dim,
                "K_padding_channels": math.ceil(k_dim / K_TILE) * K_TILE - k_dim,
                "N_padding_channels": math.ceil(output_dim / N_TILE) * N_TILE - output_dim,
                "official_policy": "direct full N; cur tile dimensions; no synthetic active padding",
            },
            "weight_mapping": phase_weight_identity(record, bank),
            "support_accounting": {
                **support_accounting,
                "active_products": support_accounting["active_tap_events"] * output_dim,
                "dense_legal_products": support_accounting["valid_tap_slots_all_time"] * output_dim,
                "structural_boundary_zeros_are_not_data_sparsity": True,
            },
            "modes": modes,
            "d0_direct_vs_n128x3_miter": d0_miters,
            "product_vs_bit_speedup": (
                modes["bit"]["total_cycles"] /
                max(1, modes["product"]["total_cycles"])
            ),
        })
    return {
        **{key: record[key] for key in (
            "sample_id", "sample_key", "sequence_key", "module_index",
            "module", "route", "admission_role", "relative_path",
            "packed_sha256", "elements", "active_elements",
        )},
        "phases": phases,
    }


def selected_phases(
    rows: list[dict[str, Any]], phase_bank: int | None = None
) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        phases = row["phases"]
        require(
            [phase["phase_bank"] for phase in phases] == list(PHASE_ORDER),
            "M700 phase population/order drift",
        )
        if phase_bank is None:
            result.extend(phases)
        else:
            require(phase_bank in PHASE_ORDER,
                    "M700 aggregate phase is outside the frozen order")
            matching = [phase for phase in phases
                        if phase["phase_bank"] == phase_bank]
            require(len(matching) == 1,
                    "M700 aggregate phase is missing or duplicated")
            result.extend(matching)
    return result


def sum_counters(
    rows: list[dict[str, Any]], mode: str, phase_bank: int | None = None
) -> dict[str, int]:
    result = {field: 0 for field in COUNTER_FIELDS}
    result.update({"dram_bits": 0, "global_buffer_bits": 0, "support_nnz": 0})
    for phase in selected_phases(rows, phase_bank):
        counters = phase["modes"][mode]
        for field in result:
            result[field] += int(counters[field])
    return result


def aggregate(
    rows: list[dict[str, Any]], phase_bank: int | None = None
) -> dict[str, Any]:
    require(rows, "M700 cannot aggregate an empty population")
    bit = sum_counters(rows, "bit", phase_bank)
    product_result = sum_counters(rows, "product", phase_bank)
    phase_ratios = [
        phase["product_vs_bit_speedup"]
        for phase in selected_phases(rows, phase_bank)
    ]
    support_accounting = defaultdict(int)
    for phase in selected_phases(rows, phase_bank):
        for field, value in phase.get("support_accounting", {}).items():
            if isinstance(value, int) and not isinstance(value, bool):
                support_accounting[field] += value
    return {
        "records": len(rows),
        "phase_bank": phase_bank,
        "support_calls_per_mode": len(phase_ratios),
        "bit": bit, "product": product_result,
        "aggregate_cycle_ratio_speedup": (
            bit["total_cycles"] / max(1, product_result["total_cycles"])
        ),
        "per_support_call_speedup_distribution": {
            "geometric_mean": math.exp(
                sum(math.log(value) for value in phase_ratios) /
                len(phase_ratios)
            ),
            "minimum": min(phase_ratios), "maximum": max(phase_ratios),
            "arithmetic_mean": sum(phase_ratios) / len(phase_ratios),
        },
        "product_support_reduction": 1.0 - (
            product_result["support_nnz"] / max(1, bit["support_nnz"])
        ),
        "mapped_support_accounting": dict(support_accounting),
    }


def aggregate_breakdowns(rows: list[dict[str, Any]]) -> dict[str, Any]:
    require(rows, "M700 cannot build breakdowns for an empty population")
    buckets: dict[str, list[dict[str, Any]]] = {"overall": list(rows)}
    for row in rows:
        buckets.setdefault(f"sample:{row['sample_id']:02d}", []).append(row)
        buckets.setdefault(f"module:{row['module_index']}", []).append(row)
        buckets[f"record:s{row['sample_id']:02d}_d{row['module_index']}"] = [row]
    result = {key: aggregate(value) for key, value in sorted(buckets.items())}
    for bank in PHASE_ORDER:
        result[f"phase:{bank}"] = aggregate(rows, phase_bank=bank)

    overall = result["overall"]
    for mode in ("bit", "product"):
        for field, expected in overall[mode].items():
            observed = sum(
                int(result[f"phase:{bank}"][mode][field])
                for bank in PHASE_ORDER
            )
            require(observed == int(expected),
                    f"M700 phase-to-overall {mode}/{field} conservation failed")
    support_fields = set(overall["mapped_support_accounting"])
    require(all(
        set(result[f"phase:{bank}"]["mapped_support_accounting"]) ==
        support_fields for bank in PHASE_ORDER
    ), "M700 phase support-accounting field population drift")
    for field, expected in overall["mapped_support_accounting"].items():
        observed = sum(
            int(result[f"phase:{bank}"]["mapped_support_accounting"][field])
            for bank in PHASE_ORDER
        )
        require(observed == int(expected),
                "M700 phase-to-overall support-accounting conservation failed")
    require(sum(result[f"phase:{bank}"]["support_calls_per_mode"]
                for bank in PHASE_ORDER) ==
            overall["support_calls_per_mode"],
            "M700 phase-to-overall call conservation failed")
    return dict(sorted(result.items()))


def validate_execution_authorization(directory: Path, expected_outer_sha: str) -> dict[str, Any]:
    directory = Path(directory)
    if not directory.is_absolute():
        directory = Path.cwd() / directory
    directory = real_directory(directory, "M700 static review")
    seal = verify_double_seal(
        directory, expected_outer_file_sha256=expected_outer_sha
    )
    review_path = trusted_file(directory, "review.json", "M700 static review JSON")
    review = strict_json(review_path)
    require(review.get("status") ==
            "GO_M700_FULL_OFFICIAL_CPU_REPLAY__P0_0_P1_0" and
            review.get("severity", {}).get("p0") == 0 and
            review.get("severity", {}).get("p1") == 0 and
            review.get("go") is True and
            review.get("execution_authorized") is True,
            "M700 static review does not authorize execution")
    target = review.get("frozen_target", {})
    require(target.get("runner_sha256") ==
            sha256_file(Path(__file__).resolve()) and
            target.get("contract_sha256") == sha256_file(CONTRACT) and
            target.get("test_sha256") == sha256_file(
                ROOT / "system_simulator/tests/test_m700_decoder_official_prosperity_adapter_r2.py") and
            target.get("m686_manifest_sha256") ==
            "c06de650b50db92dd0c374b57f0ce3ea72cfb3dcd18a369aea7d552341e5bb33" and
            target.get("m692_review_sha256") ==
            "5088e36fa935536766f51f4e58c198d16f49ac3fe415b2f3d6432b184a36f49f" and
            target.get("m697_review_sha256") ==
            "f5fd5a172cd011654224aa0591df30518c0753d9b563f88535ff42ad39188dd1",
            "M700 static review frozen-target identity mismatch")
    return {"directory": str(directory), "review_sha256": sha256_file(review_path), **seal}


def execute_records(
    records: list[dict[str, Any]], *, workers: int
) -> list[dict[str, Any]]:
    require(workers == 3, "M700 workers are frozen at exactly 3")
    context = mp.get_context("fork")
    completed = []
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context,
        initializer=worker_init, initargs=(str(CONTRACT),),
    ) as executor:
        for count, row in enumerate(executor.map(worker_run, records, chunksize=1), 1):
            completed.append(row)
            if count % 5 == 0 or count == len(records):
                print(f"M700 records {count}/{len(records)}", flush=True)
    return sorted(completed, key=lambda row: (row["sample_id"], row["module_index"]))


def build_report(
    contract: dict[str, Any], identity: dict[str, Any],
    exact_rows: list[dict[str, Any]], diagnostic_rows: list[dict[str, Any]],
    authorization: dict[str, Any], *, workers: int = 3,
) -> dict[str, Any]:
    require(workers == 3, "M700 report worker count is frozen at 3")
    exact = aggregate_breakdowns(exact_rows)
    diagnostic = aggregate_breakdowns(diagnostic_rows)
    report = {
        "schema": "m700_h67_ep35_decoder_official_prosperity_iso_workload_v2",
        "date": "2026-08-28",
        "status": "PASS_EXTERNAL_OFFICIAL_DECODER_SUPPORT_SUBSET__D1_DIAGNOSTIC_ONLY__NOT_OURS_OR_COMPLETE",
        "identity": identity,
        "execution_authorization": authorization,
        "execution": {
            "workers": workers,
            "worker_policy": "FIXED_THREE_PROCESS_CPU_REPLAY",
            "single_writer_lock": SINGLE_WRITER_LOCK.name,
        },
        "configuration": contract["official_configuration"],
        "mapping": contract["mapping"],
        "official_binary_support_subset": {
            "modules": [0, 2, 3], "records": 30,
            "support_calls_per_mode": 120, "aggregates": exact,
        },
        "d1_scaled_binary_opportunity_diagnostic": {
            "module": 1, "records": 10, "support_calls_per_mode": 40,
            "folded_weight_deployment_admitted": False,
            "aggregates": diagnostic,
        },
        "exact_decoder_complete": {
            "admitted": False,
            "total_cycles": None,
            "product_vs_bit_speedup": None,
            "blocking_reason": "D1 theta-folded FP32 execution is non-bit-exact and the original-weight output-scale candidate is unmetered/not admitted.",
        },
        "records": sorted(
            exact_rows + diagnostic_rows,
            key=lambda row: (row["sample_id"], row["module_index"]),
        ),
        "claim_boundary": contract["claim_boundary"],
    }
    report["payload_sha256"] = canonical_sha(report)
    return report


def acquire_single_writer_lock(lock_path: Path) -> dict[str, Any]:
    lock_path = Path(lock_path)
    parent = real_directory(lock_path.parent.resolve(), "M700 lock parent")
    require(lock_path.parent.resolve() == parent and
            lock_path.name == SINGLE_WRITER_LOCK.name,
            "M700 single-writer lock path drift")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(lock_path), flags, 0o600)
    except FileExistsError as error:
        raise RuntimeError("M700 single-writer lock already exists") from error
    observed = os.fstat(descriptor)
    require(stat.S_ISREG(observed.st_mode),
            "M700 single-writer lock is not a regular file")
    os.write(descriptor, (
        json.dumps({
            "schema": "m700_single_writer_lock_v1",
            "pid": os.getpid(), "workers": 3,
        }, sort_keys=True) + "\n"
    ).encode("utf-8"))
    os.fsync(descriptor)
    return {
        "descriptor": descriptor, "path": str(lock_path),
        "device": observed.st_dev, "inode": observed.st_ino,
    }


def validate_single_writer_lock(lock: dict[str, Any]) -> None:
    descriptor = lock["descriptor"]
    path = Path(lock["path"])
    by_descriptor = os.fstat(descriptor)
    by_path = os.lstat(path)
    require(stat.S_ISREG(by_path.st_mode) and not stat.S_ISLNK(by_path.st_mode) and
            (by_descriptor.st_dev, by_descriptor.st_ino) ==
            (lock["device"], lock["inode"]) ==
            (by_path.st_dev, by_path.st_ino),
            "M700 single-writer lock identity drift")


def release_single_writer_lock(lock: dict[str, Any]) -> None:
    validate_single_writer_lock(lock)
    path = Path(lock["path"])
    os.unlink(path)
    os.close(lock["descriptor"])


def write_failure_receipt(
    directory: Path, error: BaseException, stage: str
) -> dict[str, str]:
    directory = real_directory(Path(directory).resolve(),
                               "M700 failure receipt")
    require(not (directory / "SHA256SUMS").exists() and
            not (directory / "SHA256SUMS.seal.sha256").exists(),
            "M700 failure receipt directory is already sealed")
    (directory / "FAILED.json").write_text(json.dumps({
        "schema": "m700_failure_receipt_v2",
        "status": "FAIL_CLOSED_NO_CANONICAL_OUTPUT_ADMITTED",
        "stage": stage,
        "error_type": type(error).__name__,
        "error": str(error),
        "workers": 3,
        "canonical_output_admitted": False,
        "cycles_admitted": False,
        "speedup_admitted": False,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (directory / "RUN_FAILED.txt").write_text(
        "FAIL_CLOSED_M700__NO_CANONICAL_OUTPUT_ADMITTED\n",
        encoding="utf-8",
    )
    write_double_seal(directory)
    return verify_double_seal(directory)


def run_with_failure_receipt(
    run_state: Path, stage: dict[str, str], operation: Any
) -> Any:
    try:
        return operation()
    except BaseException as error:
        write_failure_receipt(run_state, error, stage["name"])
        raise


def atomic_publish(
    output: Path, report: dict[str, Any], receipt: dict[str, Any],
    single_writer_lock: dict[str, Any],
) -> None:
    validate_single_writer_lock(single_writer_lock)
    output = Path(output)
    parent = real_directory(output.parent.resolve(), "M700 output parent")
    require(output.parent.resolve() == parent,
            "M700 output parent drifted")
    output = parent / output.name
    require(not os.path.lexists(output),
            "M700 output exists or parent drifted")
    staging = Path(tempfile.mkdtemp(prefix=output.name + ".staging.", dir=parent))
    try:
        (staging / "m700_result.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging / "m700_receipt.json").write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M700_EXTERNAL_OFFICIAL_DECODER_SUPPORT_SUBSET\n",
            encoding="utf-8",
        )
        write_double_seal(staging)
        verify_double_seal(staging)
        validate_single_writer_lock(single_writer_lock)
        require(not os.path.lexists(output),
                "M700 canonical output appeared before locked publication")
        os.rename(staging, output)
        try:
            verify_double_seal(output)
        except BaseException:
            quarantine = output.with_name(
                output.name + ".quarantine.post_publish_verification_failed"
            )
            require(not quarantine.exists(),
                    "M700 post-publish quarantine already exists")
            os.rename(output, quarantine)
            raise
    except BaseException as error:
        if staging.exists():
            for name in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
                path = staging / name
                if path.exists() and path.is_file() and not path.is_symlink():
                    path.unlink()
            (staging / "FAILED.json").write_text(json.dumps({
                "schema": "m700_atomic_publish_failure_receipt_v2",
                "status": "FAIL_CLOSED_NO_CANONICAL_OUTPUT",
                "error_type": type(error).__name__, "error": str(error),
            }, indent=2) + "\n", encoding="utf-8")
            write_double_seal(staging)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--static-review-directory", type=Path, required=True)
    parser.add_argument("--allow-full-official-replay", action="store_true")
    args = parser.parse_args()
    run_state_parent = real_directory(
        DEFAULT_OUT.parent.resolve(), "M700 run-state parent"
    )
    run_state = Path(tempfile.mkdtemp(
        prefix=DEFAULT_OUT.name + ".failure.", dir=run_state_parent
    ))
    stage = {"name": "ARGUMENT_AND_AUTHORIZATION_PREFLIGHT"}

    def operation() -> int:
        lock = acquire_single_writer_lock(SINGLE_WRITER_LOCK)
        try:
            require(args.contract.resolve() == CONTRACT.resolve(),
                    "M700 accepts only its frozen contract")
            require(args.output.resolve() == DEFAULT_OUT.resolve(),
                    "M700 accepts only its canonical output")
            require(not os.path.lexists(DEFAULT_OUT),
                    "M700 canonical output already exists before preflight")
            require(args.workers == 3,
                    "M700 official replay workers are frozen at exactly 3")
            require(args.allow_full_official_replay,
                    "M700 full official replay requires explicit CLI authorization")
            expected_outer = os.environ.get(
                "M700_EXPECTED_STATIC_REVIEW_OUTER_SEAL_FILE_SHA256", ""
            )
            require(re.fullmatch(r"[0-9a-f]{64}", expected_outer) is not None,
                    "M700 static-review outer-seal SHA environment is missing/malformed")
            authorization = validate_execution_authorization(
                args.static_review_directory, expected_outer
            )
            contract = strict_json(CONTRACT)
            identity, exact_records, diagnostic_records = preflight(contract)
            require(len(exact_records) == 30 and len(diagnostic_records) == 10,
                    "M700 execution population mismatch")

            stage["name"] = "EXECUTE_EXACT_D0_D2_D3_POPULATION"
            exact_results = execute_records(exact_records, workers=3)
            stage["name"] = "EXECUTE_D1_DIAGNOSTIC_POPULATION"
            diagnostic_results = execute_records(diagnostic_records, workers=3)

            stage["name"] = "POST_EXECUTION_IDENTITY_RECHECK"
            authorization_after = validate_execution_authorization(
                args.static_review_directory, expected_outer
            )
            require(authorization_after == authorization,
                    "M700 static execution authorization drifted during replay")
            identity_after, exact_after, diagnostic_after = preflight(contract)
            require(identity_after == identity and exact_after == exact_records and
                    diagnostic_after == diagnostic_records,
                    "M700 frozen input drift across official execution")
            report = build_report(
                contract, identity, exact_results, diagnostic_results,
                authorization, workers=3,
            )
            receipt = {
                "schema": "m700_h67_decoder_official_prosperity_receipt_v2",
                "status": report["status"],
                "result_payload_sha256": report["payload_sha256"],
                "execution": {
                    "workers": 3,
                    "worker_policy": "FIXED_THREE_PROCESS_CPU_REPLAY",
                    "single_writer_lock": SINGLE_WRITER_LOCK.name,
                },
                "exact_decoder_complete_cycles": None,
                "exact_decoder_complete_speedup": None,
                "claim_boundary": contract["claim_boundary"],
            }
            stage["name"] = "ATOMIC_PUBLICATION_AND_POST_VERIFY"
            atomic_publish(args.output, report, receipt, lock)
            stage["name"] = "COMPLETE"
            return 0
        finally:
            release_single_writer_lock(lock)

    result = run_with_failure_receipt(run_state, stage, operation)
    require(stage["name"] == "COMPLETE",
            "M700 successful operation did not reach COMPLETE")
    run_state.rmdir()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
