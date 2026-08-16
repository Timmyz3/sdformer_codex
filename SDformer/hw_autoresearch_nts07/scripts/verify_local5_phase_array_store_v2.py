#!/usr/bin/env python3
"""独立 mmap 验证 Local5 Phase Array Store v2，并与旧 NPZ 做逐数组等价比较。"""

from __future__ import annotations

import argparse
import hashlib
import json
import mmap
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any, Iterator

import numpy as np


FIELDS = [
    "cycle", "event", "tile", "head", "source", "lane", "out", "delay",
    "index", "origin", "payload",
]
CLASS_ORDER = (
    "prefix", "head_seed", "inter_head_gap", "head_accumulate",
    "tile_tail", "tile_transition", "suffix",
)
ARRAY_DTYPES: dict[str, np.dtype[Any]] = {
    "schema_version": np.dtype("uint16"),
    "identity_sample": np.dtype("uint32"),
    "identity_stage": np.dtype("uint32"),
    "identity_block": np.dtype("uint32"),
    "identity_window": np.dtype("uint32"),
    "heads": np.dtype("uint16"),
    "source_trace_sha256": np.dtype("S64"),
    "class_name": np.dtype("S32"),
    "event_dictionary": np.dtype("S40"),
    "origin_dictionary": np.dtype("S64"),
    "payload_dictionary": np.dtype("S64"),
    "template_offsets": np.dtype("int64"),
    "template_event_code": np.dtype("uint8"),
    "template_origin_code": np.dtype("uint8"),
    "instance_class_code": np.dtype("uint8"),
    "instance_tile": np.dtype("int16"),
    "instance_head": np.dtype("int16"),
    "patch_offsets": np.dtype("int64"),
    "patch_cycle": np.dtype("uint32"),
    "patch_tile": np.dtype("int16"),
    "patch_head": np.dtype("int16"),
    "patch_source": np.dtype("int16"),
    "patch_lane": np.dtype("int16"),
    "patch_out": np.dtype("int16"),
    "patch_delay": np.dtype("int16"),
    "patch_index": np.dtype("int32"),
    "patch_payload_code": np.dtype("uint32"),
}
IDENTITY_NAMES = ("sample", "stage", "block", "window")
LEGACY_ARRAY_NAMES = tuple(
    name for name in ARRAY_DTYPES if not name.startswith("identity_")
)
PAGE_DROP_ROWS = 1 << 18
EXPECTED_GENERATOR_PAGE_DROP_ROWS = 1 << 20
NUMERIC_ARRAYS = {
    "cycle": "patch_cycle",
    "tile": "patch_tile",
    "head": "patch_head",
    "source": "patch_source",
    "lane": "patch_lane",
    "out": "patch_out",
    "delay": "patch_delay",
    "index": "patch_index",
}


def trace_rows(path: Path) -> Iterator[tuple[str, ...]]:
    with path.open("r", encoding="ascii", newline="") as handle:
        header = handle.readline()
        if header.rstrip("\r\n") != ",".join(FIELDS):
            raise ValueError("source trace header 不一致")
        for line in handle:
            raw = line.rstrip("\r\n")
            if not raw or '"' in raw or "\r" in raw or "\n" in raw:
                raise ValueError("source trace 行违反无引号 ASCII 合同")
            row = tuple(raw.split(","))
            if len(row) != len(FIELDS):
                raise ValueError("source trace 行列数不一致")
            yield row


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_bytes(array: np.ndarray, name: str) -> list[str]:
    result: list[str] = []
    for index, value in enumerate(array):
        raw = bytes(value).rstrip(b"\x00")
        if not raw or b"\x00" in raw:
            raise ValueError(f"{name}[{index}] 为空或含嵌入 NUL")
        try:
            result.append(raw.decode("ascii"))
        except UnicodeDecodeError as error:
            raise ValueError(f"{name}[{index}] 不是 ASCII") from error
    if len(result) != len(set(result)):
        raise ValueError(f"{name} 字典不唯一")
    return result


def safe_member(root: Path, relative: str) -> Path:
    member = PurePosixPath(relative)
    if member.is_absolute() or ".." in member.parts:
        raise ValueError("manifest 成员路径越界")
    path = root.joinpath(*member.parts)
    if not path.is_file():
        raise ValueError(f"manifest 成员不存在: {relative}")
    return path


def validate_identity(value: Any, heads: int) -> dict[str, int]:
    names = {"sample", "stage", "block", "window", "heads"}
    if not isinstance(value, dict) or set(value) != names:
        raise ValueError("identity 字段集合不一致")
    if any(
        isinstance(value[name], bool)
        or not isinstance(value[name], int)
        or value[name] < 0
        or value[name] > np.iinfo(np.uint32).max
        for name in names
    ) or value["heads"] != heads:
        raise ValueError("identity 类型、范围或 heads 不一致")
    return value


def load_frozen_identity(
    manifest_path: Path, receipt_path: Path
) -> tuple[dict[str, int], dict[str, str]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    identity_value = manifest.get("identity")
    if not isinstance(identity_value, dict) or not isinstance(identity_value.get("heads"), int):
        raise ValueError("frozen identity manifest lacks heads")
    identity = validate_identity(identity_value, identity_value["heads"])
    manifest_sha = sha256(manifest_path)
    receipt_sha = sha256(receipt_path)
    if (
        manifest.get("schema") != "local5_identity_service_tables_v4"
        or manifest.get("status") != "PASS_IDENTITY_SERVICE_TABLES_NOT_G0"
        or manifest.get("formal_g0") != "DENY"
        or receipt.get("schema") != "local5_identity_service_tables_independent_verify_v1"
        or receipt.get("status") != "PASS_INDEPENDENT_VERIFY_NOT_G0"
        or receipt.get("formal_g0") != "DENY"
        or receipt.get("manifest_sha256") != manifest_sha
    ):
        raise ValueError("frozen identity manifest/receipt contract differs")
    return identity, {
        "manifest_binding": manifest_sha,
        "receipt_binding": receipt_sha,
    }


def load_store(root: Path, trace: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "local5_phase_array_store_v2"
        or manifest.get("status") != "GENERATED_PENDING_INDEPENDENT_VERIFY_NOT_G0"
        or manifest.get("evidence") != "[rtl-trace-derived]"
        or manifest.get("formal_g0") != "DENY"
    ):
        raise ValueError("array store 顶层证据口径不一致")
    entries = manifest.get("arrays")
    if not isinstance(entries, dict) or set(entries) != set(ARRAY_DTYPES):
        raise ValueError("manifest array 成员集合不一致")
    observed_files = {
        path.name for path in (root / "arrays").iterdir() if path.is_file()
    }
    if observed_files != {f"{name}.npy" for name in ARRAY_DTYPES}:
        raise ValueError("arrays 目录存在缺失或额外文件")

    arrays: dict[str, np.ndarray] = {}
    total_nbytes = 0
    total_file_bytes = 0
    for name, dtype in ARRAY_DTYPES.items():
        entry = entries[name]
        if not isinstance(entry, dict) or set(entry) != {
            "file", "dtype", "shape", "nbytes", "file_bytes", "sha256"
        }:
            raise ValueError(f"manifest {name} 元数据字段不一致")
        expected_file = f"arrays/{name}.npy"
        if entry["file"] != expected_file:
            raise ValueError(f"manifest {name} 文件名不一致")
        path = safe_member(root, expected_file)
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if value.dtype != dtype or value.ndim != 1:
            raise ValueError(f"array {name} dtype/rank 不一致")
        actual = {
            "dtype": dtype.str,
            "shape": list(value.shape),
            "nbytes": int(value.nbytes),
            "file_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        if any(entry[key] != actual[key] for key in actual):
            raise ValueError(f"manifest {name} 绑定不一致")
        arrays[name] = value
        total_nbytes += int(value.nbytes)
        total_file_bytes += path.stat().st_size
    if (
        manifest.get("array_nbytes_total") != total_nbytes
        or manifest.get("array_file_bytes_total") != total_file_bytes
        or manifest.get("mmap_page_drop_rows") != EXPECTED_GENERATOR_PAGE_DROP_ROWS
        or manifest.get("source_trace_sha256") != sha256(trace)
        or manifest.get("source_trace_file_bytes") != trace.stat().st_size
    ):
        raise ValueError("manifest 总容量或 source trace 绑定不一致")

    bindings = manifest.get("source_bindings")
    if not isinstance(bindings, dict) or set(bindings) != {
        "generator", "independent_verifier"
    }:
        raise ValueError("source bindings 字段不一致")
    observed_source_files: set[str] = set()
    for name, entry in bindings.items():
        if not isinstance(entry, dict) or set(entry) != {"file", "sha256"}:
            raise ValueError(f"{name} source binding 格式不一致")
        path = safe_member(root, entry["file"])
        observed_source_files.add(path.name)
        if sha256(path) != entry["sha256"]:
            raise ValueError(f"{name} source binding SHA 不一致")
    actual_source_files = {
        path.name for path in (root / "source").iterdir() if path.is_file()
    }
    if actual_source_files != observed_source_files:
        raise ValueError("source 目录存在缺失或额外文件")
    if bindings["independent_verifier"]["sha256"] != sha256(Path(__file__).resolve()):
        raise ValueError("当前独立验证器与 store 中封存版本不一致")
    return manifest, arrays


def validate_offsets(offsets: np.ndarray, expected_last: int, name: str) -> None:
    if (
        len(offsets) < 2
        or int(offsets[0]) != 0
        or int(offsets[-1]) != expected_last
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(f"{name} offsets 不一致")


def drop_mmap_pages(arrays: dict[str, np.ndarray]) -> int:
    if not hasattr(mmap, "MADV_DONTNEED"):
        raise RuntimeError("platform lacks required mmap MADV_DONTNEED support")
    released = 0
    seen: set[int] = set()
    for value in arrays.values():
        raw = getattr(value, "_mmap", None)
        if raw is None or id(raw) in seen:
            continue
        if not hasattr(raw, "madvise"):
            raise RuntimeError("numpy memmap lacks madvise support")
        raw.madvise(mmap.MADV_DONTNEED)
        seen.add(id(raw))
        released += 1
    return released


def expected_instances(heads: int) -> list[tuple[int, int, int]]:
    code = {name: index for index, name in enumerate(CLASS_ORDER)}
    result = [(code["prefix"], -1, -1)]
    for tile in range(heads):
        for head in range(heads):
            result.append((
                code["head_seed" if head == 0 else "head_accumulate"], tile, head
            ))
            if head + 1 < heads:
                result.append((code["inter_head_gap"], tile, head))
            else:
                result.append((code["tile_tail"], tile, head))
        if tile + 1 < heads:
            result.append((code["tile_transition"], tile + 1, -1))
    result.append((code["suffix"], -1, -1))
    return result


def validate_semantics(
    arrays: dict[str, np.ndarray], manifest: dict[str, Any]
) -> dict[str, Any]:
    if arrays["schema_version"].tolist() != [2]:
        raise ValueError("array store schema_version 不为 2")
    if arrays["heads"].shape != (1,):
        raise ValueError("heads 数组形状不一致")
    heads = int(arrays["heads"][0])
    if not 2 <= heads <= 32:
        raise ValueError("heads 不在 2..32")
    identity = validate_identity(manifest.get("identity"), heads)
    for name in IDENTITY_NAMES:
        value = arrays[f"identity_{name}"]
        if value.shape != (1,) or int(value[0]) != identity[name]:
            raise ValueError(f"identity manifest/array {name} 不一致")
    source_sha = decode_bytes(arrays["source_trace_sha256"], "source_trace_sha256")
    if source_sha != [manifest["source_trace_sha256"]]:
        raise ValueError("array 内 source trace SHA 不一致")
    if decode_bytes(arrays["class_name"], "class_name") != list(CLASS_ORDER):
        raise ValueError("class order 不一致")
    events = decode_bytes(arrays["event_dictionary"], "event_dictionary")
    origins = decode_bytes(arrays["origin_dictionary"], "origin_dictionary")
    payloads = decode_bytes(arrays["payload_dictionary"], "payload_dictionary")
    if events != sorted(events) or origins != sorted(origins) or payloads != sorted(payloads):
        raise ValueError("字典没有按冻结顺序排序")

    template_offsets = arrays["template_offsets"]
    patch_offsets = arrays["patch_offsets"]
    template_count = len(arrays["template_event_code"])
    patch_count = len(arrays["patch_cycle"])
    instance_count = len(arrays["instance_class_code"])
    if len(template_offsets) != len(CLASS_ORDER) + 1:
        raise ValueError("template offsets 数量不一致")
    validate_offsets(template_offsets, template_count, "template")
    validate_offsets(patch_offsets, patch_count, "patch")
    if len(patch_offsets) != instance_count + 1:
        raise ValueError("patch offsets 与 instance 数量不一致")
    if len(arrays["template_origin_code"]) != template_count:
        raise ValueError("template event/origin 长度不一致")
    for name in ("instance_tile", "instance_head"):
        if len(arrays[name]) != instance_count:
            raise ValueError(f"{name} 长度不一致")
    for name in (*NUMERIC_ARRAYS.values(), "patch_payload_code"):
        if len(arrays[name]) != patch_count:
            raise ValueError(f"{name} 长度不一致")
    if (
        (template_count and int(np.max(arrays["template_event_code"])) >= len(events))
        or (template_count and int(np.max(arrays["template_origin_code"])) >= len(origins))
        or (patch_count and int(np.max(arrays["patch_payload_code"])) >= len(payloads))
    ):
        raise ValueError("字典 code 越界")

    observed_instances = [
        (
            int(arrays["instance_class_code"][index]),
            int(arrays["instance_tile"][index]),
            int(arrays["instance_head"][index]),
        )
        for index in range(instance_count)
    ]
    if observed_instances != expected_instances(heads):
        raise ValueError("instance class/tile/head 序列不一致")

    class_stats: dict[str, dict[str, int]] = {
        name: {
            "instances": 0,
            "template_rows": int(template_offsets[index + 1] - template_offsets[index]),
            "expanded_rows": 0,
        }
        for index, name in enumerate(CLASS_ORDER)
    }
    for instance, (class_code, _, _) in enumerate(observed_instances):
        if not 0 <= class_code < len(CLASS_ORDER):
            raise ValueError("instance class code 越界")
        template_length = int(
            template_offsets[class_code + 1] - template_offsets[class_code]
        )
        patch_length = int(patch_offsets[instance + 1] - patch_offsets[instance])
        if patch_length != template_length:
            raise ValueError("instance patch 长度与 template 不一致")
        stat = class_stats[CLASS_ORDER[class_code]]
        stat["instances"] += 1
        stat["expanded_rows"] += patch_length
    derived = {
        "heads": heads,
        "identity": identity,
        "store_arrays": len(arrays),
        "expanded_rows": patch_count,
        "template_rows": template_count,
        "instances": instance_count,
        "class_stats": class_stats,
        "payload_dictionary_entries": len(payloads),
        "base_event_reuse_factor": patch_count / template_count,
    }
    for key in (
        "expanded_rows", "template_rows", "instances", "class_stats",
        "payload_dictionary_entries",
    ):
        if manifest.get(key) != derived[key]:
            raise ValueError(f"manifest 派生字段 {key} 不一致")
    if abs(float(manifest.get("base_event_reuse_factor")) - derived["base_event_reuse_factor"]) > 1e-12:
        raise ValueError("manifest base_event_reuse_factor 不一致")
    return derived


def expanded_rows(
    arrays: dict[str, np.ndarray], page_drop: dict[str, int]
) -> Iterator[tuple[str, ...]]:
    events = decode_bytes(arrays["event_dictionary"], "event_dictionary")
    origins = decode_bytes(arrays["origin_dictionary"], "origin_dictionary")
    payloads = decode_bytes(arrays["payload_dictionary"], "payload_dictionary")
    template_offsets = arrays["template_offsets"]
    patch_offsets = arrays["patch_offsets"]
    emitted = 0
    for instance, class_value in enumerate(arrays["instance_class_code"]):
        class_code = int(class_value)
        template_start = int(template_offsets[class_code])
        patch_start = int(patch_offsets[instance])
        patch_stop = int(patch_offsets[instance + 1])
        for relative, patch_index in enumerate(range(patch_start, patch_stop)):
            template_index = template_start + relative
            yield (
                str(int(arrays["patch_cycle"][patch_index])),
                events[int(arrays["template_event_code"][template_index])],
                str(int(arrays["patch_tile"][patch_index])),
                str(int(arrays["patch_head"][patch_index])),
                str(int(arrays["patch_source"][patch_index])),
                str(int(arrays["patch_lane"][patch_index])),
                str(int(arrays["patch_out"][patch_index])),
                str(int(arrays["patch_delay"][patch_index])),
                str(int(arrays["patch_index"][patch_index])),
                origins[int(arrays["template_origin_code"][template_index])],
                payloads[int(arrays["patch_payload_code"][patch_index])],
            )
            emitted += 1
            if emitted % PAGE_DROP_ROWS == 0:
                page_drop["calls"] += 1
                page_drop["mappings"] += drop_mmap_pages(arrays)


def verify_expansion(
    trace: Path, arrays: dict[str, np.ndarray], manifest: dict[str, Any],
    frozen_trace_bindings: dict[str, str],
) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update((",".join(FIELDS) + "\n").encode("ascii"))
    count = 0
    event_counts: Counter[str] = Counter()
    observed_trace_bindings: dict[str, str] = {}
    page_drop = {"calls": 1, "mappings": drop_mmap_pages(arrays)}
    generated = expanded_rows(arrays, page_drop)
    for expected in trace_rows(trace):
        try:
            actual = next(generated)
        except StopIteration as error:
            raise ValueError("array store 展开提前结束") from error
        if actual != expected:
            raise ValueError(f"array store 展开在 row {count} 不一致")
        digest.update((",".join(actual) + "\n").encode("ascii"))
        event_counts[actual[1]] += 1
        if actual[1] in frozen_trace_bindings:
            if (
                actual[1] in observed_trace_bindings
                or actual[0] != "0"
                or actual[10] != "-"
            ):
                raise ValueError("frozen identity trace binding duplicated or malformed")
            observed_trace_bindings[actual[1]] = actual[9]
        count += 1
    try:
        next(generated)
    except StopIteration:
        pass
    else:
        raise ValueError("array store 展开存在额外行")
    observed_sha = digest.hexdigest()
    if observed_sha != sha256(trace):
        raise ValueError("展开 CSV 字节流 SHA 与 source trace 不一致")
    counts = dict(sorted(event_counts.items()))
    if counts != manifest.get("event_counts"):
        raise ValueError("manifest event_counts 与独立展开不一致")
    if observed_trace_bindings != frozen_trace_bindings:
        raise ValueError("trace does not bind frozen identity manifest/receipt")
    page_drop["calls"] += 1
    page_drop["mappings"] += drop_mmap_pages(arrays)
    return {
        "rows": count,
        "expanded_trace_sha256": observed_sha,
        "event_counts": counts,
        "frozen_identity_trace_bindings": observed_trace_bindings,
        "mmap_page_drop_rows": PAGE_DROP_ROWS,
        "mmap_page_drop": page_drop,
    }


def compare_legacy(path: Path | None, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    if path is None:
        return {
            "status": "SKIPPED_NOT_PROVIDED_SOURCE_TRACE_IS_PRIMARY",
            "arrays_compared": 0,
            "scalars_compared": 0,
            "mismatch": None,
        }
    compared = 0
    scalars = 0
    with np.load(path, allow_pickle=False) as legacy:
        if set(legacy.files) != set(LEGACY_ARRAY_NAMES):
            raise ValueError("legacy NPZ 成员集合不一致")
        for name in LEGACY_ARRAY_NAMES:
            old = legacy[name]
            current = arrays[name]
            if old.dtype != ARRAY_DTYPES[name] or old.shape != current.shape:
                raise ValueError(f"legacy {name} dtype/shape 不一致")
            if name == "schema_version":
                if old.tolist() != [1] or current.tolist() != [2]:
                    raise ValueError("legacy/new schema version 迁移不一致")
            elif not np.array_equal(old, current):
                raise ValueError(f"legacy/new array {name} 不等价")
            compared += 1
            scalars += int(old.size)
            drop_mmap_pages(arrays)
    return {
        "status": "PASS_LEGACY_ARRAY_EQUIVALENCE",
        "legacy_archive_sha256": sha256(path),
        "arrays_compared": compared,
        "scalars_compared": scalars,
        "only_allowed_difference": "schema_version:uint16[1]=>uint16[2]",
        "mismatch": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--legacy-archive", type=Path)
    parser.add_argument("--expected-identity-manifest", type=Path, required=True)
    parser.add_argument("--expected-identity-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.store_dir.resolve()
    trace = args.source_trace.resolve()
    legacy = args.legacy_archive.resolve() if args.legacy_archive is not None else None
    expected_identity_manifest = args.expected_identity_manifest.resolve()
    expected_identity_receipt = args.expected_identity_receipt.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output exists: {output}")
    manifest, arrays = load_store(root, trace)
    frozen_identity, frozen_trace_bindings = load_frozen_identity(
        expected_identity_manifest, expected_identity_receipt
    )
    derived = validate_semantics(arrays, manifest)
    if derived["identity"] != frozen_identity:
        raise ValueError("store identity differs from frozen expected identity")
    expansion = verify_expansion(trace, arrays, manifest, frozen_trace_bindings)
    legacy_result = compare_legacy(legacy, arrays)
    report = {
        "schema": "local5_phase_array_store_verification_v2",
        "status": (
            "PASS_STREAMING_MMAP_LEGACY_EQUIVALENT_NOT_G0"
            if legacy is not None
            else "PASS_STREAMING_MMAP_SOURCE_TRACE_EQUIVALENT_NOT_G0"
        ),
        "evidence": "[rtl-trace-derived]+[独立软件逐行展开验证]",
        "formal_g0": "DENY",
        "identity": manifest["identity"],
        "frozen_expected_identity": frozen_identity,
        "derived": derived,
        "expansion": expansion,
        "legacy_equivalence": legacy_result,
        "bindings": {
            "store_manifest_sha256": sha256(root / "manifest.json"),
            "source_trace_sha256": sha256(trace),
            "expected_identity_manifest_sha256": sha256(expected_identity_manifest),
            "expected_identity_receipt_sha256": sha256(expected_identity_receipt),
            "verifier_source_sha256": sha256(Path(__file__).resolve()),
        },
        "boundary": [
            (
                "该结果证明验证归档表示与旧 NPZ 以及原始 RTL trace 等价"
                if legacy is not None
                else "该结果只证明验证归档表示与原始 RTL trace 等价；旧 NPZ 未提供"
            ),
            "资源数据仅评价验证基础设施扩展性，不是架构吞吐、片上 SRAM 或 ASIC PPA",
            "单个参数化窗口；其他窗口、formal G0 与 full encoder 尚未通过",
        ],
    }
    if legacy is not None:
        report["bindings"]["legacy_archive_sha256"] = sha256(legacy)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"], "rows": expansion["rows"],
        "legacy_mismatch": legacy_result["mismatch"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
