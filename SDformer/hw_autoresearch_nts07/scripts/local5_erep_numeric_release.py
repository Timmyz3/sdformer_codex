#!/usr/bin/env python3
"""封存并只读复核 Local5 numeric shard 共用 RTL release。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any


HEADS = (3, 6, 12, 24)
SCHEMA = "local5_erep_numeric_rtl_release_v2"
COMPLETE_SCHEMA = "local5_erep_numeric_rtl_release_complete_v2"
TOOL_SCHEMA = "local5_erep_numeric_tool_bindings_v1"
REQUIRED_TOOLS = {
    "bash",
    "c++",
    "flock",
    "g++",
    "make",
    "python3",
    "sha256sum",
    "tar",
    "time",
    "verilator",
    "verilator_bin",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _normalize_member(name: str) -> str:
    member = PurePosixPath(name)
    if (
        not name
        or member.is_absolute()
        or ".." in member.parts
        or str(member) != name
    ):
        raise ValueError(f"source path 非规范相对路径: {name!r}")
    return name


def _read_source_bindings(path: Path) -> list[dict[str, str]]:
    bindings: list[dict[str, str]] = []
    observed: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        digest, separator, name = line.partition("  ")
        name = _normalize_member(name)
        if (
            not separator
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
            or name in observed
        ):
            raise ValueError(f"source_sha256 第{line_number}行不合法")
        observed.add(name)
        bindings.append({"path": name, "sha256": digest})
    if not bindings:
        raise ValueError("source_sha256 为空")
    return bindings


def _verify_source_tree(root: Path, bindings: list[dict[str, str]]) -> None:
    source_root = root / "source"
    expected = {binding["path"]: binding["sha256"] for binding in bindings}
    observed: dict[str, str] = {}
    if not source_root.is_dir():
        raise ValueError("release source tree 缺失")
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("release source tree 不允许 symlink")
        if path.is_file():
            name = path.relative_to(source_root).as_posix()
            observed[name] = sha256(path)
    if observed != expected:
        raise ValueError("release source tree 与 source_sha256 不一致")


def _verify_source_bundle(path: Path, bindings: list[dict[str, str]]) -> None:
    expected = {binding["path"]: binding["sha256"] for binding in bindings}
    observed: dict[str, str] = {}
    try:
        with tarfile.open(path, mode="r:") as archive:
            for member in archive.getmembers():
                name = _normalize_member(member.name)
                if not member.isfile() or name in observed:
                    raise ValueError("source bundle 仅允许唯一 regular file member")
                handle = archive.extractfile(member)
                if handle is None:
                    raise ValueError("source bundle member 无法读取")
                observed[name] = _bytes_sha256(handle.read())
    except tarfile.TarError as error:
        raise ValueError("source bundle 不是合法 tar") from error
    if observed != expected:
        raise ValueError("source bundle member/content 与 source_sha256 不一致")


def _read_tool_bindings(path: Path) -> dict[str, dict[str, str]]:
    value = _read_json(path)
    rows = value.get("tools") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema") != TOOL_SCHEMA
        or not isinstance(rows, list)
    ):
        raise ValueError("tool bindings schema 不合法")
    tools: dict[str, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("tool binding row 不是 object")
        name = row.get("name")
        tool_path = Path(str(row.get("path", "")))
        digest = row.get("sha256")
        version = row.get("version")
        if (
            not isinstance(name, str)
            or name in tools
            or not tool_path.is_absolute()
            or not tool_path.is_file()
            or not isinstance(digest, str)
            or digest != sha256(tool_path)
            or not isinstance(version, str)
            or not version.strip()
        ):
            raise ValueError(f"tool binding 失配: {name}")
        tools[name] = {
            "name": name,
            "path": str(tool_path),
            "sha256": digest,
            "version": version,
        }
    if set(tools) != REQUIRED_TOOLS:
        raise ValueError("tool binding 集合不精确")
    return tools


def _validate_compile_argv(
    argv: Any,
    heads: int,
    tools: dict[str, dict[str, str]],
    bindings: list[dict[str, str]],
) -> list[str]:
    if not isinstance(argv, list) or not argv or any(
        not isinstance(value, str) or not value for value in argv
    ):
        raise ValueError(f"H{heads} compile argv 不是非空字符串数组")
    required = {
        "--binary",
        "--timing",
        "--assert",
        "--Mdir",
        "source/tb_qfit/tb_qfit_local5_memo_multitile_cross_head.sv",
        "-GUSE_MEMO=0",
        "-GUSE_INPLACE=0",
        f"-GHEADS={heads}",
        f"-GOUTPUT_TILES={heads}",
        "-GTIMEOUT_CYCLES=100000000",
    }
    if argv[0] != tools["verilator"]["path"] or not required.issubset(set(argv)):
        raise ValueError(f"H{heads} compile argv 合同不完整")
    transaction_mode = "-GTRANSACTION_INDEXED_SERVICE=1" in argv
    identity_mode = "-GIDENTITY_DERIVED_SERVICE=1" in argv
    if transaction_mode == identity_mode:
        raise ValueError(f"H{heads} compile service mode 必须且只能选择一种")
    if identity_mode and "-GTRANSACTION_INDEXED_SERVICE=0" not in argv:
        raise ValueError(f"H{heads} identity compile 未显式关闭 transaction mode")
    for option, expected in (
        ("--top-module", "tb_qfit_local5_memo_multitile_cross_head"),
        ("--Mdir", f"build/h{heads}/obj"),
    ):
        try:
            index = argv.index(option)
        except ValueError as error:
            raise ValueError(f"H{heads} compile argv 缺少 {option}") from error
        if index + 1 >= len(argv) or argv[index + 1] != expected:
            raise ValueError(f"H{heads} compile {option} 不合法")
    expected_sources = {
        binding["path"] for binding in bindings if binding["path"].endswith(".sv")
    }
    observed_sources = {
        value.removeprefix("source/")
        for value in argv
        if value.startswith("source/") and value.endswith(".sv")
    }
    if observed_sources != expected_sources:
        raise ValueError(f"H{heads} compile source 集合不精确")
    if any(
        Path(value).is_absolute()
        for value in argv[1:]
        if value.endswith(".sv") or value.startswith("build/")
    ):
        raise ValueError(f"H{heads} compile release-local path 不能为绝对路径")
    return argv


def _build_binding(
    root: Path,
    heads: int,
    tools: dict[str, dict[str, str]],
    sources: list[dict[str, str]],
) -> dict[str, Any]:
    build = root / "build" / f"h{heads}"
    argv_path = build / "compile_argv.json"
    log_path = build / "compile.log"
    executable = build / "obj" / "Vtb_qfit_local5_memo_multitile_cross_head"
    argv = _validate_compile_argv(_read_json(argv_path), heads, tools, sources)
    if not log_path.is_file() or not executable.is_file():
        raise ValueError(f"H{heads} build artifact 缺失")
    return {
        "heads": heads,
        "service_mode": (
            "identity_derived" if "-GIDENTITY_DERIVED_SERVICE=1" in argv
            else "transaction_indexed"
        ),
        "compile_cwd": ".",
        "compile_argv": argv,
        "compile_argv_path": str(argv_path.relative_to(root)),
        "compile_argv_sha256": sha256(argv_path),
        "compile_log_path": str(log_path.relative_to(root)),
        "compile_log_sha256": sha256(log_path),
        "executable_path": str(executable.relative_to(root)),
        "executable_sha256": sha256(executable),
    }


def seal_release(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "release_manifest.json"
    complete_path = root / "release_complete.json"
    if manifest_path.exists() or complete_path.exists():
        raise ValueError("release 已存在；seal 不允许覆盖")
    source_sha = root / "source_sha256.txt"
    source_bundle = root / "source_bundle.tar"
    tool_versions = root / "tool_versions.txt"
    tool_bindings_path = root / "tool_bindings.json"
    for artifact in (source_sha, source_bundle, tool_versions, tool_bindings_path):
        if not artifact.is_file():
            raise ValueError(f"release artifact 缺失: {artifact.name}")
    sources = _read_source_bindings(source_sha)
    _verify_source_bundle(source_bundle, sources)
    _verify_source_tree(root, sources)
    tools = _read_tool_bindings(tool_bindings_path)
    manifest = {
        "schema": SCHEMA,
        "status": "SEALED_RTL_RELEASE_NOT_G0",
        "evidence": "[rtl-build-provenance]",
        "formal_g0": "DENY",
        "source_sha256_path": "source_sha256.txt",
        "source_sha256_sha256": sha256(source_sha),
        "source_bundle_path": "source_bundle.tar",
        "source_bundle_sha256": sha256(source_bundle),
        "source_tree_path": "source",
        "tool_versions_path": "tool_versions.txt",
        "tool_versions_sha256": sha256(tool_versions),
        "tool_bindings_path": "tool_bindings.json",
        "tool_bindings_sha256": sha256(tool_bindings_path),
        "source_bindings": sources,
        "tool_bindings": tools,
        "builds": {
            str(heads): _build_binding(root, heads, tools, sources)
            for heads in HEADS
        },
        "publication_contract": {
            "external_lock": "flock on <release-dir>.lock",
            "staged_build": True,
            "atomic_publish": "rename staging directory to release directory",
        },
        "boundary": [
            "该 release 仅冻结 numeric shard 的 RTL/TB/adapter/tool/executable",
            "源码从封存 source tree 编译；验证不读取当前工作树",
            "不包含 workload vectors、数值 miter、phase ledger 或 formal admission",
        ],
    }
    _write_atomic(manifest_path, manifest)
    complete = {
        "schema": COMPLETE_SCHEMA,
        "status": "PASS_RELEASE_SEALED_NOT_G0",
        "formal_g0": "DENY",
        "release_manifest_sha256": sha256(manifest_path),
    }
    _write_atomic(complete_path, complete)
    return verify_release(root)


def verify_release(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "release_manifest.json"
    complete_path = root / "release_complete.json"
    manifest = _read_json(manifest_path)
    complete = _read_json(complete_path)
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != SCHEMA
        or manifest.get("status") != "SEALED_RTL_RELEASE_NOT_G0"
        or manifest.get("formal_g0") != "DENY"
        or not isinstance(complete, dict)
        or complete.get("schema") != COMPLETE_SCHEMA
        or complete.get("status") != "PASS_RELEASE_SEALED_NOT_G0"
        or complete.get("formal_g0") != "DENY"
        or complete.get("release_manifest_sha256") != sha256(manifest_path)
    ):
        raise ValueError("release seal/schema 失效")
    fixed_artifacts = (
        ("source_sha256_path", "source_sha256_sha256"),
        ("source_bundle_path", "source_bundle_sha256"),
        ("tool_versions_path", "tool_versions_sha256"),
        ("tool_bindings_path", "tool_bindings_sha256"),
    )
    for path_key, sha_key in fixed_artifacts:
        artifact = root / str(manifest.get(path_key, ""))
        if not artifact.is_file() or manifest.get(sha_key) != sha256(artifact):
            raise ValueError(f"release artifact 失配: {path_key}")
    sources = _read_source_bindings(root / str(manifest["source_sha256_path"]))
    if sources != manifest.get("source_bindings"):
        raise ValueError("release source bindings 与 manifest 不一致")
    _verify_source_bundle(root / str(manifest["source_bundle_path"]), sources)
    _verify_source_tree(root, sources)
    tools = _read_tool_bindings(root / str(manifest["tool_bindings_path"]))
    if tools != manifest.get("tool_bindings"):
        raise ValueError("release tool bindings 与 manifest 不一致")
    builds = manifest.get("builds")
    if not isinstance(builds, dict) or set(builds) != {str(h) for h in HEADS}:
        raise ValueError("release build 集合不精确")
    for heads in HEADS:
        expected = builds[str(heads)]
        if not isinstance(expected, dict):
            raise ValueError(f"H{heads} build binding 非 object")
        observed = _build_binding(root, heads, tools, sources)
        if observed != expected:
            raise ValueError(f"H{heads} build binding 失配")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("seal", "verify"))
    parser.add_argument("--release-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = (
        seal_release(args.release_dir)
        if args.mode == "seal"
        else verify_release(args.release_dir)
    )
    print(
        json.dumps(
            {
                "status": "PASS_RELEASE_SEALED_NOT_G0",
                "formal_g0": "DENY",
                "release_manifest_sha256": sha256(
                    args.release_dir.resolve() / "release_manifest.json"
                ),
                "build_heads": sorted(int(key) for key in manifest["builds"]),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
