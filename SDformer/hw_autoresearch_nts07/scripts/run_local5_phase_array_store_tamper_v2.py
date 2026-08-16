#!/usr/bin/env python3
"""对 Phase Array Store v2 做可审计的语义篡改拒绝回归。"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def clone_store(source: Path, destination: Path) -> None:
    subprocess.run(
        ["cp", "-a", "--reflink=auto", str(source), str(destination)],
        check=True,
    )


def invoke(
    verifier: Path, store: Path, trace: Path, legacy: Path | None, output: Path,
    expected_identity_manifest: Path, expected_identity_receipt: Path,
) -> subprocess.CompletedProcess[str]:
    argv = [
        "python3", str(verifier), "--store-dir", str(store),
        "--source-trace", str(trace), "--output", str(output),
        "--expected-identity-manifest", str(expected_identity_manifest),
        "--expected-identity-receipt", str(expected_identity_receipt),
    ]
    if legacy is not None:
        argv.extend(["--legacy-archive", str(legacy)])
    return subprocess.run(
        argv,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def rebind_array(store: Path, name: str) -> None:
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = store / "arrays" / f"{name}.npy"
    manifest["arrays"][name]["sha256"] = sha256(path)
    manifest["arrays"][name]["file_bytes"] = path.stat().st_size
    write_json(manifest_path, manifest)


def snapshot_case(store: Path, evidence: Path, names: list[str]) -> None:
    evidence.mkdir()
    shutil.copy2(store / "manifest.json", evidence / "manifest.json")
    (evidence / "array_listing.txt").write_text(
        "\n".join(sorted(path.name for path in (store / "arrays").iterdir())) + "\n",
        encoding="utf-8",
    )
    for name in names:
        path = store / "arrays" / name
        if path.exists() and path.stat().st_size <= 1 << 20:
            shutil.copy2(path, evidence / name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store-dir", type=Path, required=True)
    parser.add_argument("--source-trace", type=Path, required=True)
    parser.add_argument("--legacy-archive", type=Path)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--expected-identity-manifest", type=Path, required=True)
    parser.add_argument("--expected-identity-receipt", type=Path, required=True)
    parser.add_argument("--evidence-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = args.store_dir.resolve()
    trace = args.source_trace.resolve()
    legacy = args.legacy_archive.resolve() if args.legacy_archive is not None else None
    verifier = args.verifier.resolve()
    expected_identity_manifest = args.expected_identity_manifest.resolve()
    expected_identity_receipt = args.expected_identity_receipt.resolve()
    evidence_root = args.evidence_dir.resolve()
    output = args.output.resolve()
    if output.exists() or evidence_root.exists():
        raise FileExistsError("tamper output/evidence 已存在")
    evidence_root.mkdir(parents=True)

    cases = [
        ("rebound_patch_offset", "instance patch 长度与 template 不一致"),
        ("rebound_payload_code", "array store 展开在 row 0 不一致"),
        ("rebound_template_event", "array store 展开在 row 0 不一致"),
        ("identity_relabel", "identity manifest/array window 不一致"),
        ("identity_dual_rebind", "store identity differs from frozen expected identity"),
        ("source_trace_replacement", "manifest 总容量或 source trace 绑定不一致"),
        ("verifier_source_replacement", "当前独立验证器与 store 中封存版本不一致"),
        ("manifest_derived_stat", "manifest 派生字段 class_stats 不一致"),
        ("extra_array_file", "arrays 目录存在缺失或额外文件"),
        ("missing_array_file", "arrays 目录存在缺失或额外文件"),
    ]
    results = []
    with tempfile.TemporaryDirectory(prefix="local5_phase_store_tamper_") as temporary:
        temporary_root = Path(temporary)
        for name, expected_message in cases:
            store = temporary_root / name
            clone_store(source, store)
            mutation: dict[str, object] = {"case": name}
            snapshots: list[str] = []
            case_trace = trace
            if name == "rebound_patch_offset":
                path = store / "arrays" / "patch_offsets.npy"
                value = np.load(path, mmap_mode="r+", allow_pickle=False)
                before = int(value[1])
                value[1] = before + 1
                value.flush()
                del value
                rebind_array(store, "patch_offsets")
                mutation.update({
                    "target": "patch_offsets[1]", "before": before,
                    "after": before + 1, "manifest_sha_rebound": True,
                    "target_sha256": sha256(path),
                })
                snapshots.append("patch_offsets.npy")
            elif name == "rebound_payload_code":
                path = store / "arrays" / "patch_payload_code.npy"
                dictionary = np.load(
                    store / "arrays" / "payload_dictionary.npy",
                    mmap_mode="r", allow_pickle=False,
                )
                if len(dictionary) < 2:
                    raise ValueError("payload dictionary lacks tamper alternative")
                value = np.load(path, mmap_mode="r+", allow_pickle=False)
                before = int(value[0])
                after = (before + 1) % len(dictionary)
                value[0] = after
                value.flush()
                del value
                rebind_array(store, "patch_payload_code")
                mutation.update({
                    "target": "patch_payload_code[0]", "before": before,
                    "after": after, "manifest_sha_rebound": True,
                    "target_sha256": sha256(path),
                })
            elif name == "rebound_template_event":
                path = store / "arrays" / "template_event_code.npy"
                dictionary = np.load(
                    store / "arrays" / "event_dictionary.npy",
                    mmap_mode="r", allow_pickle=False,
                )
                if len(dictionary) < 2:
                    raise ValueError("event dictionary lacks tamper alternative")
                value = np.load(path, mmap_mode="r+", allow_pickle=False)
                before = int(value[0])
                after = (before + 1) % len(dictionary)
                value[0] = after
                value.flush()
                del value
                rebind_array(store, "template_event_code")
                mutation.update({
                    "target": "template_event_code[0]", "before": before,
                    "after": after, "manifest_sha_rebound": True,
                    "target_sha256": sha256(path),
                })
                snapshots.append("template_event_code.npy")
            elif name == "identity_relabel":
                path = store / "manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                before = manifest["identity"]["window"]
                manifest["identity"]["window"] = before + 1
                write_json(path, manifest)
                mutation.update({
                    "target": "identity.window", "before": before,
                    "after": before + 1,
                })
            elif name == "identity_dual_rebind":
                manifest_path = store / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                before = manifest["identity"]["window"]
                after = before + 1
                manifest["identity"]["window"] = after
                write_json(manifest_path, manifest)
                array_path = store / "arrays" / "identity_window.npy"
                value = np.load(array_path, mmap_mode="r+", allow_pickle=False)
                value[0] = after
                value.flush()
                del value
                rebind_array(store, "identity_window")
                mutation.update({
                    "target": "identity.window + identity_window[0]",
                    "before": before, "after": after,
                    "manifest_sha_rebound": True,
                    "target_sha256": sha256(array_path),
                })
                snapshots.append("identity_window.npy")
            elif name == "source_trace_replacement":
                case_trace = temporary_root / "replacement_trace.csv"
                case_trace.write_text(
                    "cycle,event,tile,head,source,lane,out,delay,index,origin,payload\n"
                    "0,replaced,-1,-1,-1,-1,-1,-1,-1,test,-\n",
                    encoding="ascii",
                )
                mutation.update({
                    "target": "source trace argument", "action": "replace",
                    "replacement_sha256": sha256(case_trace),
                })
            elif name == "verifier_source_replacement":
                manifest_path = store / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                relative = manifest["source_bindings"]["independent_verifier"]["file"]
                path = store / relative
                path.write_text(
                    path.read_text(encoding="utf-8") + "\n# tampered verifier source\n",
                    encoding="utf-8",
                )
                manifest["source_bindings"]["independent_verifier"]["sha256"] = sha256(path)
                write_json(manifest_path, manifest)
                mutation.update({
                    "target": relative, "action": "append_and_rebind",
                    "target_sha256": sha256(path),
                })
                if path.parent == store / "source":
                    snapshots.append(path.name)
            elif name == "manifest_derived_stat":
                path = store / "manifest.json"
                manifest = json.loads(path.read_text(encoding="utf-8"))
                before = manifest["class_stats"]["prefix"]["expanded_rows"]
                manifest["class_stats"]["prefix"]["expanded_rows"] = before + 1
                write_json(path, manifest)
                mutation.update({
                    "target": "class_stats.prefix.expanded_rows",
                    "before": before, "after": before + 1,
                })
            elif name == "extra_array_file":
                np.save(store / "arrays" / "unexpected.npy", np.asarray([1], dtype=np.uint8))
                mutation.update({"target": "arrays/unexpected.npy", "action": "create"})
                snapshots.append("unexpected.npy")
            elif name == "missing_array_file":
                path = store / "arrays" / "schema_version.npy"
                mutation.update({
                    "target": "arrays/schema_version.npy", "action": "delete",
                    "deleted_sha256": sha256(path),
                })
                path.unlink()
            snapshot_case(store, evidence_root / name, snapshots)
            if name == "source_trace_replacement":
                shutil.copy2(case_trace, evidence_root / name / "replacement_trace.csv")
            elif name == "verifier_source_replacement":
                relative = str(mutation["target"])
                shutil.copy2(store / relative, evidence_root / name / Path(relative).name)
            run = invoke(
                verifier, store, case_trace, legacy, temporary_root / f"{name}.json",
                expected_identity_manifest, expected_identity_receipt,
            )
            rejected = run.returncode != 0 and expected_message in run.stderr
            if not rejected:
                raise RuntimeError(
                    f"tamper {name} 未按预期拒绝: rc={run.returncode}\n{run.stderr[-2000:]}"
                )
            result = {
                "case": name,
                "status": "PASS_REJECTED",
                "expected_message": expected_message,
                "returncode": run.returncode,
                "mutation": mutation,
                "stderr_tail": run.stderr[-2000:],
                "evidence_manifest_sha256": sha256(evidence_root / name / "manifest.json"),
            }
            write_json(evidence_root / name / "case.json", result)
            results.append(result)
            shutil.rmtree(store)

    report = {
        "schema": "local5_phase_array_store_tamper_regression_v2",
        "status": "PASS_ALL_TAMPERS_REJECTED_NOT_G0",
        "evidence": "[独立软件负例]",
        "formal_g0": "DENY",
        "cases": results,
        "passed": len(results),
        "total": len(cases),
        "bindings": {
            "store_manifest_sha256": sha256(source / "manifest.json"),
            "source_trace_sha256": sha256(trace),
            "verifier_source_sha256": sha256(verifier),
            "expected_identity_manifest_sha256": sha256(expected_identity_manifest),
            "expected_identity_receipt_sha256": sha256(expected_identity_receipt),
        },
        "boundary": [
            "patch offset、payload 和 template 篡改均重绑文件 SHA，拒绝来自语义检查而非散列检查",
            "负例仅验证归档合同，不是 RTL 功能覆盖或架构性能",
        ],
    }
    if legacy is not None:
        report["bindings"]["legacy_archive_sha256"] = sha256(legacy)
    write_json(output, report)
    print(json.dumps({"status": report["status"], "passed": len(results)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
