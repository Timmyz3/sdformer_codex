#!/usr/bin/env python3
"""Build a relocatable H67 ep35 checkpoint/profile/QK-trace handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
HW = REPO / "hw_autoresearch_nts07"
EXP = REPO / "neuron_experiments/H9_bipolar_self_attention"
RUN = EXP / "results/dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_20260805"
TRACE = HW / "results/h67_ep35_multisample100_t450_real_rtl_bit_trace"
PROFILE = EXP / "results/h67_fullres_ep35_t450_profile100_20260818"
PACKAGE_NAME = "h67_ep35_system_trace_handoff_20260821"

CHECKPOINT = RUN / "checkpoint_epoch35.pth"
CONFIG = EXP / (
    "configs/generated/"
    "dsec_fullres_w15_H67_bb1e4_equal_plus10_ep40_"
    "hardware_order_q7q17_deploy.yml"
)

LOCKED_SHA256 = {
    CHECKPOINT: "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    CONFIG: "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49",
    TRACE / "manifest.json": "2bb0dc3e7bfd6187ba66feab4a07a1113a522165aa224152d6f27f652e9ea4f2",
    PROFILE / "nts11_hardware_p0_profile.json": (
        "e564f801130271e24c82bdc9dcd2242b8a1d7f0d048488e51fd1a445d8df930b"
    ),
    RUN / "standard_valid825/epoch35/spike_profile.json": (
        "6ab20f196d8efca78bdc82bdb67a9ad768edf525e0763fe0d424a85afea19ad3"
    ),
    RUN / "deploy_valid825/hardware_order_q7q17/epoch35/spike_profile.json": (
        "ba94788b969497a2b3e74bc943136157a32e231fc9f2ce05555f898ff545cbce"
    ),
    EXP / "entrypoints/profile_nts11_hardware_p0.py": (
        "5f21c8d7eae27e251edfc07d9cfa3a75307893d1d2a5e1a522b1fa027c4bdf22"
    ),
    EXP / "entrypoints/h67_bit_trace.py": (
        "75c9134061aa06c8050389cbaac0a80a7956911cda0f8ce7b4144ba40ab3f58e"
    ),
}

CODE_FILES = [
    EXP / "entrypoints/profile_nts11_hardware_p0.py",
    EXP / "entrypoints/h67_bit_trace.py",
    EXP / "entrypoints/run_h67_ep35_profile100_bit_trace_20260818.py",
    EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py",
    REPO / "third_party/SDformerFlow/eval_DSEC_flow_SNN.py",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"required file missing: {path}")


def verify_locked_files() -> None:
    for path, expected in LOCKED_SHA256.items():
        require_file(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(
                f"identity drift: {path}\nexpected={expected}\nactual={actual}"
            )
    for path in CODE_FILES:
        require_file(path)


def verify_trace_population() -> dict:
    manifest_path = TRACE / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = manifest.get("records")
    if not isinstance(records, list) or len(records) != 1200:
        raise RuntimeError(f"expected 1200 trace records, got {len(records or [])}")

    seen: set[str] = set()
    for index, record in enumerate(records):
        source = TRACE / Path(record["file"]).name
        require_file(source)
        if source.name in seen:
            raise RuntimeError(f"duplicate trace basename: {source.name}")
        seen.add(source.name)
        actual = sha256(source)
        if actual != record["sha256"]:
            raise RuntimeError(
                f"trace SHA mismatch at record {index}: {source.name}"
            )

    npz_files = sorted(TRACE.glob("sample*_*.npz"))
    if len(npz_files) != 1200 or {p.name for p in npz_files} != seen:
        raise RuntimeError(
            f"trace population mismatch: manifest={len(seen)} npz={len(npz_files)}"
        )
    return {
        "records": len(records),
        "samples": 100,
        "blocks_per_sample": 12,
        "windows_per_block": 1,
    }


def link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def link_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file():
            link_file(path, destination / path.relative_to(source))


def write_verifier(destination: Path) -> None:
    verifier = """#!/usr/bin/env python3
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / "handoff_manifest.json").read_text(encoding="utf-8"))
errors = []
for entry in manifest["files"]:
    path = root / entry["path"]
    if not path.is_file():
        errors.append(f"missing: {entry['path']}")
        continue
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != entry["sha256"] or path.stat().st_size != entry["size"]:
        errors.append(f"identity mismatch: {entry['path']}")
if errors:
    raise SystemExit("FAIL\\n" + "\\n".join(errors))
print(f"PASS files={len(manifest['files'])} bytes={manifest['total_bytes']}")
"""
    destination.write_text(verifier, encoding="utf-8")
    destination.chmod(0o755)


def stage_package(stage: Path, trace_summary: dict) -> dict:
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    link_file(CHECKPOINT, stage / "checkpoint/checkpoint_epoch35.pth")
    link_file(CONFIG, stage / "config/deploy_q7q17.yml")
    link_tree(TRACE, stage / "trace_qk_100sample_12block")
    link_tree(PROFILE, stage / "profile100")
    link_tree(RUN / "standard_valid825/epoch35", stage / "valid825/float_epoch35")
    link_tree(
        RUN / "deploy_valid825/hardware_order_q7q17/epoch35",
        stage / "valid825/hardware_order_q7q17_epoch35",
    )
    for path in CODE_FILES:
        link_file(path, stage / "source" / path.relative_to(REPO))
    link_file(HW / "system_handoff/README.md", stage / "README.md")

    relocated = json.loads((TRACE / "manifest.json").read_text(encoding="utf-8"))
    for record in relocated["records"]:
        record["file"] = str(
            Path("trace_qk_100sample_12block") / Path(record["file"]).name
        )
    identity = relocated["run_context"]["artifact_identity"]
    identity["config_path"] = "config/deploy_q7q17.yml"
    identity["checkpoint_path"] = "checkpoint/checkpoint_epoch35.pth"
    relocated_path = stage / "trace_qk_100sample_12block/manifest.relocated.json"
    relocated_path.write_text(
        json.dumps(relocated, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    write_verifier(stage / "verify_handoff.py")
    files = []
    total_bytes = 0
    for path in sorted(stage.rglob("*")):
        if not path.is_file() or path.name == "handoff_manifest.json":
            continue
        size = path.stat().st_size
        total_bytes += size
        files.append(
            {
                "path": str(path.relative_to(stage)),
                "size": size,
                "sha256": sha256(path),
            }
        )
    manifest = {
        "schema": "h67_ep35_system_trace_handoff_v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "claim_boundary": (
            "Attention-complete for 100 samples x 12 blocks x one window; "
            "not a full-network transaction trace and not encoder PPA."
        ),
        "checkpoint_sha256": LOCKED_SHA256[CHECKPOINT],
        "config_sha256": LOCKED_SHA256[CONFIG],
        "trace_population": trace_summary,
        "total_bytes": total_bytes,
        "files": files,
    }
    (stage / "handoff_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def pack(output_dir: Path, keep_stage: bool, trace_summary: dict) -> Path:
    stage = output_dir / PACKAGE_NAME
    manifest = stage_package(stage, trace_summary)
    archive = output_dir / f"{PACKAGE_NAME}.tar.zst"
    if archive.exists():
        archive.unlink()
    subprocess.run(
        [
            "tar",
            "--zstd",
            "-cf",
            str(archive),
            "-C",
            str(output_dir),
            PACKAGE_NAME,
        ],
        check=True,
    )
    archive_sha = sha256(archive)
    checksum = archive.with_suffix(archive.suffix + ".sha256")
    checksum.write_text(f"{archive_sha}  {archive.name}\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "PASS",
                "archive": str(archive),
                "archive_sha256": archive_sha,
                "archive_bytes": archive.stat().st_size,
                "payload_bytes": manifest["total_bytes"],
                "files": len(manifest["files"]),
            },
            indent=2,
        )
    )
    if not keep_stage:
        shutil.rmtree(stage)
    return archive


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--verify-only", action="store_true")
    action.add_argument("--pack", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HW / "system_handoff/packs",
    )
    parser.add_argument("--keep-stage", action="store_true")
    args = parser.parse_args()

    verify_locked_files()
    trace_summary = verify_trace_population()
    if args.verify_only:
        print(json.dumps({"status": "PASS", **trace_summary}, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pack(args.output_dir.resolve(), args.keep_stage, trace_summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
