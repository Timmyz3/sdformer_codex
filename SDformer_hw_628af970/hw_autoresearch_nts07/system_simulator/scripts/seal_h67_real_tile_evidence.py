#!/usr/bin/env python3
"""Seal a completed H67 real-tile profiling run into an immutable evidence manifest."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def identity(path, root):
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    repo_root = args.repo_root.resolve()
    tile_dir = run_dir / "real_tiles"
    required = [
        run_dir / "nts11_hardware_p0_profile.json",
        run_dir / "execution_trace.csv",
        run_dir / "dual_line_operator_trace.csv",
        run_dir / "profile_cmdline.txt",
        run_dir / "profile_environment.txt",
        run_dir / "console.log",
        tile_dir / "manifest.json",
        tile_dir / "tile_records.csv",
        tile_dir / "packed_tiles.npz",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise SystemExit("missing evidence files: " + ", ".join(missing))

    manifest = json.loads((tile_dir / "manifest.json").read_text(encoding="utf-8"))
    for name in ("tile_records.csv", "packed_tiles.npz"):
        actual = sha256(tile_dir / name)
        expected = manifest["sha256"][name]
        if actual != expected:
            raise SystemExit("tile manifest hash mismatch for {}".format(name))

    command_tokens = [
        line for line in (run_dir / "profile_cmdline.txt").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if "--dual-line-tile-pairs-per-call" not in command_tokens:
        raise SystemExit("captured command lacks tile pair count")
    pair_index = command_tokens.index("--dual-line-tile-pairs-per-call")
    if pair_index + 1 >= len(command_tokens):
        raise SystemExit("captured command has no tile pair value")
    command_pairs = int(command_tokens[pair_index + 1])
    manifest_pairs = int(manifest["pairs_per_operator_call"])
    if command_pairs != manifest_pairs:
        raise SystemExit("command/manifest pair count mismatch")
    if not command_tokens or not command_tokens[0].endswith("python"):
        raise SystemExit("captured command is not a Python profiler invocation")

    source_hashes = manifest["run_context"]["source_sha256"]
    profiler = repo_root / "neuron_experiments/H9_bipolar_self_attention/entrypoints/profile_nts11_hardware_p0.py"
    writer = repo_root / "neuron_experiments/H9_bipolar_self_attention/entrypoints/h67_dual_line_tile_trace.py"
    if sha256(profiler) != source_hashes["profiler"]:
        raise SystemExit("profiler source no longer matches runtime identity")
    if sha256(writer) != source_hashes["tile_writer"]:
        raise SystemExit("tile writer source no longer matches runtime identity")

    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    snapshot_sources = {
        "profile_nts11_hardware_p0.py": profiler,
        "h67_dual_line_tile_trace.py": writer,
        "seal_h67_real_tile_evidence.py": Path(__file__).resolve(),
    }
    for name, source in snapshot_sources.items():
        shutil.copy2(str(source), str(snapshot_dir / name))
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(repo_root), universal_newlines=True
    ).strip()
    git_status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=str(repo_root), universal_newlines=True,
    )
    (snapshot_dir / "git_head.txt").write_text(git_head + "\n", encoding="utf-8")
    (snapshot_dir / "git_status_porcelain.txt").write_text(git_status, encoding="utf-8")
    required.extend([
        snapshot_dir / "profile_nts11_hardware_p0.py",
        snapshot_dir / "h67_dual_line_tile_trace.py",
        snapshot_dir / "seal_h67_real_tile_evidence.py",
        snapshot_dir / "git_head.txt",
        snapshot_dir / "git_status_porcelain.txt",
    ])

    files = {}
    for path in required:
        files[str(path.relative_to(run_dir))] = identity(path, run_dir)
    payload = {
        "schema": "h67_real_tile_evidence_v1",
        "status": "PASS_IMMUTABLE_PROFILE_EVIDENCE_NOT_FULL_SPATIAL_OR_CYCLE_PROOF",
        "records": manifest["records"],
        "row_chunk_identities": manifest["row_chunk_identities"],
        "pairs_per_operator_call": manifest_pairs,
        "command_tokens": command_tokens,
        "environment": (run_dir / "profile_environment.txt").read_text(encoding="utf-8").splitlines(),
        "artifact_identity": manifest["run_context"]["artifact_identity"],
        "source_sha256": source_hashes,
        "git_head": git_head,
        "files": files,
        "claim_boundary": (
            "Immutable checkpoint/config/source/command/environment/ordered-trace/bitmap identity; "
            "sampling evidence only, not a full-spatial cycle, speed, energy, or PPA result."
        ),
    }
    output = args.output.resolve() if args.output else run_dir / "evidence_manifest.json"
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("PASS_H67_REAL_TILE_EVIDENCE records={} pairs={} output={}".format(
        payload["records"], manifest_pairs, output
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
