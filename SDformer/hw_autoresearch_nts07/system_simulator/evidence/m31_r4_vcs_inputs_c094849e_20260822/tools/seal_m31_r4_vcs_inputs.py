#!/usr/bin/env python3
"""Seal the exact six M31-r4 VCS PASS inputs into a read-only snapshot."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


MANIFEST_SHA256 = (
    "efdc366f86198519f2b58fbd9e155e4aed2f2a8f0785225c6defd719ae8b3093")
CORE_SHA256 = (
    "c094849e88c0d9fc3a390d0cf6fc9adf10ff4dc31d77e265e425e5cf71b5ef15")
EXPECTED = {
    "rtl_m31/qfit_signed_int8_mul96_pool.sv":
        "7872d25c01c112f07a7d8e3cfe728029eef1f68e0f7bf87bdf2a50416776ea18",
    "rtl_m31/qfit_atlif_unified_t10_t2_stream_core.sv": CORE_SHA256,
    "verif_m31/qfit_atlif_unified_t10_t2_stream_assertions.sv":
        "695fd1923d0a9f6a2af40fb008e1c3ff4c1fec7aa88b6724cb7c7bac29e8f5da",
    "tb_m31/tb_qfit_atlif_unified_t10_t2_stream_core.sv":
        "9d1a59b59e8711d137ac64be2f8b2e0314ea5b3fd08d75f44bbbf21d42ea7b79",
    "dc_handoff/filelists/date_m31_unified_t10_t2_vcs.f":
        "435550cf64b2a71debefd69cf582f37adc0a30b49b886c46e4087d1b37cc94a9",
    "dc_handoff/scripts/run_vcs_m31_unified_t10_t2_sva.sh":
        "a8469c5d4e61943339788134023c72474c24009448ab3d45f88762435d763d59",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def contained(path, root):
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def parse_manifest(manifest, hw_root):
    manifest = Path(manifest).resolve()
    hw_root = Path(hw_root).resolve()
    if not manifest.is_file() or manifest.is_symlink():
        raise ValueError("M31 r4 VCS input manifest is missing or a symlink")
    if sha256(manifest) != MANIFEST_SHA256:
        raise ValueError("M31 r4 VCS input manifest identity drift")
    rows = {}
    canonical_sources = set()
    for line_number, line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(), 1):
        match = re.match(r"^([0-9a-f]{64})  ([^\0]+)$", line)
        if not match:
            raise ValueError("malformed M31 r4 manifest line {}".format(
                line_number))
        expected, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("M31 r4 manifest path escape")
        if relative in rows:
            raise ValueError("duplicate M31 r4 manifest target")
        raw_source = hw_root / relative_path
        if raw_source.is_symlink():
            raise ValueError("M31 r4 manifest source is a symlink")
        source = raw_source.resolve()
        if (not contained(source, hw_root) or not source.is_file()
                or source.stat().st_size == 0):
            raise ValueError("M31 r4 manifest source is missing or escapes root")
        if source in canonical_sources:
            raise ValueError("M31 r4 canonical source collision")
        canonical_sources.add(source)
        if sha256(source) != expected:
            raise ValueError("M31 r4 live input hash drift: {}".format(relative))
        rows[relative] = (expected, source)
    if (set(rows) != set(EXPECTED)
            or any(rows[key][0] != EXPECTED[key] for key in EXPECTED)):
        raise ValueError("M31 r4 exact six-input manifest population drift")
    return manifest, rows


def write_exclusive(path, content, mode=0o644):
    descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    with os.fdopen(descriptor, "w") as handle:
        handle.write(content)


def seal(hw_root, manifest, output_dir, ledger_path):
    hw_root = Path(hw_root).resolve()
    if not hw_root.is_dir():
        raise ValueError("M31 r4 hardware root is missing")
    manifest, rows = parse_manifest(manifest, hw_root)
    output_dir = Path(output_dir).resolve()
    ledger_path = Path(ledger_path).resolve()
    if "c094849e" not in output_dir.name:
        raise ValueError("M31 r4 snapshot name must bind core SHA prefix c094849e")
    if output_dir.exists() or output_dir.is_symlink() \
            or ledger_path.exists() or ledger_path.is_symlink():
        raise ValueError("refusing to overwrite M31 r4 VCS input snapshot")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if ledger_path.parent != output_dir.parent:
        raise ValueError("M31 r4 snapshot ledger must be adjacent to snapshot")

    staging = Path(tempfile.mkdtemp(prefix=".m31_r4_vcs_inputs.",
                                    dir=str(output_dir.parent)))
    staging_ledger = output_dir.parent / (
        ".{}.{}.tmp".format(ledger_path.name, os.getpid()))
    try:
        input_root = staging / "inputs/hw_root"
        source_map_rows = []
        for relative in sorted(rows):
            expected, source = rows[relative]
            target = (input_root / relative).resolve()
            if not contained(target, staging):
                raise ValueError("M31 r4 snapshot target escape")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(str(source), str(target))
            if sha256(target) != expected:
                raise ValueError("M31 r4 snapshot copy hash drift")
            source_map_rows.append("{}\t{}\t{}\n".format(
                expected, source, target.relative_to(staging)))

        frozen_manifest = staging / "input_sha256.txt"
        shutil.copyfile(str(manifest), str(frozen_manifest))
        if sha256(frozen_manifest) != MANIFEST_SHA256:
            raise ValueError("M31 r4 frozen manifest hash drift")
        source_map_rows.append("{}\t{}\t{}\n".format(
            MANIFEST_SHA256, manifest, frozen_manifest.relative_to(staging)))
        sealer_source = Path(__file__).resolve()
        frozen_sealer = staging / "tools/seal_m31_r4_vcs_inputs.py"
        frozen_sealer.parent.mkdir(parents=True)
        shutil.copyfile(str(sealer_source), str(frozen_sealer))
        sealer_sha256 = sha256(sealer_source)
        if sha256(frozen_sealer) != sealer_sha256:
            raise ValueError("M31 r4 frozen sealer hash drift")
        source_map_rows.append("{}\t{}\t{}\n".format(
            sealer_sha256, sealer_source, frozen_sealer.relative_to(staging)))
        (staging / "source_map.tsv").write_text(
            "".join(source_map_rows), encoding="utf-8")
        admission = {
            "schema": "m31_r4_vcs_six_input_snapshot_v1",
            "status": "PASS_EXACT_FROZEN_SIX_INPUTS_READ_ONLY",
            "manifest_sha256": MANIFEST_SHA256,
            "core_rtl_sha256": CORE_SHA256,
            "sealer_sha256": sealer_sha256,
            "input_count": 6,
            "all_live_inputs_rehashed_before_copy": True,
            "all_snapshot_inputs_rehashed_after_copy": True,
            "claim_boundary": {
                "permitted": "immutable byte snapshot of the exact six M31-r4 VCS PASS inputs",
                "forbidden": "new VCS execution, DC/Formality/PPA/power/system claims, or live-source replacement",
            },
        }
        (staging / "snapshot_admission.json").write_text(
            json.dumps(admission, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")

        files = sorted(path for path in staging.rglob("*") if path.is_file())
        ledger_rows = []
        for path in files:
            final_relative = Path(output_dir.name) / path.relative_to(staging)
            ledger_rows.append("{}  {}\n".format(sha256(path), final_relative))
        write_exclusive(staging_ledger, "".join(ledger_rows))
        os.rename(str(staging), str(output_dir))
        os.rename(str(staging_ledger), str(ledger_path))

        listed = set()
        for line in ledger_path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^([0-9a-f]{64})  ([^\0]+)$", line)
            if not match:
                raise ValueError("M31 r4 final snapshot ledger malformed")
            expected, relative = match.groups()
            path = (output_dir.parent / relative).resolve()
            if not contained(path, output_dir) or sha256(path) != expected:
                raise ValueError("M31 r4 final snapshot ledger hash/containment drift")
            listed.add(path)
        actual = set(path.resolve() for path in output_dir.rglob("*")
                     if path.is_file())
        if listed != actual:
            raise ValueError("M31 r4 final snapshot ledger is not exact")

        for path in actual:
            path.chmod(0o444)
        for path in sorted(
                (path for path in output_dir.rglob("*") if path.is_dir()),
                key=lambda item: len(item.parts), reverse=True):
            path.chmod(0o555)
        output_dir.chmod(0o555)
        ledger_path.chmod(0o444)
        return {
            "snapshot_directory": str(output_dir),
            "snapshot_ledger": str(ledger_path),
            "snapshot_ledger_sha256": sha256(ledger_path),
            "snapshot_file_count": len(actual),
        }
    except Exception:
        if staging.exists():
            shutil.rmtree(str(staging))
        if staging_ledger.exists():
            staging_ledger.unlink()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    args = parser.parse_args()
    result = seal(args.hw_root, args.manifest, args.output_dir, args.ledger)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
