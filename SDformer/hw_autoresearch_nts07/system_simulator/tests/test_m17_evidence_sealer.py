#!/usr/bin/env python3
"""Self-containment and no-overwrite check for the M17 evidence sealer."""

from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
SEALER = REPO / "hw_autoresearch_nts07/system_simulator/scripts/seal_m17_full_spatial_c4_evidence.py"


def test_sealer() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        run = root / "run"
        (run / "full_spatial_c4").mkdir(parents=True)
        for relative in (
            "full_spatial_c4/manifest.json", "full_spatial_c4/prototypes.json",
            "full_spatial_c4/ordered_stream.npz", "dual_line_operator_trace.csv",
            "m17_reconciliation.json", "profile_cmdline.txt", "analyzer_cmdline.txt",
            "profile_environment.txt", "console.log",
        ):
            (run / relative).write_bytes((relative + "\n").encode())
        evidence_input = root / "dependency_manifest.json"
        evidence_input.write_text("{}\n", encoding="utf-8")
        output = root / "seal.tar"
        command = [
            sys.executable, str(SEALER), "--repo-root", str(REPO), "--run-dir", str(run),
            "--output", str(output), "--input", "dependency_manifest", str(evidence_input),
        ]
        subprocess.run(command, check=True, stdout=subprocess.PIPE, text=True)
        assert output.is_file() and (root / "seal.tar.sha256").is_file()
        with tarfile.open(output, "r") as archive:
            names = set(archive.getnames())
            assert "evidence_manifest.json" in names
            assert "results/m17_reconciliation.json" in names
            assert "inputs/dependency_manifest/dependency_manifest.json" in names
            assert any(name.endswith("profile_nts11_hardware_p0.py") for name in names)
            payload = json.load(archive.extractfile("evidence_manifest.json"))
            assert payload["status"].startswith("SEALED_HASH_COMPLETE")
            assert payload["results"]["console.log"]["bytes"] > 0
        repeated = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        assert repeated.returncode != 0 and "refusing to overwrite" in repeated.stderr


if __name__ == "__main__":
    test_sealer()
    print("PASS m17-evidence-sealer 1/1")
