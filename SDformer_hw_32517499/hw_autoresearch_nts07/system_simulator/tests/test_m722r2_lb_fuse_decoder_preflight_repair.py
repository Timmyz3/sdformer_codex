#!/usr/bin/env python3
"""Regression for the M722-r2 nested-seal population repair."""

import importlib.util
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "system_simulator/scripts/analyze_m722r2_lb_fuse_decoder_cpu_fastkill.py"
SPEC = importlib.util.spec_from_file_location("m722r2", SCRIPT)
M722R2 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M722R2)


class M722R2Tests(unittest.TestCase):
    def test_nested_manifests_are_members_not_root_seals(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "sealed"
            nested = root / "nested"
            nested.mkdir(parents=True)
            (nested / "payload.bin").write_bytes(b"m722")
            (nested / "SHA256SUMS").write_text(
                "{}  payload.bin\n".format(
                    M722R2.R1.sha256(nested / "payload.bin")),
                encoding="utf-8")
            (nested / "SHA256SUMS.seal.sha256").write_text(
                "{}  SHA256SUMS\n".format(
                    M722R2.R1.sha256(nested / "SHA256SUMS")),
                encoding="utf-8")
            members = sorted(path for path in root.rglob("*")
                             if path.is_file())
            (root / "SHA256SUMS").write_text("".join(
                "{}  {}\n".format(M722R2.R1.sha256(path),
                                    path.relative_to(root).as_posix())
                for path in members), encoding="utf-8")
            (root / "SHA256SUMS.seal.sha256").write_text(
                "{}  SHA256SUMS\n".format(
                    M722R2.R1.sha256(root / "SHA256SUMS")),
                encoding="utf-8")
            identity = M722R2.verify_directory(root)
            self.assertEqual(identity["manifest_sha256"],
                             M722R2.R1.sha256(root / "SHA256SUMS"))

    def test_r1_model_is_frozen(self):
        self.assertEqual(
            M722R2.R1.sha256(M722R2.R1_PATH),
            "3693fd1078738e8e3e0928080802cf2f276d5cb5951f72134a4482ce364077df")


if __name__ == "__main__":
    unittest.main()
