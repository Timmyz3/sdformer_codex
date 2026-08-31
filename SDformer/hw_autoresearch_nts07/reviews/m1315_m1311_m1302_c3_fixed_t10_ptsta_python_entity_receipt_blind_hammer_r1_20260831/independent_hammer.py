#!/usr/bin/python3.12
"""Receipt-blind static/mock hammer for the M1311 Python-entity PT wrapper.

This file never calls the one-shot entry point, lmutil, PrimeTime, or any EDA tool.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import tempfile
import unittest


HW = Path(__file__).resolve().parents[3] / "hw_autoresearch_nts07"
HELPER = HW / "dc_handoff/scripts/check_m1311_python_symlink_entity.sh"
WRAPPER = HW / "dc_handoff/scripts/run_m1311_m1302_fixed_t10_ptsta_python_entity_one_shot.sh"
ORCHESTRATOR = HW / "dc_handoff/scripts/run_m1311_m1302_fixed_t10_ptsta_python_entity_one_shot.py"
AUTHOR_TEST = HW / "tests/test_m1311_m1302_python_entity_ptsta_source_static.py"
ADMISSION = HW / "contracts/m1311_m1302_c3_fixed_t10_ptsta_python_entity_launch_admission_r1_20260831.json"
CONTRACT = HW / "contracts/m1311_m1302_c3_fixed_t10_ptsta_python_entity_source_contract_r1_20260831.json"
AUTHOR = HW / "reviews/m1311_m1302_c3_fixed_t10_ptsta_python_entity_author_receipt_r1_20260831"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

PINS = {
    HELPER: "25e4bf69d69f9ac6069ce160d252539cbd4c15232284c0a6256fa5d19dcce223",
    WRAPPER: "c15bc5d5ba0e1faa4dae513a74f6def972b4b683cd3d7b936062889f11429bd5",
    ORCHESTRATOR: "8ac4517f7a6a7dfc6d1abedd4488b4a40f53d58f1f0a0fc69fbf8f212fceb178",
    AUTHOR_TEST: "afabd72595e3e22182081fd26bb78aefb7f1411840caeef13ddafb6e92662258",
    ADMISSION: "6a0df8588af9cfac813c0db5bfbce2df04f18f2ce2ee616365ed6143546feedf",
    CONTRACT: "4b722030d438859555be1d74ca27ec07654970061d0d04ba77421578c6c01c3f",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
AUTHOR_MANIFEST_SHA = "c164c190ce0a6d0069dc88bf78d035f13985639d81b0a5e7e56d4fa94c5e1115"
AUTHOR_OUTER_FILE_SHA = "3019a6a8899b23dd192b0233df06c86742e98427e62f38307dc35a6e6b327653"
PYTHON_SHA = "9c9502e21917eff03ffe4672c4e61cf8ce651aabeaf5118e423782feba58787f"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def regular(path: Path) -> None:
    mode = os.lstat(str(path)).st_mode
    if not stat.S_ISREG(mode):
        raise AssertionError("not regular: " + str(path))


def payload_seal(path: Path) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path); regular(side); regular(outer)
    if side.read_text().split() != [sha(path), path.name]:
        raise AssertionError("payload sidecar")
    if outer.read_text().split() != [sha(side), side.name]:
        raise AssertionError("payload outer seal")


def dir_seal(path: Path) -> None:
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    regular(manifest); regular(outer)
    if outer.read_text().split() != [sha(manifest), "SHA256SUMS"]:
        raise AssertionError("directory outer seal")
    seen = set()
    for row in manifest.read_text().splitlines():
        digest, rel = row.split(None, 1)
        rel = rel.lstrip("*")
        if rel.startswith("./"):
            rel = rel[2:]
        if rel in seen or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise AssertionError("manifest path")
        seen.add(rel)
        member = path / rel
        regular(member)
        if sha(member) != digest:
            raise AssertionError("manifest member")


def load_orchestrator():
    spec = importlib.util.spec_from_file_location("m1315_target", str(ORCHESTRATOR))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def helper_fixture(attack: str | None = None) -> int:
    with tempfile.TemporaryDirectory(prefix="m1315_entity.") as tmp:
        root = Path(tmp)
        entity = root / "entity"
        link3 = root / "python3.6"
        link2 = root / "alternative"
        link1 = root / "python3"
        if attack == "nonregular":
            entity.mkdir()
        else:
            shutil.copy2("/bin/true", str(entity)); entity.chmod(0o755)
        link3.symlink_to(str(entity)); link2.symlink_to(str(link3)); link1.symlink_to(str(link2))
        if attack == "target_swap":
            other = root / "other"
            shutil.copy2("/bin/false", str(other)); other.chmod(0o755)
            link1.unlink(); link1.symlink_to(str(other))
        elif attack == "dangling":
            link3.unlink(); link3.symlink_to(str(root / "missing"))
        st = entity.stat()
        digest = sha(entity) if entity.is_file() else "0" * 64
        if attack == "sha_drift":
            digest = "0" * 64
        command = ["/usr/bin/bash", str(HELPER), str(link1), str(link2), str(link2),
                   str(link3), str(link3), str(entity), str(entity), str(st.st_dev),
                   str(st.st_ino), "%o" % stat.S_IMODE(st.st_mode), str(st.st_size), digest]
        return subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE).returncode


class Hammer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_orchestrator()

    def test_01_exact_source_identities_and_syntax(self):
        for path, wanted in PINS.items():
            with self.subTest(path=path.name):
                regular(path); self.assertEqual(sha(path), wanted)
        subprocess.run(["/usr/bin/bash", "-n", str(HELPER)], check=True)
        subprocess.run(["/usr/bin/bash", "-n", str(WRAPPER)], check=True)
        subprocess.run(["/usr/libexec/platform-python3.6", "-m", "py_compile", str(ORCHESTRATOR)], check=True)

    def test_02_source_and_admission_double_seals(self):
        payload_seal(CONTRACT); payload_seal(ADMISSION)
        self.assertEqual(sha(Path(str(CONTRACT) + ".sha256")),
                         "e8af4ef6aaab1405b383aabf647219ea2015aa3a2cdcdcd0ac141bc69a7d8a7b")
        self.assertEqual(sha(Path(str(ADMISSION) + ".sha256")),
                         "04eb3b003fc461b0609d6879a27adab90789814191f1227d959e350d2b33c719")

    def test_03_author_directory_double_seal_without_receipt_trust(self):
        self.assertEqual(sha(AUTHOR / "SHA256SUMS"), AUTHOR_MANIFEST_SHA)
        self.assertEqual(sha(AUTHOR / "SHA256SUMS.seal.sha256"), AUTHOR_OUTER_FILE_SHA)
        dir_seal(AUTHOR)

    def test_04_all_admission_exact_files_and_authority(self):
        admission = json.loads(ADMISSION.read_text())
        self.assertEqual(len(admission["exact_files"]), 26)
        for relative, wanted in admission["exact_files"].items():
            path = HW / relative
            with self.subTest(relative=relative):
                regular(path); self.assertEqual(sha(path), wanted)
        self.assertFalse(admission["authorization"]["launch_now"])
        self.assertTrue(admission["authorization"]["max_attempts_is_one"])
        self.assertTrue(admission["result_adjudication"]["fresh_result_hammer_required"])
        self.assertFalse(admission["claim_boundary"]["headline"])

    def test_05_author_static_is_independently_replayed_10_of_10(self):
        completed = subprocess.run(["/usr/bin/python3.12", str(AUTHOR_TEST)],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   text=True, check=True)
        result = json.loads(completed.stdout)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["checks"]), 10)
        self.assertTrue(all(result["checks"].values()))
        self.assertEqual(result["license_pt_eda_calls"], 0)

    def test_06_real_python_chain_entity_and_open_fd_recheck(self):
        admission = json.loads(ADMISSION.read_text())["python_entity"]
        chain = [(Path("/usr/bin/python3"), "/etc/alternatives/python3"),
                 (Path("/etc/alternatives/python3"), "/usr/bin/python3.6"),
                 (Path("/usr/bin/python3.6"), "/usr/libexec/platform-python3.6")]
        for link, target in chain:
            self.assertTrue(link.is_symlink()); self.assertEqual(os.readlink(str(link)), target)
        entity = Path(admission["resolved_path"]); regular(entity)
        st = entity.stat()
        self.assertEqual((st.st_dev, st.st_ino, stat.S_IMODE(st.st_mode), st.st_size, sha(entity)),
                         (admission["device"], admission["inode"], 0o755,
                          admission["size_bytes"], PYTHON_SHA))
        fd = os.open(str(entity), os.O_RDONLY)
        try:
            fd_path = Path("/proc/%d/fd/%d" % (os.getpid(), fd))
            fst = os.stat(str(fd_path))
            self.assertEqual((fst.st_dev, fst.st_ino, stat.S_IMODE(fst.st_mode), fst.st_size),
                             (st.st_dev, st.st_ino, stat.S_IMODE(st.st_mode), st.st_size))
            self.assertEqual(sha(fd_path), PYTHON_SHA)
        finally:
            os.close(fd)

    def test_07_entity_attack_fixtures_fail_closed(self):
        self.assertEqual(helper_fixture(), 0)
        for attack in ("target_swap", "dangling", "nonregular", "sha_drift"):
            with self.subTest(attack=attack):
                self.assertNotEqual(helper_fixture(attack), 0)

    def test_08_repo_scoped_collision_positive_and_external_negative(self):
        m = self.module
        self.assertTrue(m.process_is_repo_scoped(str(m.REPO / "hw_autoresearch_nts07"),
                                                ["/opt/synopsys/pt_shell"]))
        self.assertTrue(m.process_is_repo_scoped("/tmp", [str(m.REPO / "run.tcl")]))
        self.assertFalse(m.process_is_repo_scoped("/home/fangyl/Work/project",
                                                 ["/opt/synopsys/vcs", "simv"]))

    def test_09_strict_result_mock_rejects_hold_unconstrained_and_coverage(self):
        test_spec = importlib.util.spec_from_file_location("m1315_author_test", str(AUTHOR_TEST))
        test_module = importlib.util.module_from_spec(test_spec); test_spec.loader.exec_module(test_module)
        good = test_module.receipt_mock(self.module, 0.0, 0, 0)
        neg = test_module.receipt_mock(self.module, -0.01, 0, 0)
        unconstrained = test_module.receipt_mock(self.module, 0.0, 1, 0)
        uncovered = test_module.receipt_mock(self.module, 0.0, 0, 1)
        self.assertTrue(good["strict_timing_gate_pass"])
        self.assertFalse(neg["strict_timing_gate_pass"])
        self.assertFalse(unconstrained["strict_timing_gate_pass"])
        self.assertFalse(uncovered["strict_timing_gate_pass"])

    def test_10_fresh_three_generation_namespaces_and_no_scope_promotion(self):
        m = self.module
        for path in (m.M1288_CANONICAL, m.M1288_WORK, m.M1288_ATTEMPT,
                     m.M1302_CANONICAL, m.M1302_WORK, m.M1302_ATTEMPT,
                     m.CANONICAL, m.WORK, m.ATTEMPT):
            with self.subTest(path=path.name):
                self.assertFalse(os.path.lexists(str(path)))
        source = WRAPPER.read_text(); orchestrator = ORCHESTRATOR.read_text()
        self.assertIn("exec 9</usr/libexec/platform-python3.6", source)
        self.assertIn('m1311_sha "/proc/${BASHPID}/fd/9"', source)
        order = [orchestrator.index("validate_admission(data, args)"),
                 orchestrator.index("first_collisions = collisions()"),
                 orchestrator.index("mem = meminfo()"),
                 orchestrator.index("license_run = subprocess.run("),
                 orchestrator.index("ATTEMPT.mkdir()"),
                 orchestrator.index('subprocess.run(["/usr/bin/bash", str(M1288_RUNNER)]')]
        self.assertEqual(order, sorted(order))
        self.assertEqual(sha(DOC359), PINS[DOC359])


if __name__ == "__main__":
    unittest.main(verbosity=2)
