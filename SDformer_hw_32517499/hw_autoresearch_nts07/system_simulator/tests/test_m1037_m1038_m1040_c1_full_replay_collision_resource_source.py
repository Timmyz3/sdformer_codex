#!/opt/anaconda3/envs/pytorch310/bin/python3.10
from contextlib import contextmanager
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "system_simulator/scripts/run_m1040_m1016_c1_full_matched_address_replay_one_shot.sh"
RELEASE = HW / "contracts/m1038_m1037_m1016_c1_full_matched_address_replay_launch_release_r1_20260829.json"
CHECKER = HW / "system_simulator/scripts/check_m1037_m1038_m1040_c1_full_replay_collision_resource_source.py"
M1025_OUTER = "7004ab978588ebaed6b94e57c9c30bbaadb4c9502a57921dc1b1e40cfe7743ff"
M1036_OUTER = "476f0779ad32d40831dbcdaa5d4c223d7f6a50d9aecb196e63107ee4c1c8f5ae"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seal(directory):
    members = sorted(path for path in directory.rglob("*") if path.is_file() and
                     path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
    manifest = directory / "SHA256SUMS"
    manifest.write_text("".join(f"{sha(path)}  {path.relative_to(directory)}\n" for path in members))
    outer = directory / "SHA256SUMS.seal.sha256"
    outer.write_text(f"{sha(manifest)}  SHA256SUMS\n")
    return sha(outer)


@contextmanager
def sandbox(meminfo_text="CommitLimit: 100000000 kB\nCommitted_AS: 10000000 kB\nMemAvailable: 100000000 kB\n"):
    with tempfile.TemporaryDirectory(prefix="m1037_source_test_") as td:
        root = Path(td)
        runner = root / "runner.sh"
        release = root / "release.json"
        hammer = root / "m1039"
        hammer.mkdir()
        meminfo = root / "meminfo"
        meminfo.write_text(meminfo_text)
        lockfile = root / "global.lock"
        result, attempt = root / "result", root / "attempt"
        work, failure = root / "work.$$", root / "failure.$$.quarantine"
        false_sha = sha("/bin/false")
        text = RUNNER.read_text()
        replacements = {
            'hw_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"': f'hw_root="{HW}"',
            'engine="${hw_root}/system_simulator/scripts/run_m1016_c1_full_matched_address_replay.py"': 'engine="/bin/false"',
            'release="${hw_root}/contracts/m1038_m1037_m1016_c1_full_matched_address_replay_launch_release_r1_20260829.json"': f'release="{release}"',
            'release_hammer="${hw_root}/reviews/m1039_m1038_m1036_m1040_c1_full_replay_release_hammer_r1_20260829"': f'release_hammer="{hammer}"',
            'meminfo=/proc/meminfo': f'meminfo="{meminfo}"',
            'lockfile="${hw_root}/results/.c1_full_matched_address_replay_global.lock"': f'lockfile="{lockfile}"',
            'result="${hw_root}/results/m1040_m1016_c1_full_matched_address_replay_r1_20260829"': f'result="{result}"',
            'attempt="${hw_root}/results/.m1040_m1016_c1_full_matched_address_replay_attempt_consumed"': f'attempt="{attempt}"',
            'work="${hw_root}/results/.m1040_m1016_c1_full_matched_address_replay_work.$$"': f'work="{work}"',
            'failure="${hw_root}/results/m1040_m1016_c1_full_matched_address_replay_r1_20260829.failed_or_incomplete.$$.quarantine"': f'failure="{failure}"',
            'readonly expected_engine_sha=d505b5608641ae28a6b6c913c3779acf5e81e15fec436a0180c4c7e7ab6db4fa': f'readonly expected_engine_sha={false_sha}',
        }
        for old, new in replacements.items():
            if text.count(old) != 1:
                raise RuntimeError("patch anchor drift: " + old)
            text = text.replace(old, new)
        runner.write_text(text)
        runner.chmod(0o755)
        release_value = json.loads(RELEASE.read_text())
        release_value["engine_sha256"] = false_sha
        release_value["runner_sha256"] = sha(runner)
        release.write_text(json.dumps(release_value, indent=2, sort_keys=True) + "\n")
        sidecar = Path(str(release) + ".sha256")
        sidecar.write_text(f"{sha(release)}  {release.name}\n")
        Path(str(release) + ".sha256.seal.sha256").write_text(
            f"{sha(sidecar)}  {sidecar.name}\n")
        hammer_review = {
            "status": "PASS_M1039_M1038_M1036_M1040_C1_FULL_REPLAY_RELEASE_HAMMER",
            "identity": {"m1038_release_sha256": sha(release),
                         "m1040_runner_sha256": sha(runner),
                         "m1025_outer_seal_file_sha256": M1025_OUTER,
                         "m1036_outer_seal_file_sha256": M1036_OUTER}}
        (hammer / "review.json").write_text(json.dumps(hammer_review) + "\n")
        hammer_outer = seal(hammer)
        env = {"PATH": "/usr/bin:/bin",
               "M1040_EXPECTED_RUNNER_SHA256": sha(runner),
               "M1040_EXPECTED_M1025_OUTER_SHA256": M1025_OUTER,
               "M1040_EXPECTED_M1036_OUTER_SHA256": M1036_OUTER,
               "M1040_EXPECTED_M1039_OUTER_SHA256": hammer_outer}
        yield {"root": root, "runner": runner, "release": release, "hammer": hammer,
               "meminfo": meminfo, "lockfile": lockfile, "result": result,
               "attempt": attempt, "env": env}


def run(box):
    return subprocess.run([str(box["runner"])], env=box["env"],
                          universal_newlines=True, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, timeout=20)


class TestM1037Source(unittest.TestCase):
    def test_static_checker(self):
        proc = subprocess.run(["/opt/anaconda3/envs/pytorch310/bin/python3.10", str(CHECKER)],
                              universal_newlines=True, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, check=True)
        self.assertIn("PASS_M1037_M1038_M1040_COLLISION_RESOURCE_SOURCE", proc.stdout)

    def test_each_exact_process_collision_is_pre_attempt(self):
        for name in ("vcs1", "vlogan", "dc_shell", "dc_shell-t", "fm_shell", "pt_shell"):
            with self.subTest(name=name), sandbox() as box:
                tool_dir = box["root"] / "tool"
                tool_dir.mkdir()
                fake = tool_dir / name
                fake.symlink_to("/usr/bin/sleep")
                blocker = subprocess.Popen([str(fake), "5"])
                try:
                    for _ in range(50):
                        if subprocess.run(["/usr/bin/pgrep", "-x", name],
                                          stdout=subprocess.DEVNULL).returncode == 0:
                            break
                        time.sleep(0.01)
                    proc = run(box)
                    self.assertNotEqual(proc.returncode, 0)
                    self.assertIn("process collision: " + name, proc.stderr)
                    self.assertFalse(box["attempt"].exists())
                finally:
                    blocker.terminate()
                    try:
                        blocker.wait(timeout=1)
                    except subprocess.TimeoutExpired:
                        blocker.kill(); blocker.wait(timeout=1)

    def test_lock_collision_is_pre_attempt(self):
        with sandbox() as box:
            fd = os.open(box["lockfile"], os.O_CREAT | os.O_WRONLY, 0o664)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                proc = run(box)
                self.assertIn("C1 full replay lock collision", proc.stderr)
                self.assertFalse(box["attempt"].exists())
            finally:
                os.close(fd)

    def test_low_commit_headroom_is_pre_attempt(self):
        with sandbox("CommitLimit: 20000000 kB\nCommitted_AS: 10000000 kB\nMemAvailable: 100000000 kB\n") as box:
            proc = run(box)
            self.assertIn("CommitLimit-Committed_AS below 16GiB floor", proc.stderr)
            self.assertFalse(box["attempt"].exists())

    def test_low_memavailable_is_pre_attempt(self):
        with sandbox("CommitLimit: 100000000 kB\nCommitted_AS: 10000000 kB\nMemAvailable: 10000000 kB\n") as box:
            proc = run(box)
            self.assertIn("MemAvailable below 16GiB floor", proc.stderr)
            self.assertFalse(box["attempt"].exists())

    def test_unrelated_cpu_name_is_not_blacklisted(self):
        with sandbox() as box:
            tool_dir = box["root"] / "tool"
            tool_dir.mkdir()
            fake = tool_dir / "unrelated_cpu"
            fake.symlink_to("/usr/bin/sleep")
            blocker = subprocess.Popen([str(fake), "5"])
            try:
                proc = run(box)
                self.assertNotEqual(proc.returncode, 0)  # /bin/false sandbox engine
                self.assertTrue(box["attempt"].exists())
                self.assertNotIn("process collision", proc.stderr)
                self.assertNotIn("below 16GiB", proc.stderr)
            finally:
                blocker.terminate()
                try:
                    blocker.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    blocker.kill(); blocker.wait(timeout=1)


if __name__ == "__main__":
    unittest.main()
