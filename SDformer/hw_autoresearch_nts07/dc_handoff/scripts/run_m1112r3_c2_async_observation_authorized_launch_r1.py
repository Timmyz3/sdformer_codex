#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1119r3 fixed zero-argument launcher for the M1112r3 C2 engine.

SOURCE ONLY until a different author seals the M1118r3 final launch hammer.
This wrapper accepts no caller-selected path, digest, metric, authority, or
environment value.  It launches exactly one pinned engine process; that engine
consumes the only attempt before DC and performs exactly one DC -> mapped-VCS
flow.  This source must never be imported as production evidence.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
ENGINE = HERE / "m1112r3_c2_async_observation_authorized_engine_source_r1.py"
ENGINE_SHA256 = "48616ebde16e07b132bbb2e686bd34a9f18270d0bc0693ab0ee956beb60f02be"
ENGINE_CONTRACT = HW / "contracts/m1112r3_c2_async_observation_shadow_source_contract_r1_20260830.json"
ENGINE_CONTRACT_ID = (
    "92117a56e50a946d674c82ce9fc084548b480df139e0a4e5a9b4aed391292bef",
    "cfe40a1d11bcdf77cd4ac33e149381b202c57cc8edc22cb1131559fba8e412fd",
    "ddda54a99c1638f39c828faf75775a7f5c0dae975ee26f7b251cbafa926906cf",
)
ENGINE_AUTHOR_RECEIPT = HW / "reviews/m1112r3_c2_launch_chain_source_receipt_r1_20260830"
ENGINE_AUTHOR_RECEIPT_ID = (
    "0cf65f1015e45ae70fb352bb86518d98784e3f436de4648bdf0d9c726efbf69b",
    "e30b75f496507f1d34ebf25fa6cdc9d5087adfc758bb5ce0b99e9a35cb8d3e69",
    "7f9d0205b9ba2f53fd642b05b0cd4faf9aa3e8e5bf14a6047c23ac6fba3ea7ff",
)
M1117R3 = HW / "reviews/m1117r3_m1112r3_c2_async_observation_engine_hammer_r1_20260830"
M1117R3_ID = (
    "cc35e5a21f148f8da7f04ca71cd2385da46a2a37af5f0387fdd0c3f0b3d7e12c",
    "b1f3b03f3ecf1f7a8fc2b38f5d50dec3a452cc30967cd00fe851d183b32fa1b3",
    "41b4950ac4e1a175379e4d0ae34fd5335e339e320f716cd5e2b073dc9aa00d82",
)
M1117R3_HAMMER_SHA256 = "d54d3d1c412c8c0bb4f89a49e3dc65cd0ec3a48f133b4cd561e96089e29181ea"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")
LICENSE_FILE_SHA256 = "fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490"
SNPSLMD_LICENSE_FILE = "27030@ic.ismd-nemo"

ATTEMPT = HW / "results/.m1112r3_c2_async_observation_dc_mapped_vcs_attempt_consumed"
RESULT = HW / "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830"
WORK_GLOB = ".m1112r3_c2_async_observation_dc_mapped_vcs_work.*"
FAILURE_GLOB = "m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*"
LOCK = Path("/tmp/m1112r3_c2_async_observation_eda.lock")
MIN_MEM_AVAILABLE_KIB = 8 * 1024 * 1024
MIN_COMMIT_HEADROOM_KIB = 8 * 1024 * 1024
EDA_PROCESS_NAMES = (
    "vcs", "vcs1", "vlogan", "dc_shell", "dc_shell-t", "fm_shell",
    "pt_shell", "simv",
)
ROOT_ENV_KEYS = {
    "LANG", "LC_ALL", "PATH", "TMPDIR", "PYTHONNOUSERSITE",
    "PYTHONDONTWRITEBYTECODE",
}


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(
        path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)),
    )


def verify_regular(path: Path, expected: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError("missing pinned regular file: " + str(path)) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "pinned file is not direct regular: " + str(path))
    require(sha256(path) == expected, "pinned file identity drift: " + str(path))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0])
    verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() ==
            [identity[0], path.relative_to(HW).as_posix()],
            "double-seal sidecar content drift")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], side.relative_to(HW).as_posix()],
            "double-seal outer content drift")


def verify_flat(directory: Path, identity: tuple[str, str, str], status: str) -> None:
    try:
        mode = directory.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError("sealed authority absent: " + str(directory)) from error
    require(stat.S_ISDIR(mode) and not directory.is_symlink(),
            "sealed authority is not a direct directory")
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(review, identity[0])
    verify_regular(manifest, identity[1])
    verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], "SHA256SUMS"], "sealed authority outer content drift")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "sealed authority malformed manifest")
        relative = fields[1].lstrip("*")
        relpath = Path(relative)
        require(relative not in expected and relative == relpath.as_posix() and
                not relpath.is_absolute() and ".." not in relpath.parts,
                "sealed authority unsafe/duplicate member")
        expected[relative] = fields[0]
    actual: set[str] = set()
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        if relative in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        member_mode = member.lstat().st_mode
        require(not stat.S_ISLNK(member_mode), "sealed authority live symlink")
        if stat.S_ISREG(member_mode):
            actual.add(relative)
        else:
            require(stat.S_ISDIR(member_mode), "sealed authority special member")
    require(actual == set(expected), "sealed authority exact member-set drift")
    for relative, expected_sha in expected.items():
        verify_regular(directory / relative, expected_sha)
    require(strict_json(review).get("status") == status,
            "sealed authority status drift")


def validate_hardcoded_authorities(enforce_runtime: bool) -> dict[str, Any]:
    if enforce_runtime:
        require(len(sys.argv) == 1, "M1119r3 accepts zero arguments")
        require(Path(sys.executable) == PYTHON and
                tuple(sys.version_info[:3]) == (3, 10, 18),
                "M1119r3 requires pinned Python 3.10.18")
        require(set(os.environ) == ROOT_ENV_KEYS,
                "M1119r3 requires exact env -i root environment")
        require(os.environ == {
            "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin",
            "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }, "M1119r3 root environment value drift")
    verify_regular(PYTHON, PYTHON_SHA256)
    verify_regular(ENGINE, ENGINE_SHA256)
    verify_regular(DOCS359, DOCS359_SHA256)
    verify_regular(LICENSE_FILE, LICENSE_FILE_SHA256)
    verify_double(ENGINE_CONTRACT, ENGINE_CONTRACT_ID)
    verify_flat(
        ENGINE_AUTHOR_RECEIPT, ENGINE_AUTHOR_RECEIPT_ID,
        "PASS_M1112R3_ACYCLIC_LAUNCH_CHAIN_SOURCE_AUTHOR_RECEIPT__M1117R3_REQUIRED__NO_EDA",
    )
    verify_flat(
        M1117R3, M1117R3_ID,
        "PASS_M1117R3_M1112R3_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA",
    )
    verify_regular(M1117R3 / "independent_engine_hammer.py", M1117R3_HAMMER_SHA256)
    engine_contract = strict_json(ENGINE_CONTRACT)
    engine_review = strict_json(M1117R3 / "review.json")
    require(engine_contract["future_chain"]["launch_receipt_contains_future_m1118r3_outer"] is False and
            engine_contract["future_chain"]["placeholder_or_hash_fixed_point_allowed"] is False and
            engine_review["acyclic_proof"]["sha256_fixed_point_required"] is False and
            engine_review["authorization"]["different_author_launcher_authoring"] is True,
            "M1112r3 acyclic authoring authority drift")
    return {
        "status": "PASS_M1119R3_PREEXISTING_HARDCODED_AUTHORITIES",
        "engine_sha256": ENGINE_SHA256,
        "engine_contract_outer_seal_file_sha256": ENGINE_CONTRACT_ID[2],
        "engine_author_receipt_outer_seal_file_sha256": ENGINE_AUTHOR_RECEIPT_ID[2],
        "m1117r3_outer_seal_file_sha256": M1117R3_ID[2],
    }


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        fields = raw.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0])
    require(all(key in values for key in
                ("MemAvailable", "CommitLimit", "Committed_AS")),
            "meminfo schema drift")
    return values


def collision_gate() -> list[str]:
    uid = str(os.getuid())
    collisions: list[str] = []
    for name in EDA_PROCESS_NAMES:
        completed = subprocess.run(
            ["/usr/bin/pgrep", "-u", uid, "-x", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True, check=False,
        )
        if completed.returncode == 0:
            collisions.append(name)
        else:
            require(completed.returncode == 1, "pgrep diagnostic failure")
    require(not collisions, "EDA collision: " + ",".join(collisions))
    return collisions


def namespace_resource_gate() -> dict[str, Any]:
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink(),
            "M1112r3 attempt namespace not fresh")
    require(not RESULT.exists() and not RESULT.is_symlink(),
            "M1112r3 result namespace not fresh")
    require(not LOCK.exists() and not LOCK.is_symlink(),
            "M1112r3 lock namespace not fresh")
    require(not any((HW / "results").glob(WORK_GLOB)),
            "M1112r3 stale work namespace")
    require(not any((HW / "results").glob(FAILURE_GLOB)),
            "M1112r3 prior failure/quarantine forbids retry")
    collisions = collision_gate()
    info = read_meminfo()
    headroom = info["CommitLimit"] - info["Committed_AS"]
    require(info["MemAvailable"] >= MIN_MEM_AVAILABLE_KIB,
            "insufficient MemAvailable")
    require(headroom >= MIN_COMMIT_HEADROOM_KIB,
            "insufficient commit headroom")
    return {
        "status": "PASS_M1119R3_FRESH_NAMESPACE_RESOURCE_COLLISION_GATE",
        "mem_available_kib": info["MemAvailable"],
        "commit_headroom_kib": headroom,
        "eda_collisions": collisions,
    }


def clean_child_environment(private_home: Path) -> dict[str, str]:
    """Construct constants only; no caller environment value is consulted."""
    return {
        "HOME": str(private_home),
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": "/tmp",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SNPSLMD_LICENSE_FILE": SNPSLMD_LICENSE_FILE,
        "LM_LICENSE_FILE": str(LICENSE_FILE),
    }


def main() -> int:
    validate_hardcoded_authorities(enforce_runtime=True)
    namespace_resource_gate()
    private_home = Path(tempfile.mkdtemp(prefix="m1112r3_c2_home.", dir="/tmp"))
    private_home.chmod(0o700)
    try:
        completed = subprocess.run(
            [str(PYTHON), "-I", str(ENGINE), "--authorized-launch"],
            cwd=str(HW), env=clean_child_environment(private_home),
            close_fds=True, check=False,
        )
        return completed.returncode
    finally:
        require(private_home.parent == Path("/tmp") and
                private_home.name.startswith("m1112r3_c2_home.") and
                private_home.is_dir() and not private_home.is_symlink(),
                "private HOME cleanup identity drift")
        shutil.rmtree(private_home)


if __name__ == "__main__":
    raise SystemExit(main())
