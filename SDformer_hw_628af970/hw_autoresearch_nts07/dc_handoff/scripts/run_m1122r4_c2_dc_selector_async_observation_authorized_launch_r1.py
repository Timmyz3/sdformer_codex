#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1124r4 fixed zero-argument launcher for the M1122r4 C2 engine.

SOURCE ONLY until a different author seals the M1125r4 final launch hammer.
The wrapper accepts no caller-selected path, digest, metric, authority, or
environment value.  It invokes exactly one pinned engine process and never
retries.  The engine, not this authoring step, owns the sole future attempt.
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
ENGINE = HERE / "m1122r4_c2_dc_selector_async_observation_engine_source_r1.py"
ENGINE_SHA256 = "f278052d251af0c2d150872391306c2f3922049ca04c7df2a0d9d3d074b55007"
ENGINE_CONTRACT = HW / "contracts/m1122r4_c2_dc_selector_async_observation_source_contract_r1_20260830.json"
ENGINE_CONTRACT_ID = (
    "cee4ddc66c244bf4e19e2ce193573b55bf4fd973c7c1bcd53d609d77a9b8cea3",
    "0a1ed1ad054b8a778c17c71eb9fdd82d5df943d77989280bd43660736795b617",
    "373e6b86bdfdf94584f289f8c0fc1af1dc9a7ea19be656cba93159b3efb06987",
)
ENGINE_AUTHOR_RECEIPT = HW / "reviews/m1122r4_c2_dc_selector_engine_author_receipt_r1_20260830"
ENGINE_AUTHOR_RECEIPT_ID = (
    "c8614bdf7b1ae3c1c6df330ae7ab7f6dffc1fd0e59bd409d995caace0b168b7e",
    "bebb65e161a0ff39d2eda1989345d52f183dd5a282fd185ea9acab1f5d2626d7",
    "c36311a8ac2d5b425c2e3b45a7fee665d9f93cd07e06fd4a095746d7c7c99c9b",
)
M1121 = HW / "reviews/m1121_m1112r3_c2_dc_invocation_failure_audit_r1_20260830"
M1121_ID = (
    "910bdf5733a2287fa17ef6186f4814ef2c40b1216e4c6dc7378026b9a9cff525",
    "ac977fb671794a7efffbadcd7cd9f23b6f1185dad15fa2e27d69ee69f1390dcf",
    "dc0135b61750134c37b6e3eba47350a0d9838c9ed0a07ca5ecab3bb93c3ff828",
)
M1123R4 = HW / "reviews/m1123r4_m1122r4_c2_dc_selector_engine_hammer_r1_20260830"
M1123R4_ID = (
    "95aee43c7616073fdeabe42716c8e0cd99461e4b4e8420a283d92b277c82b1b6",
    "9f2361ba5e99d520d7704f6d296988fe80159ed35801253bfd6232c3957dd4fc",
    "c90174a5a981668d3d27a65c9c73d27370e96039bccaefc68f97a775c9640d5a",
)
M1123R4_HAMMER_SHA256 = "cbbfd1c4457929951ebb5ffb75c5a482e33152a3e17d144c0281d8cd0cc6f373"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")
LICENSE_FILE_SHA256 = "fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490"
SNPSLMD_LICENSE_FILE = "27030@ic.ismd-nemo"

ATTEMPT = HW / "results/.m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
RESULT = HW / "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830"
WORK_GLOB = ".m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_work.*"
FAILURE_GLOB = "m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*"
LOCK = Path("/tmp/m1122r4_c2_dc_selector_async_observation_eda.lock")
MIN_MEM_AVAILABLE_KIB = 8 * 1024 * 1024
MIN_COMMIT_HEADROOM_KIB = 8 * 1024 * 1024
EDA_PROCESS_NAMES = (
    "vcs", "vcs1", "vlogan", "dc_shell", "dc_shell-t", "fm_shell",
    "pt_shell", "simv", "common_shell_exec", "common_shell_exe",
)
ROOT_ENV = {
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PATH": "/usr/bin:/bin",
    "TMPDIR": "/tmp",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
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
        require(len(sys.argv) == 1, "M1124r4 accepts zero arguments")
        require(Path(sys.executable) == PYTHON and
                tuple(sys.version_info[:3]) == (3, 10, 18),
                "M1124r4 requires pinned Python 3.10.18")
        require(os.environ == ROOT_ENV,
                "M1124r4 requires exact env -i root environment")
    verify_regular(PYTHON, PYTHON_SHA256)
    verify_regular(ENGINE, ENGINE_SHA256)
    verify_regular(DOCS359, DOCS359_SHA256)
    verify_regular(LICENSE_FILE, LICENSE_FILE_SHA256)
    verify_double(ENGINE_CONTRACT, ENGINE_CONTRACT_ID)
    verify_flat(
        ENGINE_AUTHOR_RECEIPT, ENGINE_AUTHOR_RECEIPT_ID,
        "PASS_M1122R4_DC_SELECTOR_ENGINE_SOURCE_AUTHOR_RECEIPT__M1123R4_REQUIRED__NO_EDA",
    )
    verify_flat(
        M1121, M1121_ID,
        "PASS_M1121_FAILURE_AUDIT__M1112R3_DO_NOT_RETRY__ADDITIVE_R4_INVOCATION_SELECTOR_REPAIR_ONLY",
    )
    verify_flat(
        M1123R4, M1123R4_ID,
        "PASS_M1123R4_M1122R4_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA",
    )
    verify_regular(M1123R4 / "independent_engine_hammer.py", M1123R4_HAMMER_SHA256)
    contract = strict_json(ENGINE_CONTRACT)
    review = strict_json(M1123R4 / "review.json")
    require(
        contract["future_chain"]["launch_receipt_contains_future_m1125r4_outer"] is False and
        contract["future_chain"]["placeholder_or_hash_fixed_point_allowed"] is False and
        contract["future_chain"]["m1125r4_outer_discovery"] ==
            "verify_flat_self_consistent at authorized execution" and
        review["authorization"]["zero_argument_launcher_authoring"] is True and
        review["authorization"]["launch"] is False,
        "M1122r4 acyclic launcher authority drift",
    )
    return {
        "status": "PASS_M1124R4_PREEXISTING_HARDCODED_AUTHORITIES",
        "engine_sha256": ENGINE_SHA256,
        "engine_contract_outer_seal_file_sha256": ENGINE_CONTRACT_ID[2],
        "engine_author_receipt_outer_seal_file_sha256": ENGINE_AUTHOR_RECEIPT_ID[2],
        "m1121_outer_seal_file_sha256": M1121_ID[2],
        "m1123r4_outer_seal_file_sha256": M1123R4_ID[2],
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
            "M1122r4 attempt namespace not fresh")
    require(not RESULT.exists() and not RESULT.is_symlink(),
            "M1122r4 result namespace not fresh")
    require(not LOCK.exists() and not LOCK.is_symlink(),
            "M1122r4 lock namespace not fresh")
    require(not any((HW / "results").glob(WORK_GLOB)),
            "M1122r4 stale work namespace")
    require(not any((HW / "results").glob(FAILURE_GLOB)),
            "M1122r4 prior failure/quarantine forbids retry")
    collisions = collision_gate()
    info = read_meminfo()
    headroom = info["CommitLimit"] - info["Committed_AS"]
    require(info["MemAvailable"] >= MIN_MEM_AVAILABLE_KIB,
            "insufficient MemAvailable")
    require(headroom >= MIN_COMMIT_HEADROOM_KIB,
            "insufficient commit headroom")
    return {
        "status": "PASS_M1124R4_FRESH_NAMESPACE_RESOURCE_COLLISION_GATE",
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
    private_home = Path(tempfile.mkdtemp(prefix="m1122r4_c2_home.", dir="/tmp"))
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
                private_home.name.startswith("m1122r4_c2_home.") and
                private_home.is_dir() and not private_home.is_symlink(),
                "private HOME cleanup identity drift")
        shutil.rmtree(private_home)


if __name__ == "__main__":
    raise SystemExit(main())
