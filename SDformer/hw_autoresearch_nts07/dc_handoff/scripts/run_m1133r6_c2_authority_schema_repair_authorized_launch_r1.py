#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1135r6 zero-argument launcher for the frozen M1133r6 C2 engine.

SOURCE ONLY until a different author seals M1136r6.  This wrapper accepts no
caller-selected path, authority, digest, argument, or environment value.  It
can invoke exactly one pinned engine child and never retries.  This authoring
step does not launch the wrapper or execute EDA/VCS.
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
ENGINE = HERE / "m1133r6_c2_authority_schema_repair_engine_source_r1.py"
ENGINE_SHA256 = "1f8a190d7d1c8b7804e7302c8b6a38c30a49df466b6394a82e8f0cf4cec2ee40"
ENGINE_CONTRACT = HW / "contracts/m1133r6_c2_authority_schema_repair_engine_source_contract_r1_20260830.json"
ENGINE_CONTRACT_ID = (
    "4dc16ffccb3c4a145f69f565500d67407ca821304ee838f93659918055a3ac8a",
    "bfd415d8540c2cb44b66683e127d66b7f3444b70840423ca4c66cf51f58e5ec7",
    "82b6d6a6568fc8fc95f1a1b7b6bf05690e06e064a143de41eadfa0e76ac9b849",
)
ENGINE_AUTHOR = HW / "reviews/m1133r6_c2_authority_schema_repair_engine_author_receipt_r1_20260830"
ENGINE_AUTHOR_ID = (
    "8fb36c424903047227a05c059ef3435e8a7769dd261245362c15292e22fe0777",
    "8412b78900050b0bee3556244d4283c1a84847946f7a0622d6deada1c5473b04",
    "5b2e0a659992c006d5caee72f5bcd72fd28dfdc07266d7edd2c814f1bc4a3b68",
)
M1121 = HW / "reviews/m1121_m1112r3_c2_dc_invocation_failure_audit_r1_20260830"
M1121_ID = (
    "910bdf5733a2287fa17ef6186f4814ef2c40b1216e4c6dc7378026b9a9cff525",
    "ac977fb671794a7efffbadcd7cd9f23b6f1185dad15fa2e27d69ee69f1390dcf",
    "dc0135b61750134c37b6e3eba47350a0d9838c9ed0a07ca5ecab3bb93c3ff828",
)
M1132R5_STOP = HW / "reviews/m1132r5_m1129r5_c2_dc_selector_launch_hammer_r1_20260830"
M1132R5_STOP_ID = (
    "87590e26217bb13a93fbf1e546597f160c706842ec71711e01a90e994c0887bd",
    "b55337586ad012ea4fafede034e41492b846e9298af5768fec154c5a0e565144",
    "bc073b90787189710986381b74c18b9a3afbe4ccd2f7969e85b596d3df1adf48",
)
M1134R6 = HW / "reviews/m1134r6_m1133r6_c2_authority_schema_engine_hammer_r1_20260830"
M1134R6_ID = (
    "3042ef28e939349044da57e5dd5a39bc3624bac4f66b1e1d0c42e773a0fe8855",
    "fd0cfb8751f015388160d643f380375c25085ef2201b7c060d7b4b5cd67552a4",
    "7c61ff53aaee7711fda0f79fd3a3bc9d99decfc9b7fbda377f22caf29fa72226",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")
LICENSE_FILE_SHA256 = "fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490"
SNPSLMD_LICENSE_FILE = "27030@ic.ismd-nemo"

ATTEMPT = HW / "results/.m1133r6_c2_authority_schema_repair_dc_mapped_vcs_attempt_consumed"
RESULT = HW / "results/m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830"
WORK_GLOB = ".m1133r6_c2_authority_schema_repair_dc_mapped_vcs_work.*"
FAILURE_GLOB = "m1133r6_c2_authority_schema_repair_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*"
LOCK = Path("/tmp/m1133r6_c2_authority_schema_repair_eda.lock")
R5_ATTEMPT = HW / "results/.m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
R5_RESULT = HW / "results/m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830"
R5_WORK_GLOB = ".m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_work.*"
R5_FAILURE_GLOB = "m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*"
R5_LOCK = Path("/tmp/m1129r5_c2_dc_selector_async_observation_eda.lock")
MIN_MEM_AVAILABLE_KIB = 8 * 1024 * 1024
MIN_COMMIT_HEADROOM_KIB = 8 * 1024 * 1024
EDA_PROCESS_NAMES = (
    "vcs", "vcs1", "vlogan", "dc_shell", "dc_shell-t", "fm_shell",
    "pt_shell", "simv", "common_shell_exec", "common_shell_exe",
)
ROOT_ENV = {
    "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PATH": "/usr/bin:/bin",
    "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
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
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


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
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() ==
            [identity[0], path.relative_to(HW).as_posix()], "double-seal side content")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], side.relative_to(HW).as_posix()], "double-seal outer content")


def verify_flat(directory: Path, identity: tuple[str, str, str], status: str) -> dict[str, Any]:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(),
            "sealed authority directory drift")
    review = directory / "review.json"; manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(review, identity[0]); verify_regular(manifest, identity[1])
    verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() == [identity[1], "SHA256SUMS"],
            "sealed authority outer content")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*")
        relative = Path(name)
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest) and
                name not in expected and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "unsafe sealed authority manifest")
        expected[name] = digest
    actual: set[str] = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed authority symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed authority special member")
    require(actual == set(expected), "sealed authority exact member set")
    for name, digest in expected.items():
        verify_regular(directory / name, digest)
    value = strict_json(review)
    require(value.get("status") == status, "sealed authority status drift")
    return value


def namespace_absent(path: Path) -> bool:
    return not path.exists() and not path.is_symlink()


def validate_hardcoded_authorities(enforce_runtime: bool) -> dict[str, Any]:
    if enforce_runtime:
        require(len(sys.argv) == 1, "M1135r6 accepts zero arguments")
        require(Path(sys.executable) == PYTHON and tuple(sys.version_info[:3]) == (3, 10, 18),
                "M1135r6 requires pinned Python 3.10.18")
        require(os.environ == ROOT_ENV, "M1135r6 requires exact env -i root environment")
    verify_regular(PYTHON, PYTHON_SHA256); verify_regular(ENGINE, ENGINE_SHA256)
    verify_regular(DOCS359, DOCS359_SHA256); verify_regular(LICENSE_FILE, LICENSE_FILE_SHA256)
    verify_double(ENGINE_CONTRACT, ENGINE_CONTRACT_ID)
    engine_author = verify_flat(ENGINE_AUTHOR, ENGINE_AUTHOR_ID,
        "PASS_M1133R6_AUTHORITY_SCHEMA_REPAIR_ENGINE_AUTHOR_RECEIPT__M1134R6_REQUIRED__NO_EDA")
    m1121 = verify_flat(M1121, M1121_ID,
        "PASS_M1121_FAILURE_AUDIT__M1112R3_DO_NOT_RETRY__ADDITIVE_R4_INVOCATION_SELECTOR_REPAIR_ONLY")
    stopped = verify_flat(M1132R5_STOP, M1132R5_STOP_ID,
        "FAIL_M1132R5_M1129R5_POSTSEAL_FUTURE_AUTHORITY__ADDITIVE_R6_REQUIRED__NO_LAUNCH")
    hammer = verify_flat(M1134R6, M1134R6_ID,
        "PASS_M1134R6_M1133R6_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA")
    contract = strict_json(ENGINE_CONTRACT)
    require(contract["future_chain"]["launch_receipt_contains_future_m1136r6_outer"] is False and
            contract["future_chain"]["m1136r6_outer_discovery"] ==
                "verify_flat_self_consistent at authorized execution" and
            contract["future_chain"]["placeholder_or_hash_fixed_point_allowed"] is False,
            "acyclic future M1136r6 discovery contract drift")
    require(engine_author["identity"]["engine_sha256"] == ENGINE_SHA256 and
            engine_author["identity"]["contract_outer_seal_file_sha256"] == ENGINE_CONTRACT_ID[2] and
            m1121["status"].startswith("PASS_M1121_FAILURE_AUDIT") and
            stopped["authorization"]["r5_command_withdrawn"] is True and
            stopped["execution"]["r5_namespace_created"] is False and
            hammer["identity"] == {
                "engine_sha256": ENGINE_SHA256,
                "contract_sha256": ENGINE_CONTRACT_ID[0],
                "author_receipt_outer_seal_file_sha256": ENGINE_AUTHOR_ID[2]},
            "M1121/M1132r5 STOP/M1134r6 exact authority drift")
    return {
        "status": "PASS_M1135R6_PREEXISTING_HARDCODED_AUTHORITIES",
        "engine_sha256": ENGINE_SHA256,
        "engine_contract_outer_seal_file_sha256": ENGINE_CONTRACT_ID[2],
        "engine_author_receipt_outer_seal_file_sha256": ENGINE_AUTHOR_ID[2],
        "m1121_outer_seal_file_sha256": M1121_ID[2],
        "m1132r5_stop_outer_seal_file_sha256": M1132R5_STOP_ID[2],
        "m1134r6_outer_seal_file_sha256": M1134R6_ID[2],
        "future_m1136r6_discovery_acyclic": True,
    }


def read_meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1); fields = raw.split()
        if fields and fields[0].isdigit():
            values[key] = int(fields[0])
    require(all(key in values for key in ("MemAvailable", "CommitLimit", "Committed_AS")),
            "meminfo schema drift")
    return values


def collision_gate() -> list[str]:
    uid = str(os.getuid()); collisions: list[str] = []
    for name in EDA_PROCESS_NAMES:
        completed = subprocess.run(["/usr/bin/pgrep", "-u", uid, "-x", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            close_fds=True, check=False)
        if completed.returncode == 0:
            collisions.append(name)
        else:
            require(completed.returncode == 1, "pgrep diagnostic failure")
    require(not collisions, "EDA collision: " + ",".join(collisions))
    return collisions


def namespace_resource_gate() -> dict[str, Any]:
    require(namespace_absent(R5_ATTEMPT) and namespace_absent(R5_RESULT) and
            namespace_absent(R5_LOCK) and
            not any((HW / "results").glob(R5_WORK_GLOB)) and
            not any((HW / "results").glob(R5_FAILURE_GLOB)),
            "withdrawn r5 namespace must remain absent")
    require(namespace_absent(ATTEMPT), "M1133r6 attempt namespace not fresh")
    require(namespace_absent(RESULT), "M1133r6 result namespace not fresh")
    require(namespace_absent(LOCK), "M1133r6 lock namespace not fresh")
    require(not any((HW / "results").glob(WORK_GLOB)), "M1133r6 stale work namespace")
    require(not any((HW / "results").glob(FAILURE_GLOB)), "M1133r6 prior failure forbids retry")
    collisions = collision_gate(); info = read_meminfo()
    headroom = info["CommitLimit"] - info["Committed_AS"]
    require(info["MemAvailable"] >= MIN_MEM_AVAILABLE_KIB, "insufficient MemAvailable")
    require(headroom >= MIN_COMMIT_HEADROOM_KIB, "insufficient commit headroom")
    return {"status": "PASS_M1135R6_FRESH_R6_AND_STOPPED_R5_RESOURCE_COLLISION_GATE",
            "mem_available_kib": info["MemAvailable"],
            "commit_headroom_kib": headroom, "eda_collisions": collisions}


def clean_child_environment(private_home: Path) -> dict[str, str]:
    """Construct constants only; no caller environment value is consulted."""
    return {"HOME": str(private_home), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
            "PATH": "/usr/bin:/bin", "TMPDIR": "/tmp", "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "SNPSLMD_LICENSE_FILE": SNPSLMD_LICENSE_FILE,
            "LM_LICENSE_FILE": str(LICENSE_FILE)}


def main() -> int:
    validate_hardcoded_authorities(enforce_runtime=True)
    namespace_resource_gate()
    private_home = Path(tempfile.mkdtemp(prefix="m1133r6_c2_home.", dir="/tmp"))
    private_home.chmod(0o700)
    try:
        completed = subprocess.run(
            [str(PYTHON), "-I", str(ENGINE), "--authorized-launch"],
            cwd=str(HW), env=clean_child_environment(private_home),
            close_fds=True, check=False)
        return completed.returncode
    finally:
        require(private_home.parent == Path("/tmp") and
                private_home.name.startswith("m1133r6_c2_home.") and
                private_home.is_dir() and not private_home.is_symlink(),
                "private HOME cleanup identity drift")
        shutil.rmtree(private_home)


if __name__ == "__main__":
    raise SystemExit(main())
