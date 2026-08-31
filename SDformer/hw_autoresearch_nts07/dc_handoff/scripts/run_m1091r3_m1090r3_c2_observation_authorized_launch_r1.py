#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1094C2 fixed zero-argument launcher for the M1091r3 C2 engine.

SOURCE ONLY.  M1096r2 must independently hammer this exact launcher and its
double-sealed launch receipt before root may execute the unique command.  This
file never accepts a path, digest, metric, or authority from argv or the caller
environment.  The child receives a newly constructed, minimal environment.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
PYTHON = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")
PYTHON_SHA256 = "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115"
ENGINE = HERE / "m1091r3_m1090r3_c2_observation_authorized_engine_r1.py"
ENGINE_SHA256 = "41b7899083152f8099acac759109a8eb22c381cb6a17506ae85e6666656daf04"
SOURCE_RECEIPT = HW / "reviews/m1090r3_m1091r3_c2_observation_fixed_history_source_receipt_r1_20260830"
SOURCE_RECEIPT_OUTER_SHA256 = "8bc6f725ef0ec7055441afafa2c0bd5c5ba54620c4354feaf2a6763fbabedd9e"
M1093R2 = HW / "reviews/m1093r2_m1090r3_m1091r3_c2_observation_engine_hammer_r1_20260830"
M1093R2_OUTER_SHA256 = "d6fa5ecb89342188586fb179d9dcaa1018078b4f3db6c609f6f1fd1b0559f9cc"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
LICENSE_FILE = Path("/opt/synopsys/Synopsys.dat")
LICENSE_FILE_SHA256 = "fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490"
SNPSLMD_LICENSE_FILE = "27030@ic.ismd-nemo"


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
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=lambda token: (_ for _ in ()).throw(
            RuntimeError("nonfinite JSON: " + token)
        ),
    )


def verify_regular(path: Path, expected: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise RuntimeError("missing pinned regular file: " + str(path)) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "pinned file is not direct regular: " + str(path))
    require(sha256(path) == expected, "pinned file identity drift: " + str(path))


def verify_flat(directory: Path, expected_outer: str, expected_status: str) -> None:
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed authority absent or symlink: " + str(directory))
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    verify_regular(outer, expected_outer)
    require(outer.read_text(encoding="utf-8").split() ==
            [sha256(manifest), "SHA256SUMS"], "sealed authority outer drift")
    seen: set[str] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        require(relative not in seen and not Path(relative).is_absolute() and
                ".." not in Path(relative).parts, "sealed authority path drift")
        verify_regular(directory / relative, expected)
        seen.add(relative)
    review = strict_json(directory / "review.json")
    require(review["status"] == expected_status, "sealed authority status drift")


def validate_source_only_authority() -> None:
    require(len(sys.argv) == 1, "M1094C2 launcher accepts zero arguments")
    require(Path(sys.executable) == PYTHON and tuple(sys.version_info[:3]) == (3, 10, 18),
            "pinned launcher Python identity/version required")
    verify_regular(PYTHON, PYTHON_SHA256)
    verify_regular(ENGINE, ENGINE_SHA256)
    verify_regular(DOCS359, DOCS359_SHA256)
    verify_regular(LICENSE_FILE, LICENSE_FILE_SHA256)
    verify_flat(
        SOURCE_RECEIPT,
        SOURCE_RECEIPT_OUTER_SHA256,
        "PASS_M1090R3_M1091R3_FIXED_HISTORY_SOURCE_ONLY__M1093R2_REQUIRED__NO_EDA",
    )
    verify_flat(
        M1093R2,
        M1093R2_OUTER_SHA256,
        "PASS_M1093R2_M1090R3_M1091R3_ENGINE_HAMMER__AUTHOR_LAUNCH_WRAPPER_ONLY__NO_EDA",
    )


def clean_child_environment() -> dict[str, str]:
    """Return constants only; no caller environment value is consulted."""
    return {
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PATH": "/usr/bin:/bin",
        "TMPDIR": "/tmp",
        "SNPSLMD_LICENSE_FILE": SNPSLMD_LICENSE_FILE,
        "LM_LICENSE_FILE": str(LICENSE_FILE),
    }


def main() -> int:
    validate_source_only_authority()
    completed = subprocess.run(
        [str(PYTHON), "-I", str(ENGINE), "--authorized-launch"],
        cwd=str(HW),
        env=clean_child_environment(),
        close_fds=True,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
