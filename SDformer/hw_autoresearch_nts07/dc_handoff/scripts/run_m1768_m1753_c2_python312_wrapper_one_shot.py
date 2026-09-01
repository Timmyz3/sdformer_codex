#!/usr/bin/python3.12
"""One-shot Python 3.12 execution wrapper for the exact M1753 campaign.

This source does not alter M1753.  Before creating its own atomic attempt it
pins the exact interpreter, M1767 failure receipt, M1753/M1760/M1761 chain,
future different-author M1769 review, and exact M1770 release.  It then replaces
itself with the exact interpreter plus exact M1753 bytes through execve.
"""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import sys


HW = Path(__file__).resolve().parents[2]
WRAPPER = Path(__file__).resolve()
PYTHON312 = Path("/usr/bin/python3.12")
CONTRACT = HW / "contracts/m1768_m1767_m1753_c2_python312_wrapper_source_contract_r1_20260902.json"
M1753 = HW / "dc_handoff/scripts/run_m1753_m1715_c2_three_axis_mapped_directed_component_energy_one_shot.py"
M1753_CONTRACT = HW / "contracts/m1753_m1715_c2_three_axis_mapped_directed_component_energy_source_contract_r1_20260901.json"
M1767 = HW / "reviews/m1767_m1761_m1753_c2_python36_preparse_failure_receipt_r1_20260902"
M1760 = HW / "reviews/m1760_m1753_c2_three_axis_mapped_directed_component_energy_source_hammer_r1_20260901"
M1761 = HW / "contracts/m1761_m1760_m1753_c2_three_axis_mapped_directed_component_energy_launch_release_r1_20260901.json"
M1769 = HW / "reviews/m1769_m1768_m1753_c2_python312_wrapper_source_hammer_r1_20260902"
M1770 = HW / "contracts/m1770_m1769_m1768_m1753_c2_python312_wrapper_launch_release_r1_20260902.json"

ATTEMPT = HW / "results/.m1768_m1753_c2_python312_wrapper_attempt_consumed"
STAGE_PREFIX = ".m1768_m1753_c2_python312_wrapper_attempt_stage."
M1753_ATTEMPT = HW / "results/.m1753_c2_three_axis_mapped_directed_component_energy_attempt_consumed"
M1753_RESULT = HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901"
M1753_FAILURE = HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.failed_or_incomplete.quarantine"
M1753_PRIVATE = HW / "results/m1753_c2_three_axis_mapped_directed_component_energy_r1_20260901.private_build.unsealed_do_not_cite"

PYTHON312_SHA = "0876a8f712651a0c6a2e54aabd163fb85464b2a4ca8e96a15074f2826a1d8814"
M1767_RECEIPT_SHA = "330e533d0f545439b7b0539a0c4816e8e77a6c89330620571162ec060c6b3729"
M1767_MANIFEST_SHA = "ed9fbb6e5a3b30e77b81f74ee64861231336576ce562517a7a999f518c26d474"
M1767_OUTER_SHA = "80aaf88a542ed1fb9e754172d722ec8cdd7741bfe399ce02be212c30c60f2b71"
M1753_SHA = "adb24c20746bc95340952426dbcba1c5fde3400dce7763d73320f303d3a64d9e"
M1753_CONTRACT_SHA = "39f864a254aa3314ab2b4939997674958c7ae7cc5966273629c94d53ecbe0e21"
M1753_CONTRACT_SUM_SHA = "ec8dcccf92d8979b674008ca83edff4ae98f87e127e3212a979801853ac27092"
M1753_CONTRACT_OUTER_SHA = "2b7510d270632a1989366870abdb68e1bcb3470e665c486b89be6d4e3f50b8d9"
M1760_REVIEW_SHA = "987fccddbad6281bb31aa128987118ef4942e210d47201c528ab9be50055329c"
M1760_MANIFEST_SHA = "e8921f4612f9b0b8532b43f441ccd2b93c2600e5dca861cefcb6ef293601afcf"
M1760_OUTER_SHA = "55caca70cf9670ee8e361c062f4c73e1272c399c990eb4b1e27771008f00830e"
M1761_RELEASE_SHA = "bb5b32ead4bd2ff682abfbcedf242b645c20c89d71db0b7eeadc7c18f5191f5e"
M1761_SUM_SHA = "71b353b92b87c559b8c6501e4b8834e4c78383c5e29d1a367b1ae277423f7e3d"
M1761_OUTER_SHA = "47df6661b49128152093bce9cfaacf9868e4288d4647011f1b654f4783095074"


class Failure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def exact(path, digest):
    path = Path(path)
    if not path.is_file() or path.is_symlink() or sha(path) != digest:
        raise Failure("identity drift " + str(path))


def strict_json(path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise Failure("duplicate JSON key " + key)
            value[key] = item
        return value
    path = Path(path)
    if not path.is_file() or path.is_symlink():
        raise Failure("JSON absent/nonregular " + str(path))
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           Failure("nonfinite JSON " + token)))
    if type(value) is not dict:
        raise Failure("JSON root")
    return value


def verify_seal(root, manifest_sha, outer_sha):
    root = Path(root)
    if not root.is_dir() or root.is_symlink():
        raise Failure("sealed directory invalid " + str(root))
    exact(root / "SHA256SUMS", manifest_sha)
    exact(root / "SHA256SUMS.seal.sha256", outer_sha)
    if (root / "SHA256SUMS.seal.sha256").read_text() != manifest_sha + "  SHA256SUMS\n":
        raise Failure("outer seal content")
    listed = set()
    for row in (root / "SHA256SUMS").read_text().splitlines():
        fields = row.split(maxsplit=1)
        if len(fields) != 2:
            raise Failure("manifest syntax")
        rel = Path(fields[1].lstrip("*"))
        name = rel.as_posix()
        if rel.is_absolute() or ".." in rel.parts or name in listed:
            raise Failure("unsafe manifest")
        exact(root / rel, fields[0])
        listed.add(name)
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in sealed directory")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            actual.add(path.relative_to(root).as_posix())
    if actual != listed:
        raise Failure("sealed population drift " + str(root))


def verify_file_seal(path, payload_sha, sum_sha, outer_sha):
    path = Path(path)
    sum_path = Path(str(path) + ".sha256")
    outer_path = Path(str(path) + ".sha256.seal.sha256")
    exact(path, payload_sha)
    exact(sum_path, sum_sha)
    exact(outer_path, outer_sha)
    if sum_path.read_text() != payload_sha + "  " + path.name + "\n":
        raise Failure("file digest sidecar content")
    if outer_path.read_text() != sum_sha + "  " + sum_path.name + "\n":
        raise Failure("file outer sidecar content")


def authority_pin(name):
    value = os.environ.get(name, "")
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise Failure("exact authority absent " + name)
    return value


def seal_dir(root):
    root = Path(root)
    rows = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise Failure("symlink in candidate")
        if path.is_file() and path.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            rows.append((path.relative_to(root).as_posix(), sha(path)))
    rows.sort()
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(digest + "  " + name + "\n"
                                for name, digest in rows))
    (root / "SHA256SUMS.seal.sha256").write_text(
        sha(manifest) + "  SHA256SUMS\n")


def publish_no_replace(source, destination):
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p,
                          ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    if renameat2(-100, os.fsencode(source), -100,
                 os.fsencode(destination), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(destination))


def verify_interpreter():
    if (Path(sys.executable) != PYTHON312
            or Path(sys.executable).resolve() != PYTHON312
            or platform.python_implementation() != "CPython"
            or platform.python_version() != "3.12.13"
            or tuple(sys.version_info[:3]) != (3, 12, 13)):
        raise Failure("interpreter path/version drift")
    exact(PYTHON312, PYTHON312_SHA)
    if WRAPPER.read_bytes().splitlines()[0] != b"#!/usr/bin/python3.12":
        raise Failure("wrapper shebang drift")


def verify_authority():
    exact(WRAPPER, authority_pin("M1768_EXPECTED_WRAPPER_SHA256"))
    exact(CONTRACT, authority_pin("M1768_EXPECTED_SOURCE_CONTRACT_SHA256"))
    verify_seal(M1767, M1767_MANIFEST_SHA, M1767_OUTER_SHA)
    exact(M1767 / "receipt.json", M1767_RECEIPT_SHA)
    exact(M1753, M1753_SHA)
    verify_file_seal(M1753_CONTRACT, M1753_CONTRACT_SHA,
                     M1753_CONTRACT_SUM_SHA, M1753_CONTRACT_OUTER_SHA)
    verify_seal(M1760, M1760_MANIFEST_SHA, M1760_OUTER_SHA)
    exact(M1760 / "review.json", M1760_REVIEW_SHA)
    verify_file_seal(M1761, M1761_RELEASE_SHA, M1761_SUM_SHA, M1761_OUTER_SHA)

    verify_seal(M1769,
                authority_pin("M1768_EXPECTED_M1769_MANIFEST_SHA256"),
                authority_pin("M1768_EXPECTED_M1769_OUTER_FILE_SHA256"))
    exact(M1769 / "review.json",
          authority_pin("M1768_EXPECTED_M1769_REVIEW_SHA256"))
    exact(M1770, authority_pin("M1768_EXPECTED_M1770_RELEASE_SHA256"))
    failure = strict_json(M1767 / "receipt.json")
    review = strict_json(M1769 / "review.json")
    release = strict_json(M1770)
    if (failure.get("status") !=
            "SEALED_M1767_OPERATOR_ENVIRONMENT_FAILURE__M1753_BODY_NOT_ENTERED__EDA_LICENSE_ATTEMPT_RESULT_ZERO__AUTHORIZE_SOURCE_ONLY_M1768_WRAPPER"
            or review.get("status") !=
            "PASS_M1769_M1768_C2_PYTHON312_WRAPPER_SOURCE_HAMMER__AUTHORIZE_ONE_WRAPPER_ATTEMPT"
            or release.get("status") !=
            "AUTHORIZE_ONE_M1768_C2_PYTHON312_WRAPPER_ATTEMPT"):
        raise Failure("failure/review/release status drift")
    if release.get("identity") != {
            "wrapper_sha256": authority_pin("M1768_EXPECTED_WRAPPER_SHA256"),
            "source_contract_sha256": authority_pin("M1768_EXPECTED_SOURCE_CONTRACT_SHA256"),
            "m1769_review_sha256": authority_pin("M1768_EXPECTED_M1769_REVIEW_SHA256")}:
        raise Failure("M1770 identity drift")
    if release.get("authorization") != {
            "future_m1768_wrapper_attempts": 1,
            "automatic_retry": False,
            "underlying_m1753_campaigns": 1}:
        raise Failure("M1770 budget drift")

    expected_underlying = {
        "M1753_EXPECTED_RUNNER_SHA256": M1753_SHA,
        "M1753_EXPECTED_SOURCE_CONTRACT_SHA256": M1753_CONTRACT_SHA,
        "M1753_EXPECTED_M1760_REVIEW_SHA256": M1760_REVIEW_SHA,
        "M1753_EXPECTED_M1760_MANIFEST_SHA256": M1760_MANIFEST_SHA,
        "M1753_EXPECTED_M1760_OUTER_FILE_SHA256": M1760_OUTER_SHA,
        "M1753_EXPECTED_M1761_RELEASE_SHA256": M1761_RELEASE_SHA,
    }
    for name, digest in expected_underlying.items():
        if os.environ.get(name) != digest:
            raise Failure("underlying authority drift " + name)


def namespaces_fresh():
    fixed = (ATTEMPT, M1753_ATTEMPT, M1753_RESULT, M1753_FAILURE, M1753_PRIVATE)
    for path in fixed:
        if os.path.lexists(str(path)):
            raise Failure("namespace residue " + str(path))
    stages = list((HW / "results").glob(STAGE_PREFIX + "*"))
    if stages:
        raise Failure("wrapper stage residue")


def main():
    if len(sys.argv) != 1:
        raise Failure("M1768 accepts no arguments")
    verify_interpreter()
    verify_authority()
    namespaces_fresh()
    stage = HW / "results" / (STAGE_PREFIX + str(os.getpid()))
    stage.mkdir()
    (stage / "attempt.json").write_text(json.dumps({
        "schema": "m1768_m1753_c2_python312_wrapper_attempt_r1_v1",
        "status": "M1768_WRAPPER_ATTEMPT_CONSUMED__EXECVE_M1753_PENDING",
        "wrapper_sha256": sha(WRAPPER),
        "python_path": str(PYTHON312),
        "python_sha256": PYTHON312_SHA,
        "python_version": "3.12.13",
        "target": str(M1753.relative_to(HW)),
        "target_sha256": M1753_SHA,
        "automatic_retry": False,
        "m1753_attempt_preexisting": False,
        "eda_or_license_before_execve": False,
    }, indent=2, sort_keys=True, allow_nan=False) + "\n")
    seal_dir(stage)
    publish_no_replace(stage, ATTEMPT)
    os.execve(str(PYTHON312), [str(PYTHON312), str(M1753)], dict(os.environ))


if __name__ == "__main__":
    main()
