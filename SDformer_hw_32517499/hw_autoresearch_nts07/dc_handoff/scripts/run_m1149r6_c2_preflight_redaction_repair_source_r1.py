#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1149R6 additive M1146 preflight-redaction repair; source only.

The frozen M1146 namespace is still fresh and is deliberately reused, so its
single future attempt remains exact.  No lmstat, VCS, DC, or launch occurs on
import or during source_static_self_test.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True
SOURCE_FILE = Path(__file__).resolve()
HW = SOURCE_FILE.parent.parent.parent
BASE_SOURCE = HW / "dc_handoff/scripts/run_m1146r6_c2_license_route_frozen_netlist_mapped_vcs_successor_source_r1.py"
BASE_SOURCE_SHA = "69c30ccfdc884aecca407b6b86b66bc82f97dd02abdb353561daa083934d591c"
BASE_CONTRACT = HW / "contracts/m1146r6_c2_additive_license_route_successor_source_contract_r1_20260830.json"
BASE_CONTRACT_ID = (
    "dda71d8842325b3c26cd3046d1e93781103f5e5b4277af71602793c68ef7bfd5",
    "1182ef948471bd0235ff8d57817f4dc09295165f2be6ef600d7591348f4f5975",
    "b28d565b1c1ef7b3c79724bf06bc4be202e55010f88b7b0274adb068a9fb82e6",
)
M1146 = HW / "reviews/m1146r6_c2_additive_license_route_successor_author_receipt_r1_20260830"
M1146_ID = (
    "b011596046c724665b71352045c23e82044eaabd5a6ce849be5892b362781fe4",
    "13fe84f0aef4dfc000278a9e1629368b6256742c6c1ad7f2e769174ac1a6360c",
    "513813aa1915e72af18c1b059cfae77947c9ece37fc8699582cc202c489b98d1",
)
M1147 = HW / "reviews/m1147r6_m1146r6_c2_license_route_final_source_hammer_r1_20260830"
M1147_ID = (
    "d4434283285d6f536b30f3183e86b05e9bbbedbcb9689362df6114d29f9844c9",
    "b03909b2d54d971a601d52a42f0dd2f1203cde20cb3ee8c31daf27bcf7e877c1",
    "64007fe4ec37a26c54c197b80ae9f9565e8272c06fecfe3510c24aeb7c74d7e9",
)
M1148 = HW / "reviews/m1148r6_m1146r6_c2_preflight_redaction_failure_hammer_r1_20260830"
M1148_ID = (
    "5d9f9367a676be794ff39a2d1b3384d83ff60e2cdb77f846b6e99a0e9d770be5",
    "512159f0be381dcb948866cb71413cb32f1949a50e03db9fd1186d8ebbf9130d",
    "b60fb9ecd875d87dd0b1f05a8cd448c85ce551a4f589a988f6a0fa8785defb32",
)
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
MANIFEST = "SHA256SUMS"
OUTER = "SHA256SUMS.seal.sha256"


class Failure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_regular(path: Path, expected: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and
            sha256(path) == expected, "identity drift: " + str(path))


def strict_json(path: Path) -> Any:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key")
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, identity[0]); verify_regular(side, identity[1])
    verify_regular(outer, identity[2])
    require(side.read_text(encoding="utf-8").split() == [identity[0], path.name] and
            outer.read_text(encoding="utf-8").split() == [identity[1], side.name],
            "double seal content drift")


def verify_tree(directory: Path, identity: tuple[str, str, str]) -> dict[str, Any]:
    review = directory / "review.json"
    manifest = directory / MANIFEST
    outer = directory / OUTER
    verify_regular(review, identity[0]); verify_regular(manifest, identity[1])
    verify_regular(outer, identity[2])
    require(outer.read_text(encoding="utf-8").split() == [identity[1], MANIFEST],
            "flat outer content drift")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        name = name.lstrip("*"); relative = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and
                name not in expected and name == relative.as_posix() and
                not relative.is_absolute() and ".." not in relative.parts,
                "manifest member drift")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {MANIFEST, OUTER}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(name)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact member census drift")
    for name, digest in expected.items():
        verify_regular(directory / name, digest)
    return strict_json(review)


verify_regular(BASE_SOURCE, BASE_SOURCE_SHA)
_spec = importlib.util.spec_from_file_location("m1149r6_frozen_m1146r6", BASE_SOURCE)
require(_spec is not None and _spec.loader is not None, "M1146 module spec")
BASE = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = BASE
_spec.loader.exec_module(BASE)
_BASE_SOURCE_PREFLIGHT = BASE.source_preflight

# Reuse the still-fresh M1146 namespace exactly; no parallel M1149 namespace.
RESULT = BASE.RESULT
ATTEMPT = BASE.ATTEMPT
WORK_PREFIX = BASE.WORK_PREFIX
FAILURE_PREFIX = BASE.FAILURE_PREFIX
LOCK = BASE.LOCK
LICENSE_KEYS = BASE.LICENSE_KEYS


def authority_preflight() -> dict[str, Any]:
    verify_regular(BASE_SOURCE, BASE_SOURCE_SHA)
    verify_double(BASE_CONTRACT, BASE_CONTRACT_ID)
    m1146 = verify_tree(M1146, M1146_ID)
    m1147 = verify_tree(M1147, M1147_ID)
    m1148 = verify_tree(M1148, M1148_ID)
    verify_regular(DOCS359, DOCS359_SHA)
    require(m1146["status"] ==
            "PASS_M1146R6_SOURCE_CONTRACT_CONTROLLED_MOCK__DIFFERENT_AUTHOR_HAMMER_REQUIRED" and
            m1147["status"] ==
            "PASS_M1147R6_FINAL_SOURCE_HAMMER__ROOT_EXTERNAL_PREFLIGHT_THEN_ONE_EXACT_LICENSE_ROUTED_MAPPED_VCS_EXECUTION_ONLY" and
            m1148["status"] ==
            "PASS_M1148R6_M1146R6_REAL_LMSTAT_PREFLIGHT_FALSE_NEGATIVE__AUTHOR_ADDITIVE_PREFLIGHT_REDACTION_REPAIR_SOURCE_ONLY" and
            m1148["authorization"]["additive_preflight_redaction_repair_source_authoring"] is True and
            m1148["authorization"]["attempt"] is False and
            m1148["authorization"]["vcs"] is False and
            m1148["authorization"]["dc"] is False,
            "M1146/M1147/M1148 authority drift")
    require(BASE.namespace_fresh(), "reused M1146 namespace not fresh")
    return {
        "status": "PASS_M1149R6_AUTHORITY_PREFLIGHT__M1146_NAMESPACE_FRESH__NO_LMSTAT_NO_EDA",
        "m1146_outer_seal_file_sha256": M1146_ID[2],
        "m1147_outer_seal_file_sha256": M1147_ID[2],
        "m1148_outer_seal_file_sha256": M1148_ID[2],
        "namespace_reused": "M1146R6",
        "maximum_attempts": 1,
        "automatic_retry": False,
    }


def _run_lmstat_redacted(key: str, value: str,
                         environment: dict[str, str]) -> bool:
    """Return only rc==0; raw stdout/stderr never escape or persist."""
    require(environment == BASE._child_environment(key, value),
            "lmstat environment drift")
    process = None
    stdout = b""
    stderr = b""
    try:
        process = subprocess.Popen(
            [str(BASE.LMUTIL), "lmstat", "-c", value],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            env=environment, start_new_session=True)
        stdout, stderr = process.communicate(timeout=30)
        return process.returncode == 0
    except subprocess.TimeoutExpired:
        if process is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            except BaseException:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                except BaseException:
                    pass
        return False
    except BaseException:
        raise Failure("lmstat invocation failed") from None
    finally:
        # Explicitly drop transient tool bytes. Neither length nor content is
        # returned, logged, serialized, included in an exception, or sealed.
        stdout = b""
        stderr = b""


def source_preflight(require_fresh: bool = True):
    authority = authority_preflight()
    public, key, value, child = _BASE_SOURCE_PREFLIGHT(require_fresh=require_fresh)
    require("HOME" not in child and set(public["route"]) ==
            {"selected_variable", "present", "byte_length", "sha256"} and
            public["route"]["selected_variable"] == key and
            public["route"]["present"] is True and
            public["route"]["byte_length"] == len(value.encode("utf-8")) and
            public["route"]["sha256"] == hashlib.sha256(value.encode("utf-8")).hexdigest() and
            value not in json.dumps(public, sort_keys=True),
            "redacted public preflight drift")
    public = dict(public)
    public["redaction_repair"] = {
        "status": "PASS_LMSTAT_RC_ONLY__RAW_STDOUT_STDERR_DISCARDED",
        "m1148_outer_seal_file_sha256": M1148_ID[2],
        "raw_stdout_returned_or_persisted": False,
        "raw_stderr_returned_or_persisted": False,
        "route_value_returned_or_persisted": False,
        "namespace_reused": "M1146R6",
        "maximum_attempts": authority["maximum_attempts"],
        "automatic_retry": False,
    }
    require(value not in json.dumps(public, sort_keys=True),
            "license route leaked from repaired preflight")
    return public, key, value, child


def configure_base() -> None:
    BASE._run_lmstat = _run_lmstat_redacted
    BASE.source_preflight = source_preflight


def source_static_self_test() -> dict[str, Any]:
    authority = authority_preflight()
    snps_key, snps_value, snps_meta = BASE._select_license_route({
        "SNPSLMD_LICENSE_FILE": "snps-controlled",
        "LM_LICENSE_FILE": "lm-controlled",
    })
    lm_key, lm_value, lm_meta = BASE._select_license_route({
        "LM_LICENSE_FILE": "lm-controlled",
    })
    require((snps_key, snps_value) ==
            ("SNPSLMD_LICENSE_FILE", "snps-controlled") and
            (lm_key, lm_value) == ("LM_LICENSE_FILE", "lm-controlled") and
            set(snps_meta) == set(lm_meta) ==
            {"selected_variable", "present", "byte_length", "sha256"} and
            "HOME" not in BASE._child_environment(snps_key, snps_value) and
            "HOME" not in BASE._child_environment(lm_key, lm_value),
            "license priority/fallback/HOME oracle drift")
    return {
        "status": "PASS_M1149R6_SOURCE_STATIC_SELF_TEST__NO_LMSTAT_NO_VCS_NO_LAUNCH",
        "authority": authority,
        "snps_priority": True,
        "lm_fallback": True,
        "home_absent": True,
        "persistent_route_metadata_fields": [
            "selected_variable", "present", "byte_length", "sha256"],
        "real_lmstat_calls": 0,
        "vcs_calls": 0,
        "dc_calls": 0,
        "attempt_created": False,
    }


def production_main() -> dict[str, Any]:
    configure_base()
    return BASE._future_execute_once()


def main() -> int:
    require(len(sys.argv) == 1, "M1149R6 accepts zero arguments")
    print(json.dumps(production_main(), sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
