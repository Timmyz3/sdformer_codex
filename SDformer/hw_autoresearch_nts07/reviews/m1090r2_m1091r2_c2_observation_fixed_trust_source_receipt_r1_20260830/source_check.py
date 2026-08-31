#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""No-import, no-EDA static audit of the M1090r2/M1091r2 release."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import stat
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ENGINE = HW / "dc_handoff/scripts/m1091r2_m1090r2_c2_observation_authorized_engine_r1.py"
CONTRACT = HW / "contracts/m1090r2_c2_k1_observation_fixed_trust_source_contract_r1_20260830.json"
RELEASE = HW / "contracts/m1090r2_c2_k1_observation_fixed_trust_release_r1_20260830.json"
WRAPPER = HW / "rtl_m1090r2/m1090r2_c2_k1_observation_wrapper.sv"
TB = HW / "dc_handoff/tb/tb_m1090r2_c2_k1_observation_mapped_case0_short.sv"
M1092 = HW / "reviews/m1092_m1090_c2_observation_source_hammer_r1_20260830"
OLD_RUNNER = HW / "dc_handoff/scripts/run_m1091_m1090_c2_observation_dc_mapped_vcs_one_shot_r1.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def regular(path: Path, expected: str) -> None:
    require(path.exists(), f"missing {path}")
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), f"nonregular/symlink {path}")
    require(sha(path) == expected, f"hash drift {path}")


def double(path: Path, expected: str, outer_expected: str) -> None:
    regular(path, expected)
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    require(stat.S_ISREG(side.lstat().st_mode) and not side.is_symlink(), "sidecar type")
    require(side.read_text().split() == [expected, path.relative_to(HW).as_posix()], "sidecar content")
    regular(outer, outer_expected)
    require(outer.read_text().split() == [sha(side), side.relative_to(HW).as_posix()], "outer content")


contract = json.loads(CONTRACT.read_text())
release = json.loads(RELEASE.read_text())
require(contract["status"] == "M1090R2_FIXED_TRUST_SOURCE_ONLY__M1093_ENGINE_HAMMER_REQUIRED__NO_EDA", "contract status")
require(release["status"] == "M1090R2_FIXED_TRUST_RELEASE_FROZEN__M1093_ENGINE_HAMMER_REQUIRED__NO_EDA", "release status")
require(contract["launch_now"] is False and contract["max_attempts_now"] == 0, "contract launch boundary")
require(release["launch_now"] is False and release["authorization"]["attempt_now"] is False, "release launch boundary")
double(CONTRACT, "c9e7986602e14b17d38d9a67e50238b4f1b05801db93c767006fb20355fae8c5", "7bd062b64438cfebb28105e2532276d8193849e518fb4717ddcaadb4572797a3")
double(RELEASE, "d47c12f1bf235087c1ba81caa0bb385d2221833ca107838e7242d373251188b9", "fc20e494417bcd5105f4697c265f54616c68d0f03a82c7b0ad23a9df78bf8ab4")
require(release["contract_sha256"] == sha(CONTRACT), "release contract pin")

for relative, expected in contract["source_sha256"].items():
    regular(HW / relative, expected)

for name, identity in contract["external_identity"].items():
    path = Path(name)
    if identity["kind"] == "regular":
        regular(path, identity["sha256"])
    else:
        require(identity["kind"] == "exact_symlink", "unexpected external kind")
        require(stat.S_ISLNK(path.lstat().st_mode), "dc_shell must be symlink")
        require(os.readlink(path) == identity["readlink"], "dc_shell readlink drift")
        target = Path(identity["resolved"])
        require(path.resolve(strict=True) == target, "dc_shell resolve drift")
        regular(target, identity["target_sha256"])

engine = ENGINE.read_text()
ast.parse(engine, filename=ENGINE.as_posix())
require("M1091_EXPECTED" not in engine and "EXPECTED_RELEASE" not in engine and "EXPECTED_RUNNER" not in engine, "caller expected-hash variable remains")
require(not re.search(r"os\.environ(?:\.get)?\([^\n]*(?:SHA|HASH|OUTER)", engine, re.I), "caller environment selects identity")
for token in (
    'CONTRACT_SHA256 = "c9e7986602e14b17d38d9a67e50238b4f1b05801db93c767006fb20355fae8c5"',
    'CONTRACT_OUTER_SHA256 = "7bd062b64438cfebb28105e2532276d8193849e518fb4717ddcaadb4572797a3"',
    'RELEASE_SHA256 = "d47c12f1bf235087c1ba81caa0bb385d2221833ca107838e7242d373251188b9"',
    'RELEASE_OUTER_SHA256 = "fc20e494417bcd5105f4697c265f54616c68d0f03a82c7b0ad23a9df78bf8ab4"',
    'M1092_OUTER_SHA256 = "f55dc0afde8d350d1ff028c30e511eb15b2670f3ad1ee2f5643759406ca8ccb4"',
    'M1088_OUTER_SHA256 = "fb3f208dc704c7663769422ad9f27b17851cc86b11826727fe0c0c795260bd5f"',
    'M1080_ATTEMPT_OUTER_SHA256 = "21944247a673bda71a1d3f8cce2cf567b91e51a661b88d5028ed89b70d3a8f7c"',
    'M1080_FAILURE_OUTER_SHA256 = "2e3367c239cda08987027a55a01f65b0cbebbd1c0dd907a9a945aa12f5cea89d"',
    'sys.argv[1:] != ["--authorized-launch"]',
    'return verify_launch_authority()',
    'ATTEMPT.mkdir()',
):
    require(token in engine, f"engine fixed gate missing: {token}")
require(engine.index("return verify_launch_authority()") < engine.index("ATTEMPT.mkdir()"), "attempt precedes launch authority")
require("verify_parent_launcher(receipt)" in engine, "fixed parent launcher gate missing")
require("M1093_OUTER_SHA256" in engine and "ENGINE_SHA256" in engine, "future launcher hard-code checks missing")
require("verify_regular(PYTHON" in engine and "verify_dc_shell()" in engine, "tool identity gate missing")
require("for path, expected in SOURCE_SHA256.items()" in engine, "source lstat/hash loop missing")

launcher = HW / contract["future_launch_chain"]["launcher_path"]
launch_receipt = HW / contract["future_launch_chain"]["launch_receipt_path"]
require(not launcher.exists() and not launcher.is_symlink(), "future launcher exists during source stage")
require(not launch_receipt.exists() and not launch_receipt.is_symlink(), "future launch receipt exists during source stage")
for path in (
    HW / contract["future_namespaces"]["attempt"],
    HW / contract["future_namespaces"]["result"],
    HW / "results/.m1091_m1090_c2_observation_dc_mapped_vcs_attempt_consumed",
):
    require(not path.exists() and not path.is_symlink(), f"attempt/result unexpectedly exists: {path}")

require(sha(OLD_RUNNER) == "fade26df6dd3a6e3a71772c1d880ef31872be213a945c8184c966293e9791199", "old M1091 runner modified")
require(json.loads((M1092 / "review.json").read_text())["status"] == "STOP_M1092_M1090_M1091_SELF_SIGNED_CALLER_AUTHORITY__NO_M1091_ATTEMPT", "M1092 STOP drift")
require(sha(M1092 / "SHA256SUMS.seal.sha256") == "f55dc0afde8d350d1ff028c30e511eb15b2670f3ad1ee2f5643759406ca8ccb4", "M1092 outer drift")

wrapper = WRAPPER.read_text()
require("module m1090r2_c2_k1_observation_wrapper" in wrapper, "wrapper module")
require(len(set(re.findall(r"\bobs_[A-Za-z0-9_]+\b", wrapper))) == 22, "observation count")
impl = wrapper[wrapper.index(") implementation ("):wrapper.index("));", wrapper.index(") implementation ("))]
require("obs_" not in impl, "observation feedback")
tb = TB.read_text()
require(tb.count("`M1090R2_FAIL_X(") == 22, "first-X coverage")
require("window_cycle==128" in tb and "M1090R2_STAGE" in tb, "bounded stage trace")
for forbidden in ("$toggle", "+vcs+initreg", ".saif", "$fsdb", "$dumpfile"):
    require(forbidden.lower() not in tb.lower(), f"forbidden TB feature {forbidden}")

docs_sha = sha(HW / "docs/359_DATE终局冻结_20260813.md")
require(docs_sha == contract["docs359_sha256"], "docs359 drift")
print(json.dumps({
    "status": "PASS_M1090R2_M1091R2_STATIC_FIXED_TRUST_SOURCE_CHECK__NO_EDA",
    "checks": 71,
    "engine_sha256": sha(ENGINE),
    "contract_outer_seal_file_sha256": "7bd062b64438cfebb28105e2532276d8193849e518fb4717ddcaadb4572797a3",
    "release_outer_seal_file_sha256": "fc20e494417bcd5105f4697c265f54616c68d0f03a82c7b0ad23a9df78bf8ab4",
    "caller_expected_hash_variables": 0,
    "source_identities": len(contract["source_sha256"]),
    "external_identities": len(contract["external_identity"]),
    "launch_artifacts_present": False,
    "attempt_consumed": False,
    "eda_executed": False,
    "docs359_sha256": docs_sha,
}, indent=2, sort_keys=True))
