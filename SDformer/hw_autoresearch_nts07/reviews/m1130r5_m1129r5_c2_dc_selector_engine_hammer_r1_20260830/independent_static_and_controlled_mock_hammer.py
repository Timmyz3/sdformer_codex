#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Different-author M1129r5 engine hammer: static plus controlled mocks only."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Callable
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
ENGINE = HW / "dc_handoff/scripts/m1129r5_c2_real_module_async_observation_engine_source_r1.py"
R4_ENGINE = HW / "dc_handoff/scripts/m1122r4_c2_dc_selector_async_observation_engine_source_r1.py"
CONTRACT = HW / "contracts/m1129r5_c2_dc_selector_async_observation_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1129r5_c2_real_module_engine_author_receipt_r1_20260830"
RTL = HW / "rtl_m1129r5/m1129r5_c2_k1_async_observation_shadow_wrapper.sv"
BASE_RTL = HW / "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
TB = HW / "dc_handoff/tb/tb_m1129r5_c2_k1_async_observation_shadow_case0_short.sv"
BASE_TB = HW / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
FILELIST = HW / "dc_handoff/filelists/date_m1129r5_c2_k1_async_observation_shadow_logic_only_dc.f"
M1128 = HW / "reviews/m1128r5_m1122r4_c2_elaboration_failure_readonly_audit_r1_20260830"
R4_ATTEMPT = HW / "results/.m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
R4_FAILURE = HW / "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.580027.quarantine"
R4_RESULT = HW / "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830"
R3_ATTEMPT = HW / "results/.m1112r3_c2_async_observation_dc_mapped_vcs_attempt_consumed"
R3_FAILURE = HW / "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.213812.quarantine"
R3_RESULT = HW / "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830"
R5_ATTEMPT = HW / "results/.m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
R5_RESULT = HW / "results/m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830"
R5_WORK_GLOB = ".m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_work.*"
R5_FAILURE_GLOB = "m1129r5_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*"
R5_LOCK = Path("/tmp/m1129r5_c2_dc_selector_async_observation_eda.lock")
FUTURE_LAUNCHER = HW / "dc_handoff/scripts/run_m1129r5_c2_dc_selector_async_observation_authorized_launch_r1.py"
FUTURE_RECEIPT = HW / "contracts/m1129r5_c2_dc_selector_async_observation_authorized_launch_receipt_r1_20260830.json"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "engine": "c8fd3366ecf6c4377b62e5717d959348c08192ea8bdbd0afd3b0e566bd6fbd0b",
    "r4_engine": "f278052d251af0c2d150872391306c2f3922049ca04c7df2a0d9d3d074b55007",
    "contract": "25cfbf9e2d75333e27a1162ab202b9b6a9b305876ee92ce6ed9f6d30513f370d",
    "contract_side": "d7b31831edf5ced6c9df04b12aa08ee8078e10da051bfab8be24bba9ab630a6a",
    "contract_outer": "b5a389b2b76a83f6449bfcbc928c416df877f611cfbd987d828552cb4bdf50cf",
    "author_review": "b80f60a74c6c18fc055c4e8280753611de0cd600f089ca1e5a6626cd8bbbc9de",
    "author_manifest": "529f7e1443a2877428d1009a6c65d79bed44ffb654a22d2677bb3688f5c795fb",
    "author_outer": "f31e0b11049229d17d2c91eb6290ff98f5fe963dd32d0329403237d894ce2ef3",
    "rtl": "86df0f7be383e6ba8ee17c1e27fc25fd18eb6fecc01329c41a976cd836004dd0",
    "base_rtl": "95c31bc70a7617c6653eaca2f77a54388119f744b814dfc909c75edad1c39218",
    "tb": "c08d22d69c222b8c527bdb70cc5b49392c5467bc3142ebc22ec577da6918147b",
    "base_tb": "ff6bd371c3b1371c520b38680960ad0297a8c01eb92eb7b4a0f4d2e59fc861b6",
    "filelist": "1ac2715245cce259f3dcba37cbeecac0e9a2ab9b16a60463f6a53f668ff9e106",
    "m1128_outer": "9435b3e94b0053b296eccc95058b3799a1002e018d4b15b0c89058e8b68e8730",
    "r4_attempt_outer": "8a012c8638c2e8a8da743cbf570a13f5c8bc8d85716b433882d03405e12e5e37",
    "r4_failure_outer": "2f9173b1e988b1f639e6c3d683fdf720fa9debfeaca8caf27bf5845a36527f83",
    "r3_attempt_outer": "b3355ec5ad9e896512f09609d46336b32554889604a352d87dbdd11200a93816",
    "r3_failure_outer": "537981717cddd3c70fc0ddc9bd6297158884f15b5cceee7c51eab9388a1562d6",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks = 0
attacks: dict[str, str] = {}


class HammerFailure(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    global checks
    checks += 1
    if not value:
        raise HammerFailure(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          HammerFailure("nonfinite JSON: " + token)))


def verify_regular(path: Path, digest: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink() and sha(path) == digest,
            "regular identity drift: " + str(path))


def verify_double(path: Path, primary: str, side_sha: str, outer_sha: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    verify_regular(path, primary); verify_regular(side, side_sha); verify_regular(outer, outer_sha)
    require(side.read_text(encoding="utf-8").split() ==
            [primary, path.relative_to(HW).as_posix()], "sidecar content drift")
    require(outer.read_text(encoding="utf-8").split() ==
            [side_sha, side.relative_to(HW).as_posix()], "outer content drift")


def verify_flat(directory: Path, review_sha: str | None, manifest_sha: str | None,
                outer_sha: str) -> dict[str, Any]:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(),
            "sealed directory drift: " + directory.name)
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    if review_sha is not None:
        verify_regular(directory / "review.json", review_sha)
    if manifest_sha is not None:
        verify_regular(manifest, manifest_sha)
    verify_regular(outer, outer_sha)
    require(outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"],
            "sealed outer content drift")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1); name = name.lstrip("*"); rel = Path(name)
        require(re.fullmatch(r"[0-9a-f]{64}", digest) is not None and name not in expected and
                not rel.is_absolute() and ".." not in rel.parts and rel.as_posix() == name,
                "unsafe/duplicate manifest member")
        expected[name] = digest
    actual = set()
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        if relative in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode):
            actual.add(relative)
        else:
            require(stat.S_ISDIR(mode), "sealed special member")
    require(actual == set(expected), "sealed exact-member mismatch")
    for name, digest in expected.items():
        verify_regular(directory / name, digest)
    payload = next(name for name in ("review.json", "attempt.json", "failure.json")
                   if (directory / name).exists())
    return strict_json(directory / payload)


def sv_tokens(text: str) -> list[str]:
    """Lexical census after removing comments and strings, preserving identifiers."""
    clean = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    clean = re.sub(r"//[^\n]*", " ", clean)
    clean = re.sub(r'"(?:\\.|[^"\\])*"', '""', clean)
    return re.findall(r"\\\S+|[A-Za-z_$][A-Za-z0-9_$]*|\d+'[sS]?[bodhBODH][0-9a-fA-F_xXzZ?]+|\d+|==|!=|<=|>=|&&|\|\||\S", clean)


def lexical_gate(rtl: str, tb: str, filelist: str) -> dict[str, Any]:
    base_rtl = BASE_RTL.read_text(encoding="utf-8")
    base_tb = BASE_TB.read_text(encoding="utf-8")
    old_design = "m1112_c2_k1_async_observation_shadow_wrapper"
    design = "m1129r5_c2_k1_async_observation_shadow_wrapper"
    old_top = "tb_m1112_c2_k1_async_observation_shadow_case0_short"
    top = "tb_m1129r5_c2_k1_async_observation_shadow_case0_short"
    r0, r1 = sv_tokens(base_rtl), sv_tokens(rtl)
    t0, t1 = sv_tokens(base_tb), sv_tokens(tb)
    require(len(r0) == len(r1) and len(t0) == len(t1), "token length drift")
    rtl_diff = [(a, b) for a, b in zip(r0, r1) if a != b]
    tb_diff = [(a, b) for a, b in zip(t0, t1) if a != b]
    require(rtl_diff == [(old_design, design)], "RTL diff not exactly one identifier")
    require(tb_diff == [(old_top, top), (old_design, design)],
            "TB diff not exactly top plus DUT identifiers")
    require([r1[index + 1] for index, token in enumerate(r1[:-1]) if token == "module"] == [design],
            "real RTL module lexical census")
    require([t1[index + 1] for index, token in enumerate(t1[:-1]) if token == "module"] == [top],
            "real TB module lexical census")
    require(len(re.findall(r"(?m)^\s*" + re.escape(design) + r"\s+dut\s*\(", tb)) == 1,
            "real DUT lexical census")
    combined = rtl + "\n" + tb + "\n" + filelist
    require(re.search(r"`\s*(?:define|include)\b", combined) is None,
            "define/include rename forbidden")
    lines = [line.strip() for line in filelist.splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    expected = "rtl_m1129r5/m1129r5_c2_k1_async_observation_shadow_wrapper.sv"
    require(lines[-1] == expected and lines.count(expected) == 1 and
            all("m1112r2" not in line and "m1122r4" not in line for line in lines),
            "filelist does not select real r5 RTL directly")
    return {"rtl_changed_tokens": 1, "tb_changed_tokens": 2,
            "rtl_modules": 1, "tb_modules": 1, "tb_dut_types": 1,
            "define_include_rename": False}


def function_text(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next((item for item in tree.body
                 if isinstance(item, ast.FunctionDef) and item.name == name), None)
    require(node is not None, "missing function: " + name)
    return ast.unparse(node)


def engine_contract_gate(source: str, contract: dict[str, Any]) -> dict[str, Any]:
    ast.parse(source)
    require('M1128R5_OUTER_SHA256 = "' + EXPECTED["m1128_outer"] + '"' in source,
            "M1128r5 outer not bound")
    require('OLD_M1122R4_ATTEMPT_OUTER_SHA256 = "' + EXPECTED["r4_attempt_outer"] + '"' in source and
            'OLD_M1122R4_FAILURE_OUTER_SHA256 = "' + EXPECTED["r4_failure_outer"] + '"' in source and
            'OLD_M1112R3_ATTEMPT_OUTER_SHA256 = "' + EXPECTED["r3_attempt_outer"] + '"' in source and
            'OLD_M1112R3_FAILURE_OUTER_SHA256 = "' + EXPECTED["r3_failure_outer"] + '"' in source,
            "r4/r3 stopped namespaces not bound")
    require("SHADOW_REGISTER_BITS = 337" in source and
            "reset_census = structural_reset_gate(netlist)" in source,
            "337 reset provenance gate missing")
    require(source.count("ATTEMPT.mkdir(); attempted = True") == 1 and
            source.count('"dc_attempts": 1') == 1 and
            '"m1129r5_retry": False' in source,
            "max-one/no-retry source drift")
    require(contract["launch_now"] is False and contract["max_attempts_now"] == 0 and
            contract["frozen_stopped_namespaces"]["m1129r5_maximum_attempts_after_all_hammers"] == 1 and
            contract["frozen_stopped_namespaces"]["automatic_retry"] is False and
            contract["m1122r4_stopped_authority"]["retry_allowed"] is False and
            contract["m1122r4_stopped_authority"]["namespace_reuse_allowed"] is False and
            contract["m1112r3_stopped_authority"]["retry_allowed"] is False and
            contract["m1112r3_stopped_authority"]["namespace_reuse_allowed"] is False,
            "contract max-one/no-retry drift")
    chain = contract["future_chain"]
    require(chain["launch_receipt_contains_future_m1132r5_outer"] is False and
            chain["placeholder_or_hash_fixed_point_allowed"] is False and
            chain["m1132r5_outer_discovery"] == "verify_flat_self_consistent at authorized execution" and
            "m1132r5_outer_seal_file_sha256" not in chain["launch_receipt_allowed_fields"] and
            chain["launcher_exists_now"] is False and chain["launch_receipt_exists_now"] is False and
            chain["attempt_authority_now"] is False,
            "future launcher authority cycle")
    expected_receipt_keys = function_text(source, "verify_future_authority")
    require("m1132r5_outer_seal_file_sha256" in expected_receipt_keys and
            'verify_flat_self_consistent(M1132R5)' in expected_receipt_keys,
            "future hammer must be discovered, not embedded")
    require(contract["claim_boundary"] == {
        "source_only": True, "mutation_selftest_only": True, "eda_executed": False,
        "attempt_consumed": False, "mapped_functionality": False,
        "paper_citable": False, "activity_or_power": False, "performance": False,
        "system_speedup": False, "paper_ppa_ready": False}, "claim escalation")
    selector = contract["dc_selector_contract"]
    require(selector["launch_argv"] == [
        "/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell", "-f",
        str(HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl")] and
            selector["common_shell_exec"]["exact_runtime_argv"] == [
        "/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec",
        "-shell", "dc_shell", "-r", "/opt/synopsys/syn/V-2023.12-SP3", "-f",
        str(HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl")],
        "selector exact argv contract drift")
    return {"maximum_attempts": 1, "automatic_retry": False,
            "future_authority_hash_cycle": False}


def load_engine():
    spec = importlib.util.spec_from_file_location("m1130r5_subject_engine", ENGINE)
    require(spec is not None and spec.loader is not None, "cannot load engine")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakePopen:
    last: "FakePopen | None" = None

    def __init__(self, argv, stdout, stderr, env, close_fds):
        self.argv = list(argv); self.env = env; self.pid = 424242
        self.alive = True; self.terminated = False; self.killed = False
        FakePopen.last = self

    def poll(self):
        return None if self.alive else 0

    def wait(self, timeout=None):
        self.alive = False
        return 0

    def terminate(self):
        self.terminated = True; self.alive = False

    def kill(self):
        self.killed = True; self.alive = False


def controlled_selector_mock(engine, identity_override: dict[str, Any] | None = None) -> dict[str, Any]:
    expected_argv = [str(engine.DC_ACTUAL), "-shell", "dc_shell", "-r",
                     str(engine.DC_INSTALL_ROOT), "-f", str(engine.DC_TCL)]
    identity = {"pid": 424242, "ppid": 7, "starttime": 123456789,
                "uid": os.getuid(), "exe": str(engine.DC_ACTUAL), "argv": expected_argv}
    if identity_override:
        identity.update(identity_override)
    env = {"PATH": "/mock/bin", "SNPSLMD_LICENSE_FILE": "mock-license",
           "M1130R5_ENV_SENTINEL": "preserved"}
    with tempfile.TemporaryDirectory(prefix="m1130r5_controlled_mock_") as raw:
        root = Path(raw); log = root / "dc.log"; receipt = root / "identity.json"
        with patch.object(engine.subprocess, "Popen", FakePopen), \
             patch.object(engine, "process_identity", return_value=identity), \
             patch.object(engine, "verify_regular", return_value=None), \
             patch.object(engine.os, "readlink", return_value="snps_shell"):
            rc = engine.run_dc_with_selector_capture(log, 9, env, receipt)
        require(rc == 0 and FakePopen.last is not None and
                FakePopen.last.argv == [str(engine.DC_SHELL), "-f", str(engine.DC_TCL)] and
                FakePopen.last.env is env and FakePopen.last.env == env,
                "selector mock launch/env forwarding drift")
        captured = strict_json(receipt)
        require(captured["exe"] == str(engine.DC_ACTUAL) and
                captured["argv"] == expected_argv and captured["uid"] == os.getuid() and
                captured["status"] == "PASS_M1129R5_EXACT_DC_SELECTOR_RUNTIME_CAPTURE",
                "selector runtime identity receipt drift")
        return {"launch_argv": FakePopen.last.argv,
                "runtime_argv": captured["argv"], "same_pid": captured["pid"] == 424242,
                "environment_forwarded_unchanged": True}


def synthetic_reset_netlist(bits: int = 337, direct: bool = False,
                            bad_cell: bool = False) -> str:
    lines = []
    for index in range(bits):
        net = "rst_core" if direct and index == 0 else f"rst_n_{index}"
        if not (direct and index == 0):
            cell = "BUFFD1BWP35P140" if bad_cell and index == 0 else "INVD1BWP35P140"
            lines.append(f"{cell} reset_inv_{index} (.I(rst_core), .ZN({net}));")
        lines.append("DFCNQD1BWP35P140 "
                     f"shadow_counter_q_reg_{index} (.D(d_{index}), .CP(clk_core), "
                     f".CDN({net}), .Q(q_{index}));")
    return "\n".join(lines) + "\n"


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise HammerFailure("attack accepted: " + label)


def namespace_snapshot() -> dict[str, Any]:
    return {
        "attempt": R5_ATTEMPT.exists() or R5_ATTEMPT.is_symlink(),
        "result": R5_RESULT.exists() or R5_RESULT.is_symlink(),
        "work": sorted(path.name for path in (HW / "results").glob(R5_WORK_GLOB)),
        "failure": sorted(path.name for path in (HW / "results").glob(R5_FAILURE_GLOB)),
        "lock": R5_LOCK.exists() or R5_LOCK.is_symlink(),
    }


def main() -> None:
    subject_paths = (ENGINE, R4_ENGINE, CONTRACT, RTL, BASE_RTL, TB, BASE_TB,
                     FILELIST, DOCS359)
    before = {path: sha(path) for path in subject_paths}
    namespace_before = namespace_snapshot()
    require(namespace_before == {"attempt": False, "result": False, "work": [],
                                 "failure": [], "lock": False},
            "r5 namespace already exists")
    verify_regular(ENGINE, EXPECTED["engine"]); verify_regular(R4_ENGINE, EXPECTED["r4_engine"])
    verify_double(CONTRACT, EXPECTED["contract"], EXPECTED["contract_side"], EXPECTED["contract_outer"])
    verify_regular(RTL, EXPECTED["rtl"]); verify_regular(BASE_RTL, EXPECTED["base_rtl"])
    verify_regular(TB, EXPECTED["tb"]); verify_regular(BASE_TB, EXPECTED["base_tb"])
    verify_regular(FILELIST, EXPECTED["filelist"]); verify_regular(DOCS359, EXPECTED["docs359"])
    author = verify_flat(AUTHOR, EXPECTED["author_review"], EXPECTED["author_manifest"], EXPECTED["author_outer"])
    m1128 = verify_flat(M1128, None, None, EXPECTED["m1128_outer"])
    r4_attempt = verify_flat(R4_ATTEMPT, None, None, EXPECTED["r4_attempt_outer"])
    r4_failure = verify_flat(R4_FAILURE, None, None, EXPECTED["r4_failure_outer"])
    r3_attempt = verify_flat(R3_ATTEMPT, None, None, EXPECTED["r3_attempt_outer"])
    r3_failure = verify_flat(R3_FAILURE, None, None, EXPECTED["r3_failure_outer"])
    require(author["status"] ==
            "PASS_M1129R5_REAL_MODULE_ENGINE_SOURCE_AUTHOR_RECEIPT__M1130R5_REQUIRED__NO_EDA" and
            author["identity"]["engine_sha256"] == EXPECTED["engine"] and
            author["identity"]["contract_sha256"] == EXPECTED["contract"],
            "author receipt identity/status drift")
    require(m1128["status"] ==
            "PASS_M1128R5_M1122R4_FAILURE_AUDIT__R4_PERMANENT_NO_RETRY__ADDITIVE_R5_ONLY" and
            r4_attempt["dc_attempts"] == 1 and
            r4_failure["status"] == "FAILED_DIAGNOSTIC_DO_NOT_CITE" and
            r4_failure["m1122r4_retry"] is False and not R4_RESULT.exists() and
            r3_attempt["dc_attempts"] == 1 and
            r3_failure["status"] == "FAILED_DIAGNOSTIC_DO_NOT_CITE" and
            r3_failure["m1112_retry"] is False and not R3_RESULT.exists(),
            "r4/r3 consumed no-retry isolation drift")
    source = ENGINE.read_text(encoding="utf-8")
    contract = strict_json(CONTRACT)
    lexical = lexical_gate(RTL.read_text(encoding="utf-8"),
                           TB.read_text(encoding="utf-8"), FILELIST.read_text(encoding="utf-8"))
    one_shot = engine_contract_gate(source, contract)

    r4_source = R4_ENGINE.read_text(encoding="utf-8")
    inherited = {}
    for name in ("verify_dc_selector", "process_identity", "terminate_process",
                 "run_dc_with_selector_capture"):
        old = function_text(r4_source, name)
        new = function_text(source, name).replace("M1129R5", "M1122R4").replace("m1129r5", "m1122r4")
        require(old == new, "selector/capture function drift: " + name)
        inherited[name] = True

    engine = load_engine()
    selector_mock = controlled_selector_mock(engine)
    reset = engine.structural_reset_gate_text(synthetic_reset_netlist())
    require(reset["shadow_register_bits"] == 337 and reset["canonical_reset"] == "rst_core" and
            reset["inversion_depth"] == 1 and len(reset["active_low_clear_nets"]) == 337,
            "337-bit controlled reset provenance mock drift")

    rtl = RTL.read_text(encoding="utf-8"); tb = TB.read_text(encoding="utf-8"); fl = FILELIST.read_text(encoding="utf-8")
    rejected("rtl_body_token", lambda: lexical_gate(rtl.replace("obs_fault=", "obs_fault=1'b0|", 1), tb, fl))
    rejected("rtl_extra_module", lambda: lexical_gate(rtl + "\nmodule extra; endmodule\n", tb, fl))
    rejected("tb_third_token", lambda: lexical_gate(rtl, tb.replace("wait_cycles=0;", "wait_cycles=1;", 1), fl))
    rejected("tb_wrong_dut", lambda: lexical_gate(rtl, tb.replace("m1129r5_c2_", "m1112_c2_", 1), fl))
    rejected("filelist_define", lambda: lexical_gate(rtl, tb, fl + "`define ALIAS 1\n"))
    rejected("filelist_include", lambda: lexical_gate(rtl, tb, fl + "`include \"alias.sv\"\n"))
    rejected("filelist_r4_alias", lambda: lexical_gate(rtl, tb, fl.replace(
        "rtl_m1129r5/m1129r5_c2_k1_async_observation_shadow_wrapper.sv",
        "rtl_m1122r4/m1122r4_c2_k1_async_observation_shadow_wrapper.sv")))
    bad = copy.deepcopy(contract); bad["frozen_stopped_namespaces"]["m1129r5_maximum_attempts_after_all_hammers"] = 2
    rejected("max_attempts_two", lambda: engine_contract_gate(source, bad))
    bad = copy.deepcopy(contract); bad["m1122r4_stopped_authority"]["retry_allowed"] = True
    rejected("r4_retry", lambda: engine_contract_gate(source, bad))
    bad = copy.deepcopy(contract); bad["m1112r3_stopped_authority"]["namespace_reuse_allowed"] = True
    rejected("r3_namespace_reuse", lambda: engine_contract_gate(source, bad))
    bad = copy.deepcopy(contract); bad["future_chain"]["launch_receipt_contains_future_m1132r5_outer"] = True
    rejected("future_hash_cycle", lambda: engine_contract_gate(source, bad))
    bad = copy.deepcopy(contract); bad["claim_boundary"]["mapped_functionality"] = True
    rejected("mapped_claim", lambda: engine_contract_gate(source, bad))
    rejected("shadow_census_336", lambda: engine.structural_reset_gate_text(synthetic_reset_netlist(336)))
    rejected("shadow_direct_active_high_reset", lambda: engine.structural_reset_gate_text(
        synthetic_reset_netlist(direct=True)))
    rejected("shadow_buffer_reset", lambda: engine.structural_reset_gate_text(
        synthetic_reset_netlist(bad_cell=True)))
    expected_runtime = [str(engine.DC_ACTUAL), "-shell", "dc_shell", "-r",
                        str(engine.DC_INSTALL_ROOT), "-f", str(engine.DC_TCL)]
    rejected("selector_runtime_argv", lambda: controlled_selector_mock(
        engine, {"argv": expected_runtime[:-1] + ["/wrong.tcl"]}))
    rejected("selector_runtime_uid", lambda: controlled_selector_mock(
        engine, {"uid": os.getuid() + 1}))

    require({path: sha(path) for path in subject_paths} == before,
            "hammer modified subject/docs359")
    require(namespace_snapshot() == namespace_before,
            "controlled hammer created r5 namespace")
    require(not FUTURE_LAUNCHER.exists() and not FUTURE_LAUNCHER.is_symlink() and
            not FUTURE_RECEIPT.exists() and not FUTURE_RECEIPT.is_symlink(),
            "future launcher/receipt pre-exists")
    print(json.dumps({
        "schema": "m1130r5_m1129r5_c2_engine_static_controlled_mock_hammer_v1",
        "status": "PASS_M1130R5_M1129R5_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA",
        "score": 100,
        "checks": checks,
        "attacks_rejected": len(attacks),
        "attacks": attacks,
        "lexical_identity": lexical,
        "selector_capture_inherited_from_r4": inherited,
        "controlled_selector_mock": selector_mock,
        "controlled_reset_mock": reset,
        "one_shot_and_future_chain": one_shot,
        "namespace_before_after_equal": True,
        "execution": {"engine_main": False, "real_engine": False, "dc": False,
                      "vcs": False, "attempt_created": False, "r5_namespace_created": False},
        "authorization": {"zero_argument_launcher_authoring": True,
                          "launch": False, "attempt": False, "dc": False, "vcs": False},
    }, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
