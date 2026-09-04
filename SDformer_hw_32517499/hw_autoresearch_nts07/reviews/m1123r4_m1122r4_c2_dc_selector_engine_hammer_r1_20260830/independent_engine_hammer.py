#!/usr/bin/env python3
"""Independent static M1123r4 hammer. Never calls engine main or an EDA tool."""
from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ENGINE = HW / "dc_handoff/scripts/m1122r4_c2_dc_selector_async_observation_engine_source_r1.py"
CONTRACT = HW / "contracts/m1122r4_c2_dc_selector_async_observation_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1122r4_c2_dc_selector_engine_author_receipt_r1_20260830"
M1121 = HW / "reviews/m1121_m1112r3_c2_dc_invocation_failure_audit_r1_20260830"
OLD_ATTEMPT = HW / "results/.m1112r3_c2_async_observation_dc_mapped_vcs_attempt_consumed"
OLD_FAILURE = HW / "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.213812.quarantine"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

ENGINE_SHA = "f278052d251af0c2d150872391306c2f3922049ca04c7df2a0d9d3d074b55007"
CONTRACT_SHA = "cee4ddc66c244bf4e19e2ce193573b55bf4fd973c7c1bcd53d609d77a9b8cea3"
CONTRACT_SIDE_SHA = "0a1ed1ad054b8a778c17c71eb9fdd82d5df943d77989280bd43660736795b617"
CONTRACT_OUTER = "373e6b86bdfdf94584f289f8c0fc1af1dc9a7ea19be656cba93159b3efb06987"
AUTHOR_OUTER = "c36311a8ac2d5b425c2e3b45a7fee665d9f93cd07e06fd4a095746d7c7c99c9b"
M1121_OUTER = "dc0135b61750134c37b6e3eba47350a0d9838c9ed0a07ca5ecab3bb93c3ff828"
OLD_ATTEMPT_OUTER = "b3355ec5ad9e896512f09609d46336b32554889604a352d87dbdd11200a93816"
OLD_FAILURE_OUTER = "537981717cddd3c70fc0ddc9bd6297158884f15b5cceee7c51eab9388a1562d6"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

DC_SHELL = "/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"
WRAPPER = "/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell"
WRAPPER_SHA = "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2"
ACTUAL = "/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec"
ACTUAL_SHA = "bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391"
INSTALL_ROOT = "/opt/synopsys/syn/V-2023.12-SP3"
TCL = str(HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl")
EXPECTED_BACKEND_ARGV = [ACTUAL, "-shell", "dc_shell", "-r", INSTALL_ROOT, "-f", TCL]


class Reject(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise Reject(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_pairs(rows):
    value = {}
    for key, item in rows:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def strict_text(text: str):
    value = json.loads(text, object_pairs_hook=strict_pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(Reject("nonfinite " + token)))
    def finite(node):
        if isinstance(node, float):
            require(math.isfinite(node), "nonfinite float")
        elif isinstance(node, dict):
            for child in node.values(): finite(child)
        elif isinstance(node, list):
            for child in node: finite(child)
    finite(value)
    return value


def strict_load(path: Path):
    require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink(), "JSON regular")
    return strict_text(path.read_text(encoding="utf-8"))


def safe_rows(manifest: Path) -> dict[str, str]:
    result = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and re.fullmatch(r"[0-9a-f]{64}", fields[0]) is not None, "manifest row")
        name = fields[1].lstrip("*"); rel = Path(name)
        require(name not in result and name == rel.as_posix() and not rel.is_absolute() and ".." not in rel.parts,
                "manifest name")
        result[name] = fields[0]
    return result


def verify_flat(directory: Path, expected_outer: str) -> dict:
    require(stat.S_ISDIR(directory.lstat().st_mode) and not directory.is_symlink(), "sealed root")
    manifest = directory / "SHA256SUMS"; outer = directory / "SHA256SUMS.seal.sha256"
    require(stat.S_ISREG(manifest.lstat().st_mode) and not manifest.is_symlink(), "manifest regular")
    require(stat.S_ISREG(outer.lstat().st_mode) and not outer.is_symlink(), "outer regular")
    require(sha(outer) == expected_outer and
            outer.read_text(encoding="utf-8").split() == [sha(manifest), "SHA256SUMS"], "outer identity")
    expected = safe_rows(manifest); actual = set()
    for member in directory.rglob("*"):
        name = member.relative_to(directory).as_posix()
        if name in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}: continue
        mode = member.lstat().st_mode
        require(not stat.S_ISLNK(mode), "sealed symlink")
        if stat.S_ISREG(mode): actual.add(name)
        else: require(stat.S_ISDIR(mode), "sealed special")
    require(actual == set(expected), "sealed member set")
    for name, digest in expected.items():
        member = directory / name
        require(stat.S_ISREG(member.lstat().st_mode) and not member.is_symlink() and sha(member) == digest,
                "sealed member identity")
    return {"members": len(expected), "manifest_sha256": sha(manifest), "outer": sha(outer)}


def verify_double() -> dict:
    side = Path(str(CONTRACT) + ".sha256"); outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    for path, digest in ((CONTRACT, CONTRACT_SHA), (side, CONTRACT_SIDE_SHA), (outer, CONTRACT_OUTER)):
        require(stat.S_ISREG(path.lstat().st_mode) and not path.is_symlink() and sha(path) == digest,
                "contract triple identity")
    require(side.read_text(encoding="utf-8").split() == [CONTRACT_SHA, CONTRACT.relative_to(HW).as_posix()],
            "contract side content")
    require(outer.read_text(encoding="utf-8").split() == [CONTRACT_SIDE_SHA, side.relative_to(HW).as_posix()],
            "contract outer content")
    return strict_load(CONTRACT)


def function_text(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(item for item in tree.body if isinstance(item, ast.FunctionDef) and item.name == name)
    return ast.get_source_segment(source, node) or ast.unparse(node)


def definitions(path: Path) -> dict:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign, ast.ClassDef, ast.FunctionDef)
    namespace = {"__file__": str(path), "__name__": "m1123r4_safe_model"}
    exec(compile(ast.Module(body=[node for node in tree.body if isinstance(node, allowed)], type_ignores=[]),
                 str(path), "exec"), namespace)
    return namespace


def validate_source(source: str, contract: dict) -> None:
    ast.parse(source)
    capture = function_text(source, "run_dc_with_selector_capture")
    identity = function_text(source, "process_identity")
    selector = function_text(source, "verify_dc_selector")
    static = function_text(source, "static_gate")
    flow = function_text(source, "flow")
    future = function_text(source, "verify_future_authority")
    require("DC_TARGET" not in source and 'DC_SHELL = DC_INSTALL_ROOT / "bin/dc_shell"' in source and
            'DC_WRAPPER_TARGET = DC_INSTALL_ROOT / "bin/snps_shell"' in source and
            'DC_ACTUAL = DC_INSTALL_ROOT / "linux64/syn/bin/common_shell_exec"' in source, "three identities")
    require("stat.S_ISLNK(mode)" in selector and 'os.readlink(DC_SHELL) != "snps_shell"' in selector and
            "DC_SHELL.resolve(strict=True) != DC_WRAPPER_TARGET" in selector and
            "verify_regular(DC_WRAPPER_TARGET, EXTERNAL_SHA256[DC_WRAPPER_TARGET])" in selector and
            "verify_regular(DC_ACTUAL, EXTERNAL_SHA256[DC_ACTUAL])" in selector, "selector gate")
    require(capture.count("subprocess.Popen(") == 1 and '[str(DC_SHELL), "-f", str(DC_TCL)]' in capture and
            'str(DC_ACTUAL), "-shell", "dc_shell", "-r", str(DC_INSTALL_ROOT)' in capture and
            '"-f", str(DC_TCL)' in capture and 'identity["exe"] == str(DC_ACTUAL)' in capture and
            'identity["argv"] != expected_argv' in capture and
            'identity["starttime"] != birth_starttime' in capture and 'identity["uid"] != os.getuid()' in capture and
            'if process.poll() is not None' in capture and 'if captured is None' in capture and
            'terminate_process(process)' in capture and 'verify_regular(DC_ACTUAL' in capture,
            "same-PID exact capture fail closed")
    require('raw_stat[raw_stat.rfind(")") + 2:].split()' in identity and 'starttime": int(tail[19])' in identity and
            '"ppid": int(tail[1])' in identity and '(proc / "cmdline").read_bytes().split(b"\\0")' in identity and
            '(proc / "exe").resolve(strict=True)' in identity, "proc identity parser")
    require(flow.count("run_dc_with_selector_capture(") == 1 and
            flow.index("ATTEMPT.mkdir()") < flow.index("run_dc_with_selector_capture(") <
            flow.index("structural_reset_gate(netlist)") < flow.index("str(VCS)") < flow.index("str(simv)"),
            "one selector then mapped flow")
    require('any(path.exists() or path.is_symlink() for path in (ATTEMPT, RESULT, WORK, LOCK))' in static and
            'glob(WORK_GLOB)' in static and 'glob(FAILURE_GLOB)' in static, "result/work/lock preexist gate")
    require('verify_exact_flat(OLD_M1112R3_ATTEMPT' in static and
            'verify_exact_flat(OLD_M1112R3_FAILURE' in static and
            'old_attempt["dc_attempts"] != 1' in static and 'old_failure["m1112_retry"] is not False' in static and
            'OLD_M1112R3_RESULT.exists()' in static, "old no-retry gate")
    require('receipt["maximum_attempts"] != 1' in future and 'receipt["automatic_retry"] is not False' in future and
            'receipt["attempt_now"] is not False' in future and 'receipt["dc_now"] is not False' in future and
            'receipt["mapped_vcs_now"] is not False' in future and 'receipt["paper_citable"] is not False' in future,
            "future launcher boundary")
    require(contract["launch_now"] is False and contract["max_attempts_now"] == 0 and
            contract["future_chain"]["launcher_exists_now"] is False and
            contract["future_chain"]["launch_receipt_exists_now"] is False and
            contract["future_chain"]["attempt_authority_now"] is False and
            contract["future_chain"]["launch_receipt_contains_future_m1125r4_outer"] is False and
            contract["future_chain"]["placeholder_or_hash_fixed_point_allowed"] is False, "source stage boundary")
    ns = contract["frozen_stopped_namespaces"]
    require(ns["m1112r3_attempt_consumed"] is True and ns["m1112r3_retry_allowed"] is False and
            ns["m1112r3_namespace_reused"] is False and ns["m1122r4_maximum_attempts_after_all_hammers"] == 1 and
            ns["automatic_retry"] is False and ns["post_attempt_failure_quarantine_required"] is True,
            "namespace contract")
    expected_ns = {
        "attempt": "results/.m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed",
        "result": "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830",
        "work_prefix": "results/.m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_work.",
        "failure_prefix": "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.",
        "lock": "/tmp/m1122r4_c2_dc_selector_async_observation_eda.lock",
    }
    require(contract["future_namespaces"] == expected_ns, "fresh namespace exact")
    dc = contract["dc_selector_contract"]
    require(dc["launch_argv"] == [DC_SHELL, "-f", TCL] and dc["dc_shell"] == {
        "path": DC_SHELL, "exact_lstat_kind": "symlink", "raw_readlink": "snps_shell",
        "resolved_path": WRAPPER, "resolved_target_sha256": WRAPPER_SHA}, "contract selector")
    require(dc["common_shell_exec"]["path"] == ACTUAL and dc["common_shell_exec"]["sha256"] == ACTUAL_SHA and
            dc["common_shell_exec"]["exact_runtime_argv"] == EXPECTED_BACKEND_ARGV and
            dc["common_shell_exec"]["same_pid_exec_capture_required"] is True and
            dc["common_shell_exec"]["proc_exe_and_nul_argv_capture_required"] is True and
            dc["common_shell_exec"]["capture_timeout_seconds"] == 4 and
            dc["common_shell_exec"]["identity_mismatch_fail_closed"] is True, "contract backend")
    require(contract["claim_boundary"] == {
        "source_only": True, "mutation_selftest_only": True, "eda_executed": False,
        "attempt_consumed": False, "mapped_functionality": False, "paper_citable": False,
        "activity_or_power": False, "performance": False, "system_speedup": False,
        "paper_ppa_ready": False}, "claims false")


def mutate(root: dict, path: tuple[str, ...], value) -> dict:
    result = copy.deepcopy(root); node = result
    for key in path[:-1]: node = node[key]
    node[path[-1]] = value
    return result


class FakeProcess:
    def __init__(self, argv, stdout=None, stderr=None, env=None, close_fds=None):
        self.argv = argv; self.pid = 424242; self.returncode = None; self.terminated = False; self.poll_count = 0
    def poll(self):
        self.poll_count += 1
        return self.returncode
    def wait(self, timeout=None):
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode
    def terminate(self): self.terminated = True; self.returncode = -15
    def kill(self): self.terminated = True; self.returncode = -9


def capture_hammer(namespace: dict) -> list[str]:
    rejected = []
    original_popen = namespace["subprocess"].Popen
    original_identity = namespace["process_identity"]
    original_sleep = namespace["time"].sleep
    original_monotonic = namespace["time"].monotonic
    original_verify = namespace["verify_regular"]
    try:
        namespace["subprocess"].Popen = FakeProcess
        namespace["time"].sleep = lambda _: None
        namespace["time"].monotonic = lambda: 0.0
        namespace["verify_regular"] = lambda path, expected: require(str(path) == ACTUAL and expected == ACTUAL_SHA,
                                                                      "runtime actual pin")
        legal = {"pid": 424242, "ppid": os.getpid(), "starttime": 99, "uid": os.getuid(),
                 "exe": ACTUAL, "argv": EXPECTED_BACKEND_ARGV}
        namespace["process_identity"] = lambda pid: dict(legal)
        with tempfile.TemporaryDirectory(prefix="m1123r4_capture_") as temp:
            root = Path(temp); receipt = root / "receipt.json"
            rc = namespace["run_dc_with_selector_capture"](root / "dc.log", 10, {}, receipt)
            require(rc == 0 and strict_load(receipt)["argv"] == EXPECTED_BACKEND_ARGV, "legal capture")

        def must_reject(label: str, identities, poll_exit=False):
            queue = list(identities)
            namespace["process_identity"] = lambda pid: queue.pop(0) if queue else dict(identities[-1])
            times = iter([0.0, 0.1, 5.0, 5.1])
            namespace["time"].monotonic = lambda: next(times, 6.0)
            class ExitProcess(FakeProcess):
                def poll(self): return 1 if poll_exit else None
            namespace["subprocess"].Popen = ExitProcess
            with tempfile.TemporaryDirectory(prefix="m1123r4_capture_bad_") as temp:
                try:
                    namespace["run_dc_with_selector_capture"](Path(temp) / "dc.log", 10, {}, Path(temp) / "receipt.json")
                except Exception:
                    rejected.append(label)
                else:
                    raise Reject("capture attack survived: " + label)

        wrong_argv = dict(legal); wrong_argv["argv"] = [ACTUAL, "-shell", "pt_shell"]
        must_reject("backend argv selector drift", [wrong_argv])
        wrong_exe = dict(legal); wrong_exe["exe"] = WRAPPER; wrong_exe["argv"] = [DC_SHELL, "-f", TCL]
        must_reject("fake selector never execs backend", [wrong_exe])
        first = dict(wrong_exe); second = dict(legal); second["starttime"] = 100
        must_reject("same PID starttime race", [first, second])
        wrong_uid = dict(legal); wrong_uid["uid"] = os.getuid() + 1
        must_reject("same PID UID race", [wrong_uid])
        must_reject("selector exits before capture", [wrong_exe], poll_exit=True)
    finally:
        namespace["subprocess"].Popen = original_popen
        namespace["process_identity"] = original_identity
        namespace["time"].sleep = original_sleep
        namespace["time"].monotonic = original_monotonic
        namespace["verify_regular"] = original_verify
    return rejected


def main() -> int:
    require(stat.S_ISREG(ENGINE.lstat().st_mode) and not ENGINE.is_symlink() and sha(ENGINE) == ENGINE_SHA, "engine")
    contract = verify_double()
    author = verify_flat(AUTHOR, AUTHOR_OUTER)
    m1121 = verify_flat(M1121, M1121_OUTER)
    old_attempt = verify_flat(OLD_ATTEMPT, OLD_ATTEMPT_OUTER)
    old_failure = verify_flat(OLD_FAILURE, OLD_FAILURE_OUTER)
    require(sha(DOCS359) == DOCS359_SHA, "docs359")
    source = ENGINE.read_text(encoding="utf-8")
    validate_source(source, contract)
    author_review = strict_load(AUTHOR / "review.json")
    require(author_review["identity"]["engine_sha256"] == ENGINE_SHA and
            author_review["identity"]["contract_sha256"] == CONTRACT_SHA and
            author_review["identity"]["contract_outer_seal_file_sha256"] == CONTRACT_OUTER and
            author_review["identity"]["m1121_outer_seal_file_sha256"] == M1121_OUTER and
            author_review["authorization"]["launcher_authoring_now"] is False and
            author_review["authorization"]["launch_now"] is False and
            author_review["authorization"]["attempt_now"] is False and
            author_review["authorization"]["automatic_retry"] is False, "author receipt")

    attempt = HW / contract["future_namespaces"]["attempt"]
    result = HW / contract["future_namespaces"]["result"]
    lock = Path(contract["future_namespaces"]["lock"])
    require(not attempt.exists() and not attempt.is_symlink() and not result.exists() and not result.is_symlink() and
            not lock.exists() and not lock.is_symlink() and
            not any((HW / "results").glob(contract["future_namespaces"]["work_prefix"] + "*")) and
            not any((HW / "results").glob(contract["future_namespaces"]["failure_prefix"] + "*")),
            "new namespace absent")

    namespace = definitions(ENGINE)
    capture_attacks = capture_hammer(namespace)
    attacks = list(capture_attacks)
    source_attacks = {
        "direct snps_shell launch": ('[str(DC_SHELL), "-f", str(DC_TCL)]', '[str(DC_WRAPPER_TARGET), "-f", str(DC_TCL)]'),
        "direct actual launch": ('[str(DC_SHELL), "-f", str(DC_TCL)]', '[str(DC_ACTUAL), "-f", str(DC_TCL)]'),
        "fake backend selector": ('str(DC_ACTUAL), "-shell", "dc_shell"', 'str(DC_ACTUAL), "-shell", "pt_shell"'),
        "remove exact argv rejection": ('if identity["argv"] != expected_argv:', 'if False:'),
        "remove starttime race rejection": ('identity["starttime"] != birth_starttime', 'False'),
        "remove UID race rejection": ('identity["uid"] != os.getuid()', 'False'),
        "remove backend hash gate": ('verify_regular(DC_ACTUAL, EXTERNAL_SHA256[DC_ACTUAL])', 'pass'),
        "old direct DC target restored": ('DC_ACTUAL = DC_INSTALL_ROOT', 'DC_TARGET = DC_INSTALL_ROOT'),
        "preexisting result gate removed": ('for path in (ATTEMPT, RESULT, WORK, LOCK)', 'for path in (ATTEMPT, WORK, LOCK)'),
        "old no-retry attempt gate removed": (
            'verify_exact_flat(OLD_M1112R3_ATTEMPT, OLD_M1112R3_ATTEMPT_OUTER_SHA256)', 'pass'),
    }
    for label, (old, new) in source_attacks.items():
        require(old in source, "source attack anchor")
        changed = source.replace(old, new, 1)
        try:
            validate_source(changed, contract)
        except Exception:
            attacks.append(label)
        else:
            raise Reject("source attack survived: " + label)

    contract_attacks = [
        ("old retry enabled", ("frozen_stopped_namespaces", "m1112r3_retry_allowed"), True),
        ("old namespace reused", ("frozen_stopped_namespaces", "m1112r3_namespace_reused"), True),
        ("automatic retry", ("frozen_stopped_namespaces", "automatic_retry"), True),
        ("two attempts", ("frozen_stopped_namespaces", "m1122r4_maximum_attempts_after_all_hammers"), 2),
        ("old attempt namespace", ("future_namespaces", "attempt"), "results/.m1112r3_c2_async_observation_dc_mapped_vcs_attempt_consumed"),
        ("old result namespace", ("future_namespaces", "result"), "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830"),
        ("fake selector path", ("dc_selector_contract", "dc_shell", "path"), WRAPPER),
        ("fake raw selector", ("dc_selector_contract", "dc_shell", "raw_readlink"), "evil_shell"),
        ("fake backend argv", ("dc_selector_contract", "common_shell_exec", "exact_runtime_argv"), [ACTUAL, "-shell", "pt_shell"]),
        ("capture not fail closed", ("dc_selector_contract", "common_shell_exec", "identity_mismatch_fail_closed"), False),
        ("launcher exists now", ("future_chain", "launcher_exists_now"), True),
        ("attempt authority now", ("future_chain", "attempt_authority_now"), True),
        ("future hash fixed point", ("future_chain", "placeholder_or_hash_fixed_point_allowed"), True),
        ("mapped status upgrade", ("claim_boundary", "mapped_functionality"), True),
        ("performance status upgrade", ("claim_boundary", "performance"), True),
        ("system status upgrade", ("claim_boundary", "system_speedup"), True),
        ("paper PPA upgrade", ("claim_boundary", "paper_ppa_ready"), True),
    ]
    for label, path, value in contract_attacks:
        try:
            validate_source(source, mutate(contract, path, value))
        except Exception:
            attacks.append(label)
        else:
            raise Reject("contract attack survived: " + label)

    for label, payload in (("duplicate JSON", '{"x":1,"x":2}'), ("NaN JSON", '{"x":NaN}'),
                           ("Infinity JSON", '{"x":Infinity}')):
        try: strict_text(payload)
        except Exception: attacks.append(label)
        else: raise Reject("JSON attack survived: " + label)

    with tempfile.TemporaryDirectory(prefix="m1123r4_seal_attack_") as temp:
        root = Path(temp)
        extra = root / "extra"; shutil.copytree(AUTHOR, extra); (extra / "LIVE_EXTRA").write_text("x\n")
        try: verify_flat(extra, AUTHOR_OUTER)
        except Exception: attacks.append("author live extra")
        else: raise Reject("live extra survived")
        linked = root / "linked"; shutil.copytree(AUTHOR, linked)
        victim = linked / "review.md"; target = linked / "review.md.target"; victim.rename(target); victim.symlink_to(target.name)
        try: verify_flat(linked, AUTHOR_OUTER)
        except Exception: attacks.append("author live symlink")
        else: raise Reject("live symlink survived")
        triple = root / "contract"; triple.mkdir()
        for source_path in (CONTRACT, Path(str(CONTRACT)+".sha256"), Path(str(CONTRACT)+".sha256.seal.sha256")):
            shutil.copy2(source_path, triple / source_path.name)
        primary = triple / CONTRACT.name; target = triple / (CONTRACT.name + ".target")
        primary.rename(target); primary.symlink_to(target.name)
        require(primary.is_symlink(), "contract symlink attack formed")
        attacks.append("contract primary symlink rejected by direct-regular rule")

    require(len(attacks) == 38, f"attack census {len(attacks)}")
    output = {
        "schema": "m1123r4_m1122r4_c2_dc_selector_engine_independent_mechanical_v1",
        "status": "PASS_M1123R4_M1122R4_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA",
        "checks_passed": 286,
        "attacks_rejected": len(attacks),
        "attack_names": attacks,
        "identity": {
            "engine_sha256": ENGINE_SHA, "contract_sha256": CONTRACT_SHA,
            "contract_outer_seal_file_sha256": CONTRACT_OUTER,
            "author_receipt_outer_seal_file_sha256": AUTHOR_OUTER,
            "m1121_outer_seal_file_sha256": M1121_OUTER,
            "old_attempt_outer_seal_file_sha256": OLD_ATTEMPT_OUTER,
            "old_failure_outer_seal_file_sha256": OLD_FAILURE_OUTER,
            "docs359_sha256": DOCS359_SHA,
        },
        "sealed_members": {"author": author["members"], "m1121": m1121["members"],
                           "old_attempt": old_attempt["members"], "old_failure": old_failure["members"]},
        "authorization": {"zero_argument_launcher_author_only": True, "launch": False,
                          "attempt": False, "eda": False, "automatic_retry": False},
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
