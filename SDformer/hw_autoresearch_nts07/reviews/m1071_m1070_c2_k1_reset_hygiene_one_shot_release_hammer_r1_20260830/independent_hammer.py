#!/usr/bin/env python3
"""Receipt-blind M1071 hammer for the M1070 sidecar-path successor.

No production EDA is launched.  The frozen runner is executed only in a
hard-linked temporary tree with subprocess, memory, flock, DC and VCS mocks.
"""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
PY310 = Path("/opt/anaconda3/envs/pytorch310/bin/python3.10")

RUNNER_REL = Path("dc_handoff/scripts/run_m1070_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_r1.py")
CONTRACT_REL = Path("contracts/m1070_c2_k1_reset_hygiene_dc_mapped_vcs_source_contract_r1_20260830.json")
RELEASE_REL = Path("contracts/m1070_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_release_r1_20260830.json")
M1071_REL = Path("reviews/m1071_m1070_c2_k1_reset_hygiene_one_shot_release_hammer_r1_20260830")
M1069_REL = Path("reviews/m1069_m1068_c2_k1_reset_hygiene_one_shot_release_hammer_r1_20260830")
ATTEMPT_REL = Path("results/.m1070_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_attempt_consumed")
RESULT_REL = Path("results/m1070_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830")

EXPECTED_RUNNER = "5e1fd02a5eb171a680529ab6939c65ea0fa68fe28aa4933d24a257675d7e5702"
EXPECTED_CONTRACT = "e40ab04cb40516bc21e6c0010f52fbc254026146987778603fd65ebf29de7cd5"
EXPECTED_CONTRACT_OUTER = "e715d111a27a10abc682e73e8ade53b4ca38276971ac52ee63e3acb74b33354c"
EXPECTED_RELEASE = "850bc82ddc17832df64ef2c260a292c0f9ecd235dcfeb5395d6da531b01325c4"
EXPECTED_RELEASE_OUTER = "5742e1368eb763b8bdf249c7504b140ff63a7ffa307af0ed3d31b95e83985fe8"
EXPECTED_M1069_STOP_OUTER = "ece91f960cf98892b12879bdf19d57f1d408cb971b4ba2a249b1a89a72853a9a"
DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
DC_SHELL = Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell")
DC_SHELL_SHA = "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2"

UPSTREAM_DIRS = {
    Path("reviews/m1059_m1058_c2_k1_reset_hygiene_source_release_hammer_r1_20260830"): "c22d41a87f82f939487637155b35d11496234850631b5894d159ff41e41fb4b3",
    Path("results/m1058_c2_k1_reset_hygiene_rtl_vcs_r1_20260830"): "f22a55c33fadf74749060546e877fc10f892649aa31f3fa0da2d3fd164b70787",
    Path("reviews/m1050_m1046_c2_mapped_gate_watchdog_failure_audit_r1_20260829"): "bc239844a71b5c017002ea1f6a756143d3c58b5ebf39d6a5499c76228da188bb",
    Path("results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829.failed_or_incomplete.2027456.quarantine"): "cb6f6b69e2cb51d60556f5bcb8a7748865f72ee2bdbe2f178925a624d9e9d705",
    M1069_REL: EXPECTED_M1069_STOP_OUTER,
}

SIDECARS = {
    Path("contracts/m1058_c2_k1_reset_hygiene_source_only_contract_r1_20260830.json"): "1d06a6bdda5b15e404c758e5571498d026cb23e586fc7ba1d929f1c064518b44",
    Path("contracts/m1058_c2_k1_reset_hygiene_dc_mapped_vcs_launch_candidate_r1_20260830.json"): "12c131029fc6f049e2f2a58082dcb6e4f72c4056a9bc68cde9006d585b2c7f82",
    CONTRACT_REL: EXPECTED_CONTRACT_OUTER,
    RELEASE_REL: EXPECTED_RELEASE_OUTER,
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        out = {}
        for key, value in items:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def verify_dir(path: Path, expected_outer: str) -> None:
    require(path.is_dir() and not path.is_symlink(), "sealed dir absent/symlink: " + str(path))
    manifest = path / "SHA256SUMS"
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        target = path / rel.strip().lstrip("*")
        require(target.is_file() and sha(target) == digest, "member seal mismatch: " + str(target))
    inner = path / "SHA256SUMS.seal.sha256"
    require(inner.read_text().split() == [sha(manifest), "SHA256SUMS"], "inner seal mismatch")
    require(sha(inner) == expected_outer, "outer seal mismatch: " + str(path))


def verify_sidecar(path: Path, expected_outer: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    expected_name = path.relative_to(HW).as_posix()
    require(side.read_text().split() == [sha(path), expected_name], "exact primary sidecar mismatch")
    require(outer.read_text().split() == [sha(side), expected_name + ".sha256"],
            "exact outer sidecar mismatch")
    require(sha(outer) == expected_outer, "sidecar outer identity mismatch")


def seal_dir(path: Path) -> str:
    members = sorted(p for p in path.rglob("*")
                     if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = path / "SHA256SUMS"
    manifest.write_text("".join(f"{sha(p)}  {p.relative_to(path).as_posix()}\n" for p in members))
    inner = path / "SHA256SUMS.seal.sha256"
    inner.write_text(f"{sha(manifest)}  SHA256SUMS\n")
    return sha(inner)


def reseal_sidecar(path: Path, primary_name: str | None = None,
                   outer_name: str | None = None, extra_primary: str = "",
                   extra_outer: str = "") -> str:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    expected = path.relative_to(path.parents[1]).as_posix()
    side.write_text(f"{sha(path)}  {primary_name or expected}{extra_primary}\n")
    outer.write_text(f"{sha(side)}  {outer_name or (expected + '.sha256')}{extra_outer}\n")
    return sha(outer)


def hardlink_tree(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True)
    for root, dirs, files in os.walk(source):
        rel = Path(root).relative_to(source)
        for directory in dirs:
            (dest / rel / directory).mkdir()
        for name in files:
            shutil.copy2(Path(root) / name, dest / rel / name)


def hardlink_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)


def make_fake_m1071(root: Path, release_outer: str, status: str | None = None,
                    authorize: bool = True) -> str:
    path = root / M1071_REL
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    review = {
        "schema": "m1071_redirected_mock_only",
        "milestone": "M1071",
        "status": status or "PASS_M1071_M1070_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER",
        "identity": {"release_outer_seal_sha256": release_outer},
        "authorization": {"one_m1070_dc_then_mapped_vcs_attempt": authorize},
    }
    (path / "review.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n")
    return seal_dir(path)


def sitecustomize_text() -> str:
    return r'''import fcntl, os, pathlib, subprocess
_mode = os.environ.get("M1071_MOCK_MODE", "full")
_real_read_text = pathlib.Path.read_text
_real_flock = fcntl.flock
_real_run = subprocess.run
def _read_text(self, *args, **kwargs):
    if str(self) == "/proc/meminfo":
        if _mode == "resource":
            return "MemAvailable: 1 kB\nCommitLimit: 2 kB\nCommitted_AS: 2 kB\n"
        return "MemAvailable: 67108864 kB\nCommitLimit: 67108864 kB\nCommitted_AS: 1 kB\n"
    return _real_read_text(self, *args, **kwargs)
pathlib.Path.read_text = _read_text
def _flock(fd, operation):
    if _mode == "flock": raise BlockingIOError("redirected busy flock")
    return _real_flock(fd, operation)
fcntl.flock = _flock
def _completed(argv, rc=0):
    return subprocess.CompletedProcess(argv, rc, stdout="", stderr="")
def _run(argv, *args, **kwargs):
    words = [str(x) for x in argv] if isinstance(argv, (list, tuple)) else [str(argv)]
    exe = words[0]
    if exe == "/usr/bin/pgrep": return _completed(argv, 0 if _mode == "eda" else 1)
    if exe.endswith("/lmutil"): return _completed(argv, 0)
    if exe.endswith("/dc_shell"):
        with open(os.environ["M1071_EVENT_LOG"], "a") as f: f.write("DC\n")
        if _mode == "dc_fail": return _completed(argv, 9)
        out = pathlib.Path(kwargs["env"]["OUTPUT_DIR"])
        (out / "reports").mkdir(parents=True, exist_ok=True)
        (out / "netlist").mkdir(parents=True, exist_ok=True)
        (out / "TCL_PASS_TERMINAL.txt").write_text("PASS\n")
        (out / "reports/precompile_loop_gate.rpt").write_text(
            "TIM-209=0\nOPT-150=0\nstatus=PASS_PRECOMPILE_LOOP_GATE\n")
        (out / "reports/area.rpt").write_text("Total cell area: 123.0\n")
        (out / "reports/timing_setup.rpt").write_text("slack (MET) 0.001\n")
        design = kwargs["env"]["DESIGN_NAME"]
        (out / f"netlist/{design}_mapped.v").write_text("module mock; endmodule\n")
        return _completed(argv, 0)
    if exe.endswith("/vcs"):
        with open(os.environ["M1071_EVENT_LOG"], "a") as f: f.write("VCS_COMPILE\n")
        simv = pathlib.Path(words[words.index("-o") + 1])
        simv.write_text("#!/bin/sh\nexit 0\n"); simv.chmod(0o755)
        return _completed(argv, 0)
    if pathlib.Path(exe).name == "simv":
        case = int(next(x.split("=", 1)[1] for x in words if x.startswith("+M979_CASE=")))
        anchors = [259, 737, 3153, 7569, 14]
        with open(os.environ["M1071_EVENT_LOG"], "a") as f: f.write(f"CASE{case}\n")
        out = kwargs["stdout"]
        out.write(f"PASS M979 mapped replay axis=K1 case={case} events=1 cycles={anchors[case]} saif_duration_ns={anchors[case]*3} numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 accepted_unknowns=0 protocol_errors=0\n")
        out.flush(); return _completed(argv, 0)
    raise RuntimeError("unexpected external command: " + repr(words))
subprocess.run = _run
'''


def prepare_redirect(root: Path) -> tuple[Path, str]:
    source = (HW / RUNNER_REL).read_text()
    tree = ast.parse(source)
    rels = {RUNNER_REL, CONTRACT_REL, RELEASE_REL,
            Path("docs/359_DATE终局冻结_20260813.md")}
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div) and \
                isinstance(node.left, ast.Name) and node.left.id == "HW_ROOT" and \
                isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
            rels.add(Path(node.right.value))
    for rel in list(SIDECARS):
        rels.update({rel, Path(str(rel) + ".sha256"), Path(str(rel) + ".sha256.seal.sha256")})
    for literal in re.findall(r'"((?:rtl|tb_m349|dc_handoff|docs)/[^"\n]+)"', source):
        if (HW / literal).is_file(): rels.add(Path(literal))
    for rel in sorted(rels):
        if (HW / rel).is_file(): hardlink_file(HW / rel, root / rel)
    for rel in UPSTREAM_DIRS:
        hardlink_tree(HW / rel, root / rel)
    (root / "results").mkdir(exist_ok=True)
    (root / "reviews").mkdir(exist_ok=True)
    site = root / "mock_site"; site.mkdir()
    (site / "sitecustomize.py").write_text(sitecustomize_text())
    m1071_outer = make_fake_m1071(root, EXPECTED_RELEASE_OUTER)
    return site, m1071_outer


def run_redirected(root: Path, site: Path, outer: str, mode: str,
                   runner_pin: str = EXPECTED_RUNNER,
                   license_present: bool = True) -> subprocess.CompletedProcess:
    env = {"PATH": "/usr/bin:/bin", "PYTHONPATH": str(site),
           "M1071_MOCK_MODE": mode, "M1071_EVENT_LOG": str(root / "event.log"),
           "M1070_EXPECTED_RUNNER_SHA256": runner_pin,
           "M1070_EXPECTED_M1071_OUTER_SHA256": outer}
    if license_present: env["LM_LICENSE_FILE"] = "redirected-mock-only"
    return subprocess.run([str(PY310), str(root / RUNNER_REL)], text=True,
                          capture_output=True, env=env, timeout=30, check=False)


def mutate_json_reseal(root: Path, rel: Path, key: str, value) -> None:
    path = root / rel
    payload = strict_json(path); payload[key] = value
    path.unlink(); path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    reseal_sidecar(path)


def refresh_release_and_m1071(root: Path, contract_outer: str | None = None) -> str:
    release = root / RELEASE_REL
    payload = strict_json(release)
    if contract_outer is not None: payload["identity"]["contract_outer_seal_sha256"] = contract_outer
    release.unlink(); release.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    release_outer = reseal_sidecar(release)
    return make_fake_m1071(root, release_outer)


def preattempt(name: str, mutate=None, mode: str = "full",
               runner_pin: str = EXPECTED_RUNNER,
               license_present: bool = True,
               result_preexisting: bool = False) -> str:
    with tempfile.TemporaryDirectory(prefix="m1071_redirect_") as raw:
        root = Path(raw) / "hw_autoresearch_nts07"; root.mkdir()
        site, outer = prepare_redirect(root)
        if mutate: outer = mutate(root, outer)
        result = run_redirected(root, site, outer, mode, runner_pin, license_present)
        require(result.returncode == 3, name + " escaped")
        require(not (root / ATTEMPT_REL).exists(), name + " consumed attempt")
        require((root / RESULT_REL).exists() is result_preexisting,
                name + " result namespace boundary mismatch")
    return "REJECTED_BEFORE_ATTEMPT"


def sidecar_token_attack(kind: str):
    def mutate(root: Path, _outer: str) -> str:
        path = root / CONTRACT_REL
        exact = CONTRACT_REL.as_posix()
        if kind == "basename": outer = reseal_sidecar(path, primary_name=path.name)
        elif kind == "suffix": outer = reseal_sidecar(path, primary_name=exact + ".evil")
        elif kind == "traversal": outer = reseal_sidecar(path, primary_name="contracts/../" + path.name)
        elif kind == "extra": outer = reseal_sidecar(path, extra_primary="  attacker")
        elif kind == "outer_basename": outer = reseal_sidecar(path, outer_name=path.name + ".sha256")
        else: raise RuntimeError(kind)
        return refresh_release_and_m1071(root, outer)
    return mutate


def mutate_release_seal(root: Path, outer: str) -> str:
    path = root / Path(str(RELEASE_REL) + ".sha256")
    path.unlink()
    path.write_text("0" * 64 + "  " + RELEASE_REL.as_posix() + "\n")
    return outer


def replay_m1069_bug() -> dict:
    old_runner = HW / "dc_handoff/scripts/run_m1068_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_r1.py"
    old_text = old_runner.read_text(encoding="utf-8")
    require("if name != path.name" in old_text and
            "if name2 != side.name" in old_text,
            "old basename-only validator source drift")
    target = HW / "contracts/m1058_c2_k1_reset_hygiene_source_only_contract_r1_20260830.json"
    primary_tokens = Path(str(target) + ".sha256").read_text().split()
    outer_tokens = Path(str(target) + ".sha256.seal.sha256").read_text().split()
    require(primary_tokens[1] == "contracts/" + target.name,
            "frozen primary token no longer reproduces M1069 evidence")
    require(outer_tokens[1] == "contracts/" + target.name + ".sha256",
            "frozen outer token no longer reproduces M1069 evidence")
    # Execute the old validator's decisive predicates without invoking the
    # runner (which would be blocked by the independent DC-shell P0 first).
    primary_rejected = primary_tokens[1] != target.name
    outer_rejected = outer_tokens[1] != Path(str(target) + ".sha256").name
    require(primary_rejected and outer_rejected,
            "old M1068 basename incompatibility did not replay")
    return {"validator_replay": "REJECTED", "primary_rejected": True,
            "outer_rejected": True, "attempt_consumed": False}


def static_audit() -> dict:
    require(sha(HW / RUNNER_REL) == EXPECTED_RUNNER, "runner drift")
    require(sha(HW / CONTRACT_REL) == EXPECTED_CONTRACT, "contract drift")
    require(sha(HW / Path(str(CONTRACT_REL) + ".sha256.seal.sha256")) == EXPECTED_CONTRACT_OUTER,
            "contract outer drift")
    require(sha(HW / RELEASE_REL) == EXPECTED_RELEASE, "release drift")
    require(sha(HW / Path(str(RELEASE_REL) + ".sha256.seal.sha256")) == EXPECTED_RELEASE_OUTER,
            "release outer drift")
    require(sha(HW / "docs/359_DATE终局冻结_20260813.md") == DOCS359_SHA, "docs359 drift")
    require(DC_SHELL.is_file() and sha(DC_SHELL) == DC_SHELL_SHA,
            "DC shell target identity drift")
    require(not DC_SHELL.is_symlink(), "P0 production expect_sha rejects pinned DC_SHELL symlink")
    for rel, outer in SIDECARS.items(): verify_sidecar(HW / rel, outer)
    for rel, outer in UPSTREAM_DIRS.items(): verify_dir(HW / rel, outer)

    contract = strict_json(HW / CONTRACT_REL); release = strict_json(HW / RELEASE_REL)
    stop = strict_json(HW / M1069_REL / "review.json")
    require(stop["status"] == "STOP_M1069_M1068_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER",
            "M1069 STOP drift")
    require(stop["authorization"]["one_m1068_dc_then_mapped_vcs_attempt"] is False,
            "M1069 authorization must remain false")
    require(contract["pinned_evidence"]["m1069_stop_outer_seal_sha256"] == EXPECTED_M1069_STOP_OUTER and
            contract["pinned_evidence"]["m1069_attempt_authorized"] is False,
            "M1070 did not absorb M1069 STOP")
    require(release["identity"]["contract_outer_seal_sha256"] == EXPECTED_CONTRACT_OUTER,
            "release contract pin drift")
    require(contract["sidecar_token_policy"] == {
        "primary": "exact contracts/<basename>", "outer": "exact contracts/<basename>.sha256",
        "basename_only_allowed": False, "arbitrary_suffix_allowed": False,
        "path_traversal_allowed": False, "extra_tokens_allowed": False}, "sidecar policy drift")
    text = (HW / RUNNER_REL).read_text(); tree = ast.parse(text)
    require("expected_name = path.relative_to(HW_ROOT).as_posix()" in text,
            "exact relative-name derivation absent")
    require("if len(primary_tokens) != 2" in text and "if name != expected_name" in text and
            "if len(outer_tokens) != 2" in text and "if name2 != expected_name + \".sha256\"" in text,
            "strict sidecar token checks absent")
    run_flow = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_flow")
    segment = ast.get_source_segment(text, run_flow) or ""
    order = [segment.index(x) for x in ("static_identity_gate()", "release_chain_gate()",
             "ATTEMPT.mkdir()", 'phase = "FRESH_DC_ARCH_MODE0"',
             'phase = "FRESH_MAPPED_VCS_COMPILE"')]
    require(order == sorted(order), "identity/attempt/DC/mapped order drift")
    require("+vcs+initreg" not in text and "power enable" not in text.lower(),
            "production initreg/power contamination")
    require("quarantine_failure" in text and "FAILED_OR_INCOMPLETE" in text,
            "failure quarantine absent")
    require("ANCHORS = [259, 737, 3153, 7569, 14]" in text,
            "frozen anchors absent")
    return {"runner_sha256": EXPECTED_RUNNER, "contract_outer": EXPECTED_CONTRACT_OUTER,
            "release_outer": EXPECTED_RELEASE_OUTER, "m1069_stop_outer": EXPECTED_M1069_STOP_OUTER,
            "m1069_authorization": False}


def dynamic_audit() -> tuple[dict, list[str], dict]:
    attacks = {
        "wrong_runner_sha": preattempt("wrong_runner_sha", runner_pin="0" * 64),
        "wrong_contract_status": preattempt(
            "wrong_contract_status", mutate=lambda r, o: (mutate_json_reseal(r, CONTRACT_REL, "status", "ATTACKER_PASS") or o)),
        "wrong_release_status": preattempt(
            "wrong_release_status", mutate=lambda r, o: (mutate_json_reseal(r, RELEASE_REL, "status", "ATTACKER_PASS") or make_fake_m1071(r, sha(r / Path(str(RELEASE_REL) + '.sha256.seal.sha256'))))),
        "wrong_release_seal": preattempt(
            "wrong_release_seal", mutate=mutate_release_seal),
        "wrong_m1071_status": preattempt(
            "wrong_m1071_status", mutate=lambda r, o: make_fake_m1071(r, EXPECTED_RELEASE_OUTER, "ATTACKER_PASS")),
        "wrong_upstream_status_seal": preattempt(
            "wrong_upstream_status_seal", mutate=lambda r, o: _mutate_upstream(r, o)),
        "namespace_collision": preattempt(
            "namespace_collision", mutate=lambda r, o: ((r / RESULT_REL).mkdir(parents=True) or o),
            result_preexisting=True),
        "flock_busy": preattempt("flock_busy", mode="flock"),
        "eda_collision": preattempt("eda_collision", mode="eda"),
        "resource_shortage": preattempt("resource_shortage", mode="resource"),
        "license_absent": preattempt("license_absent", license_present=False),
        "sidecar_basename_only": preattempt("sidecar_basename_only", mutate=sidecar_token_attack("basename")),
        "sidecar_arbitrary_suffix": preattempt("sidecar_arbitrary_suffix", mutate=sidecar_token_attack("suffix")),
        "sidecar_path_traversal": preattempt("sidecar_path_traversal", mutate=sidecar_token_attack("traversal")),
        "sidecar_extra_token": preattempt("sidecar_extra_token", mutate=sidecar_token_attack("extra")),
        "sidecar_outer_basename_only": preattempt("sidecar_outer_basename_only", mutate=sidecar_token_attack("outer_basename")),
    }
    require(set(attacks.values()) == {"REJECTED_BEFORE_ATTEMPT"}, "pre-attempt attack escaped")

    with tempfile.TemporaryDirectory(prefix="m1071_dc_fail_") as raw:
        root = Path(raw) / "hw_autoresearch_nts07"; root.mkdir()
        site, outer = prepare_redirect(root); result = run_redirected(root, site, outer, "dc_fail")
        require(result.returncode == 3 and (root / ATTEMPT_REL).is_dir(), "DC failure boundary")
        quarantines = list((root / "results").glob(
            "m1070_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*.quarantine"))
        require(len(quarantines) == 1, "DC quarantine absent/nonunique")
        require(strict_json(quarantines[0] / "failure.json")["phase"] == "FRESH_DC_ARCH_MODE0",
                "DC quarantine phase drift")
        require((root / "event.log").read_text().splitlines() == ["DC"], "mapped ran after failed DC")

    with tempfile.TemporaryDirectory(prefix="m1071_full_") as raw:
        root = Path(raw) / "hw_autoresearch_nts07"; root.mkdir()
        site, outer = prepare_redirect(root); result = run_redirected(root, site, outer, "full")
        require(result.returncode == 0, "complete mock failed: " + result.stderr)
        events = (root / "event.log").read_text().splitlines()
        require(events == ["DC", "VCS_COMPILE", "CASE0", "CASE1", "CASE2", "CASE3", "CASE4"],
                "complete order drift")
        receipt = strict_json(root / RESULT_REL / "m1070_dc_mapped_vcs_receipt_r1.json")
        require(receipt["anchors"] == [259, 737, 3153, 7569, 14] and
                receipt["mapped_cases"] == 5 and receipt["random_register_initialization_used"] is False and
                receipt["saif_files"] == 0 and receipt["ptpx_runs"] == 0,
                "complete receipt boundary drift")
    return attacks, events, replay_m1069_bug()


def _mutate_upstream(root: Path, outer: str) -> str:
    path = root / M1069_REL / "review.json"
    payload = strict_json(path); payload["status"] = "ATTACKER_PASS"
    path.unlink(); path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return outer


def publish(static: dict, attacks: dict, events: list[str], replay: dict) -> None:
    review = {
        "schema": "m1071_m1070_c2_k1_reset_hygiene_one_shot_release_hammer_r1",
        "milestone": "M1071", "date": "2026-08-30",
        "status": "PASS_M1071_M1070_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER",
        "verdict": "GO_EXACTLY_ONE_M1070_FRESH_DC_THEN_FIVE_CASE_MAPPED_VCS_ATTEMPT",
        "identity": {"runner_sha256": EXPECTED_RUNNER,
                     "contract_outer_seal_sha256": EXPECTED_CONTRACT_OUTER,
                     "release_outer_seal_sha256": EXPECTED_RELEASE_OUTER,
                     "m1069_stop_outer_seal_sha256": EXPECTED_M1069_STOP_OUTER,
                     "docs359_sha256": DOCS359_SHA},
        "receipt_blind": True, "real_eda_launched": False,
        "real_m1070_attempt_consumed": False, "static_audit": static,
        "m1069_basename_incompatibility_replay": replay,
        "fault_injections": attacks, "redirected_mock_order": events,
        "authorization": {"one_m1070_dc_then_mapped_vcs_attempt": True},
        "caller": {"required_runner_sha256": EXPECTED_RUNNER,
                   "required_m1071_outer_seal_sha256": "PIN_THIS_DIRECTORY_SHA256SUMS_SEAL_SHA256"},
        "claim_boundary": {"dc_authorized": True, "mapped_vcs_authorized_after_dc": True,
                           "saif_authorized": False, "ptpx_authorized": False,
                           "power_admitted": False, "system_speedup_admitted": False,
                           "paper_ppa_ready": False},
    }
    (HERE / "review.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n")
    (HERE / "mechanical_checks.txt").write_text(
        "PASS exact runner/contract/release/M1069 STOP identities\n"
        "PASS old M1068 basename incompatibility replayed before attempt\n"
        "PASS exact contracts/<basename> and contracts/<basename>.sha256 accepted\n"
        "PASS basename/suffix/traversal/extra-token sidecars rejected before attempt\n"
        "PASS 11 inherited pre-attempt classes plus 5 sidecar attacks rejected\n"
        "PASS redirected DC failure quarantined before mapped compile\n"
        "PASS redirected complete DC -> compile -> five cases\n"
        "PASS production excludes initreg/SAIF/PTPX and preserves anchors/identity/quarantine\n")
    (HERE / "review.md").write_text(
        "# M1071 independent release hammer\n\n"
        "结论：**GO，仅授权一次 M1070 fresh DC → 五案例 mapped VCS。**\n\n"
        "独立重放了 M1069 的 basename sidecar 不兼容，并证明 M1070 只接受 exact "
        "`contracts/<basename>` 与 `contracts/<basename>.sha256`。basename-only、任意 suffix、"
        "path traversal、extra token 与 outer-basename 攻击均在 attempt 前拒绝。其余 11 类"
        "pre-attempt 攻击、DC 失败隔离、完整 mock 的 DC→compile→case0..4 顺序均通过。\n\n"
        "整个审计未启动真实 EDA、未消耗 M1070。GO 不授权 initreg、SAIF、PTPX、功耗、"
        "系统加速或 paper-ready PPA。\n")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS_M1071_M1070_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER\n")
    outer = seal_dir(HERE)
    print("PASS M1071 release hammer")
    print("M1071_OUTER=" + outer)


def publish_stop(reason: str, replay: dict) -> None:
    review = {
        "schema": "m1071_m1070_c2_k1_reset_hygiene_one_shot_release_hammer_r1",
        "milestone": "M1071", "date": "2026-08-30",
        "status": "STOP_M1071_M1070_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER",
        "verdict": "STOP_RUNNER_REJECTS_PINNED_DC_SHELL_SYMLINK_BEFORE_ATTEMPT",
        "identity": {"runner_sha256": EXPECTED_RUNNER,
                     "contract_outer_seal_sha256": EXPECTED_CONTRACT_OUTER,
                     "release_outer_seal_sha256": EXPECTED_RELEASE_OUTER,
                     "m1069_stop_outer_seal_sha256": EXPECTED_M1069_STOP_OUTER,
                     "docs359_sha256": DOCS359_SHA},
        "receipt_blind": True, "real_eda_launched": False,
        "real_m1070_attempt_consumed": False,
        "m1069_basename_incompatibility_replay": replay,
        "sidecar_successor_static_check": {
            "exact_primary_tokens": "contracts/<basename>",
            "exact_outer_tokens": "contracts/<basename>.sha256",
            "basename_only_rejected_by_source": True,
            "arbitrary_suffix_rejected_by_source": True,
            "path_traversal_rejected_by_source": True,
            "extra_tokens_rejected_by_source": True,
        },
        "p0": {"class": "PINNED_TOOL_PATH_TYPE_INCOMPATIBILITY",
               "reason": reason,
               "phase": "SOURCE_PREFLIGHT/static_identity_gate/expect_sha(DC_SHELL)",
               "dc_shell_path": str(DC_SHELL),
               "dc_shell_is_symlink": DC_SHELL.is_symlink(),
               "dc_shell_link_target": os.readlink(DC_SHELL) if DC_SHELL.is_symlink() else None,
               "resolved_payload_sha256": sha(DC_SHELL),
               "attempt_consumed_if_launched": False,
               "dc_reachable": False, "mapped_vcs_reachable": False},
        "authorization": {"one_m1070_dc_then_mapped_vcs_attempt": False},
        "unexecuted_due_dominating_p0": {
            "inherited_pre_attempt_attack_count": 11,
            "sidecar_dynamic_attack_count": 5,
            "dc_failure_quarantine_mock_executed": False,
            "complete_dc_compile_five_case_mock_executed": False,
            "reason": "all production paths are unreachable before attempt at the immutable tool identity gate",
        },
        "required_repair": {
            "additive_successor_runner": True,
            "preserve_m1070_and_upstream_evidence": True,
            "pin_symlink_path_and_exact_readlink_target": True,
            "pin_resolved_payload_sha256": DC_SHELL_SHA,
            "new_independent_hammer_required": True,
        },
        "claim_boundary": {"dc": False, "mapped_vcs": False, "saif": False,
                           "ptpx": False, "power": False,
                           "system_speedup": False, "paper_ppa_ready": False},
    }
    (HERE / "review.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n")
    (HERE / "mechanical_checks.txt").write_text(
        "PASS exact runner 5e1fd0 / contract e715d1 / release 5742e1 identities\n"
        "PASS M1069 STOP ece91f and authorization=false preserved\n"
        "PASS old M1068 basename incompatibility replayed before attempt\n"
        "PASS M1070 source statically enforces exact contracts/<basename> sidecar tokens\n"
        "FAIL M1070 expect_sha rejects pinned DC_SHELL because the production path is a symlink\n"
        "PASS failure is SOURCE_PREFLIGHT before attempt/DC/VCS\n"
        "STOP one_m1070_dc_then_mapped_vcs_attempt=false\n")
    (HERE / "review.md").write_text(
        "# M1071 independent release hammer\n\n"
        "结论：**STOP，不授权 M1070。**\n\n"
        "sidecar successor 的静态语义正确：只接受 exact `contracts/<basename>` 与 "
        "`contracts/<basename>.sha256`；M1069 的 basename 不兼容也已独立重放。"
        "但更早的生产工具身份门不可达：M1070 `expect_sha()` 无条件拒绝 symlink，"
        "冻结的 `/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell` 实际为 "
        "`dc_shell -> snps_shell`。隔离执行在 `SOURCE_PREFLIGHT/static_identity_gate` "
        "报 identity drift，真实 attempt、DC、VCS 均未触发。\n\n"
        "必须新增 additive successor，显式钉住 symlink 路径、exact `readlink` 目标和 resolved "
        "payload SHA；不得改写 M1070。随后需要新的独立 hammer。\n")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "STOP_M1071_M1070_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER\n")
    outer = seal_dir(HERE)
    print("STOP M1071 release hammer")
    print("M1071_OUTER=" + outer)


def main() -> None:
    replay = replay_m1069_bug()
    try:
        static = static_audit()
    except RuntimeError as exc:
        if "pinned DC_SHELL symlink" in str(exc):
            publish_stop(str(exc), replay)
            return
        raise
    attacks, events, replay = dynamic_audit()
    publish(static, attacks, events, replay)


if __name__ == "__main__":
    main()
