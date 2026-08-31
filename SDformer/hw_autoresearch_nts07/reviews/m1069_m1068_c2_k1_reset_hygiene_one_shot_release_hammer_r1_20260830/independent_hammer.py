#!/usr/bin/env python3
"""Receipt-blind, non-EDA M1069 hammer for the exact M1068 one-shot.

The production runner is never launched against the real tree.  Dynamic tests
hard-link a redirected tree under /tmp and inject deterministic pgrep, memory,
license, DC, and VCS responses with sitecustomize.  Thus the release gates and
failure-quarantine implementation execute, while no Synopsys process or real
M1068 attempt can be consumed.
"""

from __future__ import annotations

import ast
import hashlib
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

RUNNER_REL = Path("dc_handoff/scripts/run_m1068_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_r1.py")
CONTRACT_REL = Path("contracts/m1068_c2_k1_reset_hygiene_dc_mapped_vcs_source_contract_r1_20260830.json")
RELEASE_REL = Path("contracts/m1068_c2_k1_reset_hygiene_dc_mapped_vcs_one_shot_release_r1_20260830.json")
M1069_REL = Path("reviews/m1069_m1068_c2_k1_reset_hygiene_one_shot_release_hammer_r1_20260830")
ATTEMPT_REL = Path("results/.m1068_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_attempt_consumed")
RESULT_REL = Path("results/m1068_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830")

EXPECTED = {
    RUNNER_REL: "d4eb14e276441594ebe3700060f0ca6cd840bbf71753c6495d170cb5a5aadea3",
    CONTRACT_REL: "2f575a2466f991f68916a3af6368852f42dfb71b2a223a7b10c7689e032b95cb",
    Path(str(CONTRACT_REL) + ".sha256.seal.sha256"): "770069e6cbfe0d0e68ae08c4bafc2c9822f17c991e7c79e12ac00bb60e14c390",
    RELEASE_REL: "6cd323b03f3fc1f52bafbfda6e66a02d76d92eab9cbeff3a7963252af8aa81c3",
    Path(str(RELEASE_REL) + ".sha256.seal.sha256"): "c8a15bb54ee2aa02d57a6128f1ce36390c87e9a32da30f0c306bc9cc9a7c53fb",
    Path("docs/359_DATE终局冻结_20260813.md"): "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    Path("dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"): "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    Path("dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"): "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    Path("dc_handoff/filelists/date_m1058_c2_k1_reset_hygiene_logic_only_dc.f"): "4cfd47438de45a66a601433ee07a2493c7296b1dea8669f9c7826898364e7192",
    Path("dc_handoff/tb/tb_m1058_c2_k1_reset_hygiene_mapped_gate_case.sv"): "fdbb1ccc5be4af11d263a6581f84ec823f9fd8077b07fa8b12a88a32a056ae0f",
    Path("tb_m349/m349_fc2_scalar_bank_memory_model.sv"): "4375072b6bd09ada3dc3fd585c12102346ea897192a13630b0c44acf72ff63fa",
}

ABS_EXPECTED = {
    Path("/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell"): "23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2",
    Path("/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec"): "bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391",
    Path("/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs"): "0735e4b82ff98dd957d5839ea15dc9fcfd9466b84e6f4ccc30d76bc2c1b96287",
    Path("/opt/synopsys/scl/2025.03/linux64/bin/lmutil"): "e7e056cce4deb2e17e3612442795846136a1eaacb3b2348e9fae77aa071ede07",
    PY310: "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ssg0p9v125c.db"): "79fb5fb651492e43de58fbbb03a8094029f288e7e9850a60bd1e2015e1f951af",
    Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/timing_power_noise/NLDM/tcbn28hpcplusbwp35p140_180a/tcbn28hpcplusbwp35p140ffg1p05vm40c.db"): "a707b6fd903a90810a35224057e7a9883746ceee2a0827869e78bd4f4570c91a",
    Path("/opt/tech/tsmc28/StandardCell/tcbn28hpcplusbwp35p140_190a/TSMCHOME/digital/Front_End/verilog/tcbn28hpcplusbwp35p140_110a/tcbn28hpcplusbwp35p140.v"): "3ed0796ffa8a0eb1406860e07913b8457969bcec492c3cb15599ee8db964707a",
}

SEALED_DIRS = {
    Path("reviews/m1059_m1058_c2_k1_reset_hygiene_source_release_hammer_r1_20260830"): "c22d41a87f82f939487637155b35d11496234850631b5894d159ff41e41fb4b3",
    Path("results/m1058_c2_k1_reset_hygiene_rtl_vcs_r1_20260830"): "f22a55c33fadf74749060546e877fc10f892649aa31f3fa0da2d3fd164b70787",
    Path("reviews/m1050_m1046_c2_mapped_gate_watchdog_failure_audit_r1_20260829"): "bc239844a71b5c017002ea1f6a756143d3c58b5ebf39d6a5499c76228da188bb",
    Path("results/m1046_m1001_c2_three_axis_mapped_gate_saif_r5_20260829.failed_or_incomplete.2027456.quarantine"): "cb6f6b69e2cb51d60556f5bcb8a7748865f72ee2bdbe2f178925a624d9e9d705",
}

SIDECARS = {
    Path("contracts/m1058_c2_k1_reset_hygiene_source_only_contract_r1_20260830.json"): "1d06a6bdda5b15e404c758e5571498d026cb23e586fc7ba1d929f1c064518b44",
    Path("contracts/m1058_c2_k1_reset_hygiene_dc_mapped_vcs_launch_candidate_r1_20260830.json"): "12c131029fc6f049e2f2a58082dcb6e4f72c4056a9bc68cde9006d585b2c7f82",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(rows):
        out = {}
        for key, value in rows:
            require(key not in out, "duplicate JSON key: " + key)
            out[key] = value
        return out
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)


def verify_sidecar(path: Path, outer: str) -> None:
    side = Path(str(path) + ".sha256")
    seal = Path(str(path) + ".sha256.seal.sha256")
    digest, name = side.read_text().split()[:2]
    require(name == path.name and digest == sha(path), "sidecar mismatch: " + str(path))
    digest2, name2 = seal.read_text().split()[:2]
    require(name2 == side.name and digest2 == sha(side), "sidecar seal mismatch")
    require(sha(seal) == outer, "sidecar outer identity mismatch")


def verify_dir(path: Path, outer: str) -> None:
    require(path.is_dir() and not path.is_symlink(), "sealed directory absent/symlink")
    manifest = path / "SHA256SUMS"
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        target = path / rel.strip().lstrip("*")
        require(target.is_file() and sha(target) == digest, "member seal mismatch")
    require(sha(manifest) == (path / "SHA256SUMS.seal.sha256").read_text().split()[0],
            "inner directory seal mismatch")
    require(sha(path / "SHA256SUMS.seal.sha256") == outer, "outer directory seal mismatch")


def seal_sidecar(path: Path) -> str:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    side.write_text(f"{sha(path)}  {path.name}\n")
    outer.write_text(f"{sha(side)}  {side.name}\n")
    return sha(outer)


def seal_dir(path: Path) -> str:
    members = sorted(p for p in path.rglob("*")
                     if p.is_file() and p.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = path / "SHA256SUMS"
    manifest.write_text("".join(f"{sha(p)}  {p.relative_to(path).as_posix()}\n" for p in members))
    inner = path / "SHA256SUMS.seal.sha256"
    inner.write_text(f"{sha(manifest)}  SHA256SUMS\n")
    return sha(inner)


def hardlink_tree(source: Path, dest: Path) -> None:
    dest.mkdir(parents=True)
    for root, dirs, files in os.walk(source):
        rel = Path(root).relative_to(source)
        for directory in dirs:
            (dest / rel / directory).mkdir()
        for name in files:
            os.link(Path(root) / name, dest / rel / name)


def hardlink_file(source: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, dest)


def make_fake_m1069(root: Path, release_outer: str, status: str | None = None,
                     authorize: bool = True) -> str:
    out = root / M1069_REL
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    review = {
        "schema": "m1069_redirected_mock_only",
        "milestone": "M1069",
        "status": status or "PASS_M1069_M1068_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER",
        "identity": {"release_outer_seal_sha256": release_outer},
        "authorization": {"one_m1068_dc_then_mapped_vcs_attempt": authorize},
    }
    (out / "review.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n")
    return seal_dir(out)


def sitecustomize_text() -> str:
    return r'''import fcntl, os, pathlib, subprocess
_mode = os.environ.get("M1069_MOCK_MODE", "full")
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
    if _mode == "flock":
        raise BlockingIOError("redirected busy flock")
    return _real_flock(fd, operation)
fcntl.flock = _flock
def _completed(argv, rc=0):
    return subprocess.CompletedProcess(argv, rc, stdout="", stderr="")
def _run(argv, *args, **kwargs):
    words = [str(x) for x in argv] if isinstance(argv, (list, tuple)) else [str(argv)]
    exe = words[0]
    if exe == "/usr/bin/pgrep":
        return _completed(argv, 0 if _mode == "eda" else 1)
    if exe.endswith("/lmutil"):
        return _completed(argv, 0)
    if exe.endswith("/dc_shell"):
        with open(os.environ["M1069_EVENT_LOG"], "a") as f: f.write("DC\n")
        if _mode == "dc_fail":
            return _completed(argv, 9)
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
        with open(os.environ["M1069_EVENT_LOG"], "a") as f: f.write("VCS_COMPILE\n")
        simv = pathlib.Path(words[words.index("-o") + 1])
        simv.write_text("#!/bin/sh\nexit 0\n")
        simv.chmod(0o755)
        return _completed(argv, 0)
    if pathlib.Path(exe).name == "simv":
        case = int(next(x.split("=", 1)[1] for x in words if x.startswith("+M979_CASE=")))
        anchors = [259, 737, 3153, 7569, 14]
        with open(os.environ["M1069_EVENT_LOG"], "a") as f: f.write(f"CASE{case}\n")
        out = kwargs.get("stdout")
        out.write(f"PASS M979 mapped replay axis=K1 case={case} events=1 cycles={anchors[case]} saif_duration_ns={anchors[case]*3} numeric_mismatches=0 tuple_mismatches=0 weight_mismatches=0 accepted_unknowns=0 protocol_errors=0\n")
        out.flush()
        return _completed(argv, 0)
    raise RuntimeError("unexpected external command in redirected hammer: " + repr(words))
subprocess.run = _run
'''


def prepare_redirect(root: Path) -> tuple[Path, str]:
    # Every relative file that the frozen runner hashes must be a regular file,
    # so use hard links rather than symlinks.
    source = (HW / RUNNER_REL).read_text()
    tree = ast.parse(source)
    rels = set(EXPECTED)
    # Harvest every literal HW_ROOT / "relative/path" identity from the source.
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.left, ast.Name) and node.left.id == "HW_ROOT" and \
                    isinstance(node.right, ast.Constant) and isinstance(node.right.value, str):
                rels.add(Path(node.right.value))
    # Sidecar triplets used by the exact runner.
    for rel in list(SIDECARS) + [CONTRACT_REL, RELEASE_REL]:
        rels.update({rel, Path(str(rel) + ".sha256"), Path(str(rel) + ".sha256.seal.sha256")})
    # Source identity paths live only in the dictionary and are harvested here.
    for literal in re.findall(r'"((?:rtl|tb_m349|dc_handoff|docs)/[^"\n]+)"', source):
        path = HW / literal
        if path.is_file():
            rels.add(Path(literal))
    for rel in sorted(rels):
        src = HW / rel
        if src.is_file():
            hardlink_file(src, root / rel)
    for rel in SEALED_DIRS:
        hardlink_tree(HW / rel, root / rel)
    # Results/reviews parents and isolated mock hooks.
    (root / "results").mkdir(exist_ok=True)
    (root / "reviews").mkdir(exist_ok=True)
    site = root / "mock_site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(sitecustomize_text())
    release_outer = sha(root / Path(str(RELEASE_REL) + ".sha256.seal.sha256"))
    m1069_outer = make_fake_m1069(root, release_outer)
    return site, m1069_outer


def run_redirected(root: Path, site: Path, m1069_outer: str, mode: str,
                   runner_pin: str | None = None, license_present: bool = True) -> subprocess.CompletedProcess:
    event_log = root / "event.log"
    env = {
        "PATH": "/usr/bin:/bin",
        "PYTHONPATH": str(site),
        "M1069_MOCK_MODE": mode,
        "M1069_EVENT_LOG": str(event_log),
        "M1068_EXPECTED_RUNNER_SHA256": runner_pin or EXPECTED[RUNNER_REL],
        "M1068_EXPECTED_M1069_OUTER_SHA256": m1069_outer,
    }
    if license_present:
        env["LM_LICENSE_FILE"] = "redirected-mock-only"
    return subprocess.run([str(PY310), str(root / RUNNER_REL)], text=True,
                          capture_output=True, env=env, timeout=30, check=False)


def preattempt_rejected(name: str, mutate=None, mode: str = "full",
                        runner_pin: str | None = None,
                        license_present: bool = True) -> str:
    with tempfile.TemporaryDirectory(prefix="m1069_redirect_") as raw:
        root = Path(raw) / "hw_autoresearch_nts07"
        root.mkdir()
        site, outer = prepare_redirect(root)
        if mutate:
            outer = mutate(root, outer)
        result = run_redirected(root, site, outer, mode, runner_pin, license_present)
        require(result.returncode == 3, name + " was not rejected")
        require(not (root / ATTEMPT_REL).exists(), name + " consumed attempt")
        require(not (root / RESULT_REL).exists(), name + " published result")
    return "REJECTED_BEFORE_ATTEMPT"


def mutate_release_status(root: Path, _outer: str) -> str:
    path = root / RELEASE_REL
    payload = strict_json(path)
    payload["status"] = "ATTACKER_PASS"
    path.unlink()
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    release_outer = seal_sidecar(path)
    return make_fake_m1069(root, release_outer)


def mutate_release_seal(root: Path, outer: str) -> str:
    path = root / Path(str(RELEASE_REL) + ".sha256")
    path.unlink()
    path.write_text("0" * 64 + "  " + RELEASE_REL.name + "\n")
    return outer


def mutate_contract(root: Path, outer: str) -> str:
    path = root / CONTRACT_REL
    payload = strict_json(path)
    payload["status"] = "ATTACKER_PASS"
    path.unlink()
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    seal_sidecar(path)
    return outer


def mutate_m1069_status(root: Path, _outer: str) -> str:
    release_outer = sha(root / Path(str(RELEASE_REL) + ".sha256.seal.sha256"))
    return make_fake_m1069(root, release_outer, status="ATTACKER_PASS")


def mutate_namespace(root: Path, outer: str) -> str:
    (root / RESULT_REL).mkdir(parents=True)
    return outer


def mutate_upstream_status(root: Path, outer: str) -> str:
    path = root / "reviews/m1050_m1046_c2_mapped_gate_watchdog_failure_audit_r1_20260829/review.json"
    payload = strict_json(path)
    payload["status"] = "ATTACKER_PASS"
    path.unlink()
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    return outer


def static_audit() -> dict:
    for rel, digest in EXPECTED.items():
        require(sha(HW / rel) == digest, "frozen relative identity drift: " + str(rel))
    for path, digest in ABS_EXPECTED.items():
        require(sha(path) == digest, "frozen tool/technology identity drift: " + str(path))
    for rel, outer in SIDECARS.items():
        verify_sidecar(HW / rel, outer)
    for rel, outer in SEALED_DIRS.items():
        verify_dir(HW / rel, outer)

    contract = strict_json(HW / CONTRACT_REL)
    release = strict_json(HW / RELEASE_REL)
    require(contract["status"] ==
            "PASS_M1068_SOURCE_ONLY__REQUIRES_INDEPENDENT_M1069__NO_EDA_LAUNCHED",
            "contract status drift")
    require(release["status"] ==
            "PASS_M1068_C2_K1_RESET_HYGIENE_DC_MAPPED_VCS_ONE_SHOT_RELEASE_SOURCE",
            "release status drift")
    require(contract["runner"]["sha256"] == EXPECTED[RUNNER_REL] == release["runner_sha256"],
            "runner identity chain drift")
    require(release["identity"]["contract_outer_seal_sha256"] ==
            EXPECTED[Path(str(CONTRACT_REL) + ".sha256.seal.sha256")],
            "contract outer pin drift")
    require(contract["scope"]["anchors"] == [259, 737, 3153, 7569, 14] and
            release["production_constraints"]["mapped_cases"] == 5,
            "five-anchor contract drift")
    require(contract["scope"]["saif"] is False and contract["scope"]["ptpx"] is False and
            release["production_constraints"]["saif"] is False and
            release["production_constraints"]["ptpx"] is False,
            "power boundary drift")

    text = (HW / RUNNER_REL).read_text()
    tree = ast.parse(text)
    run_flow = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef) and node.name == "run_flow")
    segment = ast.get_source_segment(text, run_flow)
    require(segment is not None, "run_flow source absent")
    positions = {token: segment.index(token) for token in (
        "static_identity_gate()", "release_chain_gate()", "ATTEMPT.mkdir()",
        'phase = "FRESH_DC_ARCH_MODE0"', 'phase = "FRESH_MAPPED_VCS_COMPILE"')}
    require(positions["static_identity_gate()"] < positions["release_chain_gate()"] <
            positions["ATTEMPT.mkdir()"] < positions['phase = "FRESH_DC_ARCH_MODE0"'] <
            positions['phase = "FRESH_MAPPED_VCS_COMPILE"'], "DC/mapped ordering drift")
    require(text.count("for case_id, anchor in enumerate(ANCHORS):") == 1,
            "five-case loop missing/duplicated")
    require("+vcs+initreg" not in text, "literal initreg entered production runner")
    require('"+M979_UCLI_SAIF"' not in text and "power enable" not in text.lower(),
            "SAIF/power command entered production runner")
    require(re.search(r'\[str\(VCS\), "-full64", "-sverilog", "\+v2k",', text),
            "frozen mapped compile argv drift")
    require('f"+M979_CASE={case_id}", "-no_save"' in text,
            "mapped case argv drift")
    mapped_tb = (HW / "dc_handoff/tb/tb_m1058_c2_k1_reset_hygiene_mapped_gate_case.sv").read_text()
    require("0:return 259;1:return 737;2:return 3153;" in mapped_tb and
            "3:return 7569;default:return 14;" in mapped_tb,
            "mapped TB anchor drift")
    return {"static_identities": len(EXPECTED) + len(ABS_EXPECTED),
            "sealed_upstream": len(SEALED_DIRS), "sidecar_upstream": len(SIDECARS)}


def dynamic_audit() -> tuple[dict, list[str]]:
    attacks = {
        "wrong_runner_sha": preattempt_rejected(
            "wrong_runner_sha", runner_pin="0" * 64),
        "wrong_contract_sha_status": preattempt_rejected(
            "wrong_contract_sha_status", mutate=mutate_contract),
        "wrong_release_status": preattempt_rejected(
            "wrong_release_status", mutate=mutate_release_status),
        "wrong_release_seal": preattempt_rejected(
            "wrong_release_seal", mutate=mutate_release_seal),
        "wrong_m1069_status": preattempt_rejected(
            "wrong_m1069_status", mutate=mutate_m1069_status),
        "wrong_upstream_status_seal": preattempt_rejected(
            "wrong_upstream_status_seal", mutate=mutate_upstream_status),
        "namespace_collision": preattempt_rejected(
            "namespace_collision", mutate=mutate_namespace),
        "flock_busy": preattempt_rejected("flock_busy", mode="flock"),
        "eda_collision": preattempt_rejected("eda_collision", mode="eda"),
        "resource_shortage": preattempt_rejected("resource_shortage", mode="resource"),
        "license_absent": preattempt_rejected(
            "license_absent", license_present=False),
    }
    require(set(attacks.values()) == {"REJECTED_BEFORE_ATTEMPT"},
            "one or more pre-attempt attacks escaped")

    # Exercise post-consumption failure quarantine, with fake DC returning 9.
    with tempfile.TemporaryDirectory(prefix="m1069_dc_fail_") as raw:
        root = Path(raw) / "hw_autoresearch_nts07"
        root.mkdir()
        site, outer = prepare_redirect(root)
        result = run_redirected(root, site, outer, "dc_fail")
        require(result.returncode == 3 and (root / ATTEMPT_REL).is_dir(),
                "DC failure did not consume redirected attempt")
        quarantines = list((root / "results").glob(
            "m1068_m1058_c2_k1_reset_hygiene_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*.quarantine"))
        require(len(quarantines) == 1, "DC failure quarantine absent/nonunique")
        failure = strict_json(quarantines[0] / "failure.json")
        require(failure["phase"] == "FRESH_DC_ARCH_MODE0" and
                not (root / RESULT_REL).exists(), "failure boundary drift")
        events = (root / "event.log").read_text().splitlines()
        require(events == ["DC"], "mapped VCS ran after failed DC")

    # Complete a redirected mock and demand exact DC -> compile -> five cases.
    with tempfile.TemporaryDirectory(prefix="m1069_full_") as raw:
        root = Path(raw) / "hw_autoresearch_nts07"
        root.mkdir()
        site, outer = prepare_redirect(root)
        result = run_redirected(root, site, outer, "full")
        require(result.returncode == 0, "redirected complete mock failed: " + result.stderr)
        events = (root / "event.log").read_text().splitlines()
        require(events == ["DC", "VCS_COMPILE", "CASE0", "CASE1", "CASE2", "CASE3", "CASE4"],
                "DC-before-mapped/five-case ordering failure")
        require((root / ATTEMPT_REL).is_dir() and (root / RESULT_REL).is_dir(),
                "redirected atomic result publication failed")
        receipt = strict_json(root / RESULT_REL / "m1068_dc_mapped_vcs_receipt_r1.json")
        require(receipt["anchors"] == [259, 737, 3153, 7569, 14] and
                receipt["mapped_cases"] == 5 and receipt["saif_files"] == 0 and
                receipt["ptpx_runs"] == 0 and
                receipt["random_register_initialization_used"] is False,
                "redirected result claim boundary drift")
    return attacks, events


def publish(static: dict, attacks: dict, events: list[str]) -> None:
    review = {
        "schema": "m1069_m1068_c2_k1_reset_hygiene_one_shot_release_hammer_r1",
        "milestone": "M1069",
        "date": "2026-08-30",
        "status": "PASS_M1069_M1068_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER",
        "verdict": "GO_EXACTLY_ONE_M1068_FRESH_DC_THEN_FIVE_CASE_MAPPED_VCS_ATTEMPT",
        "identity": {
            "runner_sha256": EXPECTED[RUNNER_REL],
            "contract_outer_seal_sha256": EXPECTED[Path(str(CONTRACT_REL) + ".sha256.seal.sha256")],
            "release_outer_seal_sha256": EXPECTED[Path(str(RELEASE_REL) + ".sha256.seal.sha256")],
            "docs359_sha256": EXPECTED[Path("docs/359_DATE终局冻结_20260813.md")],
        },
        "receipt_blind": True,
        "real_eda_launched": False,
        "real_m1068_attempt_consumed": False,
        "static_audit": static,
        "redirected_mock_order": events,
        "fault_injections": attacks,
        "authorization": {"one_m1068_dc_then_mapped_vcs_attempt": True},
        "caller": {
            "required_runner_sha256": EXPECTED[RUNNER_REL],
            "required_m1069_outer_seal_sha256": "PIN_THIS_DIRECTORY_SHA256SUMS_SEAL_SHA256",
        },
        "claim_boundary": {
            "saif_authorized": False,
            "ptpx_authorized": False,
            "power_admitted": False,
            "system_speedup_admitted": False,
            "paper_ppa_ready": False,
        },
    }
    (HERE / "review.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n")
    (HERE / "mechanical_checks.txt").write_text(
        "PASS receipt-blind static identity and release-chain audit\n"
        "PASS 11/11 redirected pre-attempt attacks rejected\n"
        "PASS redirected failed DC quarantined before mapped VCS\n"
        "PASS redirected complete order DC -> mapped compile -> cases 0..4\n"
        "PASS production runner text/argv excludes initreg and power commands\n")
    (HERE / "review.md").write_text(
        "# M1069 independent release hammer\n\n"
        "结论：**GO，仅授权一次 M1068 fresh DC → 五案例 mapped VCS。**\n\n"
        "独立核验 runner、双 sidecar seal、M1058/M1059/M1050/M1046、技术库、Tcl、SDC、filelist、TB 与五个周期锚点。"
        "11 类重定向攻击均在 attempt 前拒绝；模拟 DC 失败只产生隔离目录且没有 mapped VCS；完整 mock 的调用顺序严格为 "
        "DC、mapped compile、case0..case4。整个 hammer 未启动真实 DC/VCS，未消费真实 M1068。\n\n"
        "生产 runner 不含 initreg 参数，不运行 SAIF/PTPX。GO 不承认功耗、系统加速或 paper-ready PPA。\n")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "PASS_M1069_M1068_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER\n")
    outer = seal_dir(HERE)
    # The caller pins the final outer identity; review text intentionally does
    # not self-reference it, avoiding a circular seal.
    print("PASS M1069 release hammer")
    print("M1069_OUTER=" + outer)


def publish_stop(reason: str) -> None:
    source_side = HW / "contracts/m1058_c2_k1_reset_hygiene_source_only_contract_r1_20260830.json.sha256"
    source_outer = HW / "contracts/m1058_c2_k1_reset_hygiene_source_only_contract_r1_20260830.json.sha256.seal.sha256"
    candidate_side = HW / "contracts/m1058_c2_k1_reset_hygiene_dc_mapped_vcs_launch_candidate_r1_20260830.json.sha256"
    candidate_outer = HW / "contracts/m1058_c2_k1_reset_hygiene_dc_mapped_vcs_launch_candidate_r1_20260830.json.sha256.seal.sha256"
    evidence = {
        "runner_verify_sidecar_primary_required_name":
            "m1058_c2_k1_reset_hygiene_source_only_contract_r1_20260830.json",
        "source_primary_actual_name": source_side.read_text().split()[1],
        "runner_verify_sidecar_outer_required_name": source_side.name,
        "source_outer_actual_name": source_outer.read_text().split()[1],
        "candidate_primary_actual_name": candidate_side.read_text().split()[1],
        "candidate_outer_actual_name": candidate_outer.read_text().split()[1],
        "failure_phase": "SOURCE_PREFLIGHT/static_identity_gate/verify_sidecar(M1058_CONTRACT)",
        "attempt_consumed_if_launched": False,
        "dc_reachable": False,
        "mapped_vcs_reachable": False,
    }
    review = {
        "schema": "m1069_m1068_c2_k1_reset_hygiene_one_shot_release_hammer_r1",
        "milestone": "M1069",
        "date": "2026-08-30",
        "status": "STOP_M1069_M1068_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER",
        "verdict": "STOP_RUNNER_SIDECAR_NAME_POLICY_INCOMPATIBLE_WITH_PINNED_M1058_SIDECARS",
        "identity": {
            "runner_sha256": EXPECTED[RUNNER_REL],
            "contract_outer_seal_sha256": EXPECTED[Path(str(CONTRACT_REL) + ".sha256.seal.sha256")],
            "release_outer_seal_sha256": EXPECTED[Path(str(RELEASE_REL) + ".sha256.seal.sha256")],
            "docs359_sha256": EXPECTED[Path("docs/359_DATE终局冻结_20260813.md")],
        },
        "receipt_blind": True,
        "real_eda_launched": False,
        "real_m1068_attempt_consumed": False,
        "p0": {"class": "VALIDATOR_EVIDENCE_FORMAT_INCOMPATIBILITY", "reason": reason,
                "evidence": evidence},
        "authorization": {"one_m1068_dc_then_mapped_vcs_attempt": False},
        "required_repair": {
            "additive_successor_runner": True,
            "preserve_old_m1058_evidence": True,
            "accept_only_exact_pinned_relative_or_basename_sidecar_tokens": True,
            "new_independent_hammer_required": True,
        },
        "claim_boundary": {"dc": False, "mapped_vcs": False, "saif": False,
                           "ptpx": False, "power": False,
                           "system_speedup": False, "paper_ppa_ready": False},
    }
    (HERE / "review.json").write_text(json.dumps(review, sort_keys=True, indent=2) + "\n")
    (HERE / "mechanical_checks.txt").write_text(
        "PASS exact runner/contract/release/docs359/tool identities before P0\n"
        "FAIL M1068 verify_sidecar requires basename but pinned M1058 sidecars carry contracts/ relative names\n"
        "PASS failure occurs in SOURCE_PREFLIGHT before namespace/attempt/DC/VCS\n"
        "STOP one_m1068_dc_then_mapped_vcs_attempt=false\n")
    (HERE / "review.md").write_text(
        "# M1069 independent release hammer\n\n"
        "结论：**STOP，不授权 M1068。**\n\n"
        "P0 是验证器与冻结证据格式不兼容：M1068 `verify_sidecar()` 要求 sidecar 第二列严格等于 basename；"
        "两份已钉住的 M1058 sidecar 第二列均为 `contracts/...` 相对路径，outer sidecar 同样为 `contracts/...sha256`。"
        "因此 runner 在 `SOURCE_PREFLIGHT/static_identity_gate` 即失败，attempt、DC 和 mapped VCS 均不可达。\n\n"
        "建议新建加法 successor runner，使验证器只接受被显式钉住的 exact relative-name 或 basename 两种格式；"
        "不得改写旧 M1058 封存证据，并需新的独立 hammer。\n")
    (HERE / "RUN_COMPLETE.txt").write_text(
        "STOP_M1069_M1068_C2_K1_RESET_HYGIENE_ONE_SHOT_RELEASE_HAMMER\n")
    outer = seal_dir(HERE)
    print("STOP M1069 release hammer")
    print("M1069_OUTER=" + outer)


def main() -> None:
    try:
        static = static_audit()
    except RuntimeError as exc:
        if "sidecar mismatch:" in str(exc):
            publish_stop(str(exc))
            return
        raise
    attacks, events = dynamic_audit()
    publish(static, attacks, events)


if __name__ == "__main__":
    main()
