#!/usr/bin/env python3
"""Independent no-EDA hammer for M2142's claimed ordinary-only preflight."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile


sys.dont_write_bytecode = True
HERE = Path(__file__).resolve()
REVIEW = HERE.parent
HW = REVIEW.parents[1]
REPO = HW.parent
RUNNER = HW / "dc_handoff/scripts/run_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_one_shot.py"
PARSER = HW / "system_simulator/scripts/parse_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.py"
TB = HW / "tb_m2018/tb_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.sv"
PARENT_TB = HW / "tb_m2018/tb_m2051_ep34_tsbg_full40_cycle.sv"
UCLI = HW / "dc_handoff/scripts/m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.ucli.tcl"
FILELIST = HW / "dc_handoff/filelists/tcasii_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_vcs.f"
TEST = HW / "tests/test_m2142_tsbg_ordinary_late_enable_saif_preflight.py"
CONTRACT = HW / "contracts/m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_source_contract_r1_20260904.json"
SELFCHECK = HW / "reviews/m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_source_selfcheck_r1_20260904"
M2140 = HW / "reviews/m2140_m2139_m2137_m2018_tsbg_rtl_saif_window_diagnostic_failure_hammer_r1_20260904"
M2139_ATTEMPT = HW / "results/.m2139_m2137_tsbg_rtl_saif_window_diagnostic_attempt_consumed"
M2144_RESULT = HW / "results/m2144_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight_r1_20260904"
M2144_ATTEMPT = HW / "results/.m2144_m2142_tsbg_ordinary_late_enable_saif_preflight_attempt_consumed"
M2144_LOCK = HW / "results/.m2144_m2142_tsbg_ordinary_late_enable_saif_preflight_launch_lock"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

DOC359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M2140_REVIEW_SHA = "5612255e46cb6d8017c84049aa1ebb2202f04cb1fe5ca181a3d974425bfb6ff8"
M2140_MANIFEST_SHA = "8f315963e4aede2ef2135cb2c766841b87db090dac4c381eb2a8677865ec99d2"
M2140_OUTER_SHA = "f690041cfb31564ea8d714480aa44ed8d496812dd60507f87e83ac030be1762e"
EXPECTED_SOURCE_SHA = {
    RUNNER: "2645533be2c3bbed72f5fe02e1d1c3e1075617b7c7bcdfb68eb6bc2527fae713",
    PARSER: "2b9709aba8f8245fb1e9743a06bce021ed7582ad2ac5cb81fe0e71881d7ae95e",
    TB: "913467097222eabb061d475004a9ca34125914985685b757022d1d01509d35b8",
    FILELIST: "e35851f5487ef98a500e773731ef92bfe9cac4da38e1a197074b123df2f4638d",
    UCLI: "f85194cf4fde872b9732f5c0b5e85d18811baef6b22b3a75d503c6ec9366c71b",
    TEST: "34309d430529316ee82f4865cb1a75b144f0e68ca40aa8c83898d1e1b4ad20f3",
}
EXPECTED_BUDGET = {
    "license_queries": 1, "vcs_compiles": 1, "simv_runs": 1,
    "raw_saif_files_written": 1, "admitted_saif_files": 1,
    "dc_runs": 0, "ptpx_runs": 0, "icc2_runs": 0, "gpu_runs": 0,
    "automatic_retry": False, "ordinary_only": True,
    "reuse_old_artifacts": False,
}


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def verify_seal(root: Path, manifest_sha: str | None = None,
                outer_sha: str | None = None) -> int:
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(manifest.is_file() and not manifest.is_symlink(), f"manifest: {root}")
    need(outer.is_file() and not outer.is_symlink(), f"outer: {root}")
    if manifest_sha is not None:
        need(sha(manifest) == manifest_sha, f"manifest SHA: {root}")
    if outer_sha is not None:
        need(sha(outer) == outer_sha, f"outer SHA: {root}")
    need(outer.read_text().split() == [sha(manifest), "SHA256SUMS"],
         f"outer content: {root}")
    listed: dict[str, str] = {}
    for line in manifest.read_text().splitlines():
        digest, name = line.split("  ", 1)
        need("/" not in name and name not in listed, f"member name: {name}")
        listed[name] = digest
    actual = sorted(path.name for path in root.iterdir()
                    if path.is_file() and path.name not in
                    {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    need(sorted(listed) == actual, f"non-exhaustive seal: {root}")
    for name, digest in listed.items():
        path = root / name
        need(not path.is_symlink() and sha(path) == digest, f"member SHA: {path}")
    return len(actual)


def load_parser():
    spec = importlib.util.spec_from_file_location("m2143_exact_parser", PARSER)
    need(spec is not None and spec.loader is not None, "parser import spec")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def must_fail(callable_, label: str) -> None:
    try:
        callable_()
    except Exception as exc:
        need(type(exc).__name__ == "Failure", f"wrong rejection for {label}: {exc}")
        return
    raise RuntimeError(f"mutation admitted: {label}")


def runtime_fixture() -> str:
    return "\n".join([
        "M2142_UCLI_PHASE order=1 action=power_enable timing=before_first_run scope=ordinary_implementation",
        "M2142_UCLI_PHASE order=2 action=run_reset_and_preload observer_enabled=1",
        "M2142_INTERNAL_KNOWNNESS_CENSUS phase=pre_power_reset row_live=192/192 row_live_one=149 cache_valid=4/4 cache_valid_one=0 slot_valid=8/8 slot_valid_one=0 bridge_overflow=16/16 bridge_overflow_one=0 rsp_shape_legal=8/8 rsp_shape_legal_one=8 total=228/228 observe_only=1 force=0 deposit=0 mask=0 rtl_edit=0",
        "M2142_RTL_SAIF_WINDOW_BEGIN sampling=settled_negedge global_slot=42 sample=0 layer=28 is_fc2=0 token_start=0 source_groups=48 preload_cycles=383 time_ns=1153.51 next_ucli_action=power_reset",
        "M2142_UCLI_PHASE order=3 action=first_stop_reached internal_census_preceded_stop=1",
        "M2142_UCLI_PHASE order=4 action=power_reset timing=after_first_stop_before_measurement_run",
        "M2142_RTL_SAIF_WINDOW_END axis=ordinary_lru4 sampling=settled_negedge measurement_cycles=20292 scalar_weight_reads=14304 duration_ns=60876.00",
        "PASS_M2142_ORDINARY_LATE_ENABLE_SAIF_PREFLIGHT ledger_exact=1 internal_census_exact=1 enable_before_reset_preload=1 power_reset_at_first_stop=1 initreg_diagnostic_only=1 paper_citable=0",
        "M2142_UCLI_PHASE order=5 action=second_stop_reached exact_window_complete=1",
        "M2142_UCLI_PHASE order=6 action=power_disable timing=before_report",
        "M2142_UCLI_PHASE order=7 action=power_report scope=ordinary_implementation",
        "",
    ])


def saif_fixture(parser, *, records: int = 93971, tx_first: int = 0,
                 duration: int = 60876, break_conservation: bool = False,
                 drop_critical: bool = False) -> str:
    names = list(parser.CRITICAL)
    rows: list[str] = []
    for index in range(records):
        name = names[index] if index < len(names) else f"filler_{index}"
        if drop_critical and index == 0:
            name = "missing_critical"
        tx = tx_first if index == 0 else 0
        t0 = duration - 1 - tx
        if break_conservation and index == 0:
            t0 -= 1
        rows.append(f"({name} (T0 {t0}) (T1 1) (TX {tx}) (TC 2))")
    return "\n".join(["(SAIFILE", "(TIMESCALE 1 ns)",
                       f"(DURATION {duration})", *rows, ")", ""])


def main() -> int:
    # Raw identities, contract seals, and exhaustive predecessor/selfcheck seals.
    for path, digest in EXPECTED_SOURCE_SHA.items():
        need(path.is_file() and not path.is_symlink() and sha(path) == digest,
             f"source identity: {path}")
    contract_sha = sha(CONTRACT)
    sidecar = Path(str(CONTRACT) + ".sha256")
    outer = Path(str(CONTRACT) + ".sha256.seal.sha256")
    need(sidecar.read_text().split() == [contract_sha, CONTRACT.name],
         "contract sidecar")
    need(outer.read_text().split() == [sha(sidecar), sidecar.name],
         "contract outer seal")
    selfcheck_members = verify_seal(SELFCHECK)
    m2140_members = verify_seal(M2140, M2140_MANIFEST_SHA, M2140_OUTER_SHA)
    need(sha(M2140 / "review.json") == M2140_REVIEW_SHA, "M2140 review SHA")
    need(sha(DOC359) == DOC359_SHA, "docs359 identity")

    contract = load_json(CONTRACT)
    need(contract["execution_budget"] == EXPECTED_BUDGET, "contract budget")
    need(contract["claim_boundary"]["tsbg_axis_run"] is False,
         "contract TSBG boundary")
    need(contract["claim_boundary"]["paper_citable"] is False,
         "contract paper boundary")
    need(contract["source_inventory"] == {
        str(path.relative_to(REPO)): digest
        for path, digest in EXPECTED_SOURCE_SHA.items()}, "source inventory")
    m2140 = load_json(M2140 / "review.json")
    need(m2140["m2139_disposition"]["attempt_consumed"] is True
         and m2140["m2139_disposition"]["retry_authorized"] is False,
         "M2139 disposition")
    need(M2139_ATTEMPT.is_dir(), "M2139 consumed marker")
    need(not M2144_RESULT.exists() and not M2144_ATTEMPT.exists()
         and not M2144_LOCK.exists(), "M2144 namespace not fresh")

    # Fixed launch surface and one-shot topology.  No tool is invoked here.
    runner_text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(runner_text)
    production = next(node for node in tree.body
                      if isinstance(node, ast.FunctionDef) and node.name == "production")
    production_text = ast.unparse(production)
    need(production_text.count("counts['license_queries'] += 1") == 1,
         "single license count")
    need(production_text.count("counts['vcs_compiles'] += 1") == 1,
         "single compile count")
    need(production_text.count("counts['simv_runs'] += 1") == 1,
         "single sim count")
    need("for axis" not in production_text and "while " not in production_text,
         "no axis/retry loop")
    need("+vcs+initreg+random" in production_text
         and "+vcs+initreg+0" in production_text, "initreg surface")
    need("run_dc" not in production_text and "pt_shell" not in production_text
         and "icc2_shell" not in production_text and "nvidia" not in production_text,
         "forbidden execution path")
    need(runner_text.index("source_validation(require_review=True)")
         < runner_text.index("ATTEMPT.mkdir()")
         < runner_text.index('counts["license_queries"] += 1'),
         "review/attempt/license order")

    ucli = UCLI.read_text(encoding="utf-8")
    need(ucli.index("power -enable") < ucli.index("\nrun\n")
         < ucli.index("power -reset") < ucli.rindex("\nrun\n"),
         "UCLI causal order")
    ucli_commands = [line.strip() for line in ucli.splitlines()
                     if line.strip() and not line.lstrip().startswith("#")
                     and not line.lstrip().startswith("puts ")]
    need(ucli_commands.count("power -enable") == 1
         and ucli_commands.count("power -reset") == 1
         and sum(line.startswith("power -report ")
                 for line in ucli_commands) == 1, "UCLI action surface")
    need("core.dut_base.implementation" in ucli
         and "dut_tsbg" not in ucli, "DUT-only report scope")
    tb_text = TB.read_text(encoding="utf-8")
    active_tb_text = "\n".join(line for line in tb_text.splitlines()
                                if not line.lstrip().startswith("//"))
    need("force " not in active_tb_text and "deposit " not in active_tb_text,
         "observation-only TB")
    need("$isunknown" in active_tb_text
         and "FROZEN_INTERNAL_ELEMENTS = 228" in active_tb_text,
         "228-element census")

    # Independent parser mutations.
    parser = load_parser()
    mutation_counts = {"runtime": 0, "saif": 0, "ucli": 0}
    with tempfile.TemporaryDirectory(prefix="m2143_parser_hammer.") as raw:
        root = Path(raw)
        runtime = root / "rtl_sim.log"
        saif = root / "rtl_execute.saif"
        valid_runtime = runtime_fixture()
        runtime.write_text(valid_runtime, encoding="utf-8")
        need(parser.parse_runtime(runtime)["measurement_cycles"] == 20292,
             "runtime positive")
        runtime_mutations = [
            valid_runtime.replace("order=4 action=power_reset",
                                  "order=4 action=power_reset_BAD"),
            valid_runtime.replace("row_live=192/192", "row_live=191/192"),
            valid_runtime.replace("measurement_cycles=20292",
                                  "measurement_cycles=20291"),
            valid_runtime.replace("scalar_weight_reads=14304",
                                  "scalar_weight_reads=14303"),
            valid_runtime.replace("duration_ns=60876.00", "duration_ns=60875.00"),
            valid_runtime.replace("M2142_UCLI_PHASE order=7",
                                  "M2142_UCLI_PHASE order=6"),
        ]
        for index, text in enumerate(runtime_mutations):
            runtime.write_text(text, encoding="utf-8")
            must_fail(lambda: parser.parse_runtime(runtime), f"runtime_{index}")
            mutation_counts["runtime"] += 1

        valid_saif = saif_fixture(parser)
        saif.write_text(valid_saif, encoding="utf-8")
        need(parser.parse_saif(saif)["record_count"] == 93971, "SAIF positive")
        saif_mutations = [
            saif_fixture(parser, tx_first=1),
            saif_fixture(parser, records=93970),
            saif_fixture(parser, duration=60877),
            saif_fixture(parser, break_conservation=True),
            saif_fixture(parser, drop_critical=True),
        ]
        for index, text in enumerate(saif_mutations):
            saif.write_text(text, encoding="utf-8")
            must_fail(lambda: parser.parse_saif(saif), f"saif_{index}")
            mutation_counts["saif"] += 1

        ucli_mutations = [
            ucli.replace("power -enable\n", "", 1) + "\npower -enable\n",
            ucli.replace("power -reset\n", "", 1) + "\npower -reset\n",
            ucli.replace("core.dut_base.implementation",
                         "core.dut_tsbg.implementation", 1),
        ]
        for index, text in enumerate(ucli_mutations):
            causal = (text.count("power -enable") == 1
                      and text.index("power -enable") < text.index("\nrun\n")
                      and text.count("power -reset") == 1
                      and text.index("power -reset") < text.rindex("\nrun\n")
                      and "power tb_m2142_m2018_tsbg_ordinary_late_enable_saif_preflight.core.dut_base.implementation" in text)
            need(not causal, f"UCLI mutation admitted: {index}")
            mutation_counts["ucli"] += 1

    # P0: the elaborated parent is a dual-axis harness.  The wrapper plusarg
    # cannot disable this unconditional TSBG instance/load/execution path.
    parent = PARENT_TB.read_text(encoding="utf-8")
    embedded_tsbg = {
        "wrapper_instantiates_dual_axis_parent":
            "tb_m2051_ep34_tsbg_full40_cycle core();" in tb_text,
        "parent_instantiates_schedule_mode_1":
            "`CONNECT_M1880(dut_tsbg, tsbg, 1, load_valid_tsbg);" in parent,
        "parent_asserts_tsbg_load_valid": "load_valid_tsbg = 1;" in parent,
        "parent_waits_for_tsbg_completion":
            "wait (base_done_cycle >= 0 && tsbg_done_cycle >= 0);" in parent,
        "ordinary_plusarg_not_consumed_by_parent":
            "M2142_AXIS_ORDINARY" not in parent,
    }
    need(all(embedded_tsbg.values()), f"dual-axis evidence drift: {embedded_tsbg}")

    output = {
        "schema": "m2143_m2142_independent_mechanical_checks_r1_v1",
        "status": "FAIL_M2143_INDEPENDENT_MECHANICAL_CHECKS__EMBEDDED_TSBG_EXECUTION",
        "execution_performed": {
            "license_queries": 0, "vcs_compiles": 0, "simv_runs": 0,
            "saif_files": 0, "dc_runs": 0, "ptpx_runs": 0,
            "icc2_runs": 0, "gpu_runs": 0,
        },
        "identity": {
            "runner_sha256": sha(RUNNER), "parser_sha256": sha(PARSER),
            "tb_sha256": sha(TB), "filelist_sha256": sha(FILELIST),
            "ucli_sha256": sha(UCLI), "test_sha256": sha(TEST),
            "contract_sha256": sha(CONTRACT), "docs359_sha256": sha(DOC359),
        },
        "seal_member_counts": {
            "m2142_selfcheck": selfcheck_members, "m2140": m2140_members},
        "source_inventory_count": len(EXPECTED_SOURCE_SHA),
        "m2139_consumed_no_retry": True,
        "m2144_namespace_fresh_but_not_authorized": True,
        "fixed_launch_surface": {
            "license_queries": 1, "vcs_compiles": 1, "simv_runs": 1,
            "raw_saif_max": 1, "compile_initreg": "+vcs+initreg+random",
            "runtime_initreg": "+vcs+initreg+0", "automatic_retry": False,
            "dc_ptpx_icc2_gpu_paths": 0,
        },
        "ucli_and_parser_positive_gates": {
            "power_enable_before_first_run": True,
            "census_228_before_first_stop": True,
            "power_reset_before_measurement_run": True,
            "measurement_cycles": 20292, "measurement_duration_ns": 60876.0,
            "saif_record_count": 93971, "every_tx_zero": True,
            "dut_base_only_report_scope": True,
        },
        "independent_mutations_rejected": mutation_counts,
        "p0_embedded_tsbg_axis": embedded_tsbg,
        "contract_claim_contradicted": {
            "execution_budget_ordinary_only": True,
            "claim_tsbg_axis_run": False,
            "actual_elaborated_tsbg_datapath_runs": True,
        },
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
