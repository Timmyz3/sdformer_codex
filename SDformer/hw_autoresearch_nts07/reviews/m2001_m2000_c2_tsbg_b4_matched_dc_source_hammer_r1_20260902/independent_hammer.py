#!/usr/bin/env python3
"""Read-only static and mutation hammer for the exact M2000 matched-DC runner."""

import hashlib
import json
import re
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
RUNNER = HW / "dc_handoff/scripts/run_m2000_m1999_c2_tsbg_b4_matched_two_axis_logic_only_dc_one_shot.sh"
FILELIST = HW / "dc_handoff/filelists/iscas_m2000_c2_tsbg_b4_matched_two_axis_logic_only_dc.f"
RTL = HW / "rtl_m1995/m1995_m1880_c2_tsbg_b4_dc_keyword_legal_frontend.sv"
ADAPTER = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
TCL = HW / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = HW / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1999_DIR = HW / "reviews/m1999_m1998_m1995_c2_tsbg_keyword_legal_vcs_result_hammer_r1_20260902"
M1866_DIR = HW / "reviews/m1866_tsbg_ep34_same_io_b2_b4_b8_quickkill_independent_hammer_r1_20260902"
M1995_DIR = HW / "reviews/m1995_m1992_tsbg_dc_keyword_failure_hammer_r1_20260902"
M1997_DIR = HW / "reviews/m1997_m1996_m1995_c2_tsbg_keyword_legal_vcs_source_hammer_r1_20260902"


EXPECTED = {
    "runner": "3f9b4fc3d8fc2e309394cdedb23d88b3f921d36604844ff5ead5ec580946671a",
    "filelist": "6d2ce30b81ad00da3159f5b4c1f297e3615388db46823c8bead3eb692590085f",
    "rtl": "2c1a8a7644b359a153decdc3106a8718992d37d54809007b61e184121fcc14fd",
    "adapter": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    "tcl": "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    "sdc": "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "m1999": "ce6c68463fc258b62920fc603344cb67b367e81e41d971fa4d6779de47c2af16",
    "m1866": "6560b3660d247440691d31dea7cccd0ca0294cd203c7f2d957a183116eb81830",
    "m1995_failure": "37adc83f6b6f70457d06e8ba215dba64d345fd03e2c2b8f3ea5ed363f11a5c01",
    "m1997_source": "b2545bd3b3c0d819e6c8bf8a506286f5f725dde204eb251e6b84b3e6307909f5",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_seal(directory: Path) -> None:
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    if not manifest.is_file() or manifest.is_symlink() or not outer.is_file() or outer.is_symlink():
        raise AssertionError(f"missing/non-regular seal in {directory}")
    for line in manifest.read_text().splitlines():
        digest, rel = line.split(None, 1)
        rel = rel.lstrip(" *")
        target = directory / rel
        if not target.is_file() or target.is_symlink() or sha(target) != digest:
            raise AssertionError(f"inner seal mismatch: {target}")
    digest, rel = outer.read_text().strip().split(None, 1)
    if rel.lstrip(" *") != "SHA256SUMS" or sha(manifest) != digest:
        raise AssertionError(f"outer seal mismatch: {directory}")


def require_count(text: str, token: str, count: int = 1) -> None:
    actual = text.count(token)
    if actual != count:
        raise AssertionError(f"expected {token!r} exactly {count} time(s), found {actual}")


def static_audit(runner: str, filelist: str, rtl: str) -> None:
    expected_filelist = (
        "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv\n"
        "rtl_m1995/m1995_m1880_c2_tsbg_b4_dc_keyword_legal_frontend.sv\n"
    )
    if filelist != expected_filelist:
        raise AssertionError("filelist contents/order differ")

    require_count(runner, "DESIGN=m1880_c2_tsbg_b4_real_channel_signed_frontend")
    require_count(runner, "axis_names=(ordinary_lru4 tsbg_b4)")
    require_count(runner, "axis_modes=(0 1)")
    require_count(runner, "for index in 0 1; do")
    require_count(runner, 'ELAB_PARAMETERS="SCHEDULE_MODE=${mode}"')
    if "SOURCE_GROUPS=" in re.search(r'ELAB_PARAMETERS=.*', runner).group(0):
        raise AssertionError("production G48 overridden")
    require_count(runner, "/usr/bin/timeout --signal=TERM --kill-after=60s 21600s")
    require_count(runner, 'verify_dir_seal "${M1999_DIR}"')
    require_count(runner, 'verify_dir_seal "${M1866_DIR}"')
    require_count(runner, 'verify_dir_seal "${SOURCE_REVIEW_DIR}"')
    require_count(runner, "'dc_shell_runs': 2, 'all_other_eda_runs': 0")
    require_count(runner, "assert r['authorization'] == {'dc_shell_runs': 2, 'all_other_eda_runs': 0}")
    require_count(runner, "assert v['status'].startswith('PASS_M1999')")
    require_count(runner, "['status'].startswith('PASS_INDEPENDENT_REPLAY')")
    require_count(runner, 'mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"')
    if runner.index('mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"') > runner.index('"${LMUTIL}" lmstat'):
        raise AssertionError("attempt is not consumed before license preflight")
    require_count(runner, 'mv -T -- "${WORK}" "${RESULT}"')
    require_count(runner, 'mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine"')
    if runner.index('seal_dir "${WORK}"\nmv -T -- "${WORK}" "${RESULT}"') < 0:
        raise AssertionError("result is not sealed before atomic publish")
    require_count(runner, "error_line='Error: Error during sourcing of /opt/synopsys/syn/V-2023.12-SP3/auxx/gui/dv/.synopsys_dv.tcl'")
    require_count(runner, "3f0791c8c38447275968806360703faa95ef6a45ae53bd3502d09a6c535049e1")
    require_count(runner, "grep -Fxq 'TIM-209=0'")
    require_count(runner, "grep -Fxq 'OPT-150=0'")
    require_count(runner, "reports/constraint_max_capacitance.rpt", 2)
    require_count(runner, "reports/constraint_max_transition.rpt", 2)
    require_count(runner, "reports/constraint_max_fanout.rpt", 2)
    require_count(runner, "return min(float(value) for value in matches)")
    require_count(runner, "area_ratio <= 1.10 and port_equal and setup_met")
    require_count(runner, "'production_g48_dynamically_verified': False")
    require_count(runner, "'exact_cycle_ratio': False")
    require_count(runner, "'system_speedup': False")
    require_count(runner, "'paper_ppa_ready': False")
    require_count(runner, "'same_area': False")
    require_count(runner, "'hold_closed': False")
    require_count(runner, "'power': False")
    require_count(runner, "'energy': False")
    require_count(runner, "'cross_layer_flush_or_rebind_implemented': False")
    require_count(runner, "'state_arrays_synthesized_as_standard_cells': True")
    require_count(runner, "'both_axes_own_bundle4_candidate_state': True")
    require_count(runner, "'physical_schedule_ablation_not_conventional_baseline_ppa': True")
    require_count(runner, "'full_conventional_baseline_area_priced': False")
    require_count(runner, "'cpu_premodel_speedup_not_upgraded_to_rtl': True")
    require_count(runner, "'directed_weight_bundle_reduction_fraction_from_m1999': 0.75")
    require_count(runner, "'directed_scalar_bank_request_reduction_fraction_from_m1999': 0.75")

    require_count(rtl, "module m1880_c2_tsbg_b4_real_channel_signed_frontend #(")
    require_count(rtl, "parameter int SCHEDULE_MODE = 1")
    require_count(rtl, "parameter int BUNDLE = 4")
    require_count(rtl, "parameter int SOURCE_GROUPS = 48")
    require_count(rtl, "parameter int CACHE_ROWS = 4")
    require_count(rtl, "if (SCHEDULE_MODE == 0)")


files = {
    "runner": RUNNER,
    "filelist": FILELIST,
    "rtl": RTL,
    "adapter": ADAPTER,
    "tcl": TCL,
    "sdc": SDC,
    "docs359": DOCS359,
    "m1999": M1999_DIR / "review.json",
    "m1866": M1866_DIR / "review.json",
    "m1995_failure": M1995_DIR / "review.json",
    "m1997_source": M1997_DIR / "review.json",
}
for name, path in files.items():
    if sha(path) != EXPECTED[name]:
        raise AssertionError(f"identity mismatch: {name}")
for directory in (M1999_DIR, M1866_DIR, M1995_DIR, M1997_DIR):
    verify_seal(directory)

m1999 = json.loads(files["m1999"].read_text())
m1866 = json.loads(files["m1866"].read_text())
m1995 = json.loads(files["m1995_failure"].read_text())
m1997 = json.loads(files["m1997_source"].read_text())
assert m1999["status"].startswith("PASS_M1999")
assert m1999["severity_counts"] == {"p0": 0, "p1": 0, "p2": 0}
assert m1999["identity"]["m1995_rtl_sha256"] == EXPECTED["rtl"]
assert m1999["identity"]["m1995_failure_review_sha256"] == EXPECTED["m1995_failure"]
assert m1999["identity"]["m1997_source_review_sha256"] == EXPECTED["m1997_source"]
assert m1866["status"].startswith("PASS_INDEPENDENT_REPLAY")
assert m1995["status"].startswith("PASS_M1995")
assert m1995["failure_diagnosis"]["observed_unique_first_cause_class"].endswith("keyword context")
assert m1995["successor_decision"]["old_m1990_vcs_may_be_rebound_to_new_source"] is False
assert m1997["status"].startswith("PASS_M1997")

runner_text = RUNNER.read_text()
filelist_text = FILELIST.read_text()
rtl_text = RTL.read_text()
static_audit(runner_text, filelist_text, rtl_text)

mutations = [
    ("duplicate_schedule_modes", runner_text.replace("axis_modes=(0 1)", "axis_modes=(0 0)"), filelist_text, rtl_text),
    ("reverse_axis_names", runner_text.replace("axis_names=(ordinary_lru4 tsbg_b4)", "axis_names=(tsbg_b4 ordinary_lru4)"), filelist_text, rtl_text),
    ("single_axis_loop", runner_text.replace("for index in 0 1; do", "for index in 0; do"), filelist_text, rtl_text),
    ("g12_override", runner_text.replace('ELAB_PARAMETERS="SCHEDULE_MODE=${mode}"', 'ELAB_PARAMETERS="SCHEDULE_MODE=${mode},SOURCE_GROUPS=12"'), filelist_text, rtl_text),
    ("wrong_top", runner_text.replace("DESIGN=m1880_c2_tsbg_b4_real_channel_signed_frontend", "DESIGN=wrong_top"), filelist_text, rtl_text),
    ("third_file", runner_text, filelist_text + "rtl/extra.sv\n", rtl_text),
    ("remove_review_seal", runner_text.replace('verify_dir_seal "${SOURCE_REVIEW_DIR}"', ": # removed"), filelist_text, rtl_text),
    ("remove_m1999_seal", runner_text.replace('verify_dir_seal "${M1999_DIR}"', ": # removed"), filelist_text, rtl_text),
    ("remove_m1866_seal", runner_text.replace('verify_dir_seal "${M1866_DIR}"', ": # removed"), filelist_text, rtl_text),
    ("loosen_authorization", runner_text.replace("{'dc_shell_runs': 2, 'all_other_eda_runs': 0}", "{'dc_shell_runs': 3, 'all_other_eda_runs': 1}"), filelist_text, rtl_text),
    ("remove_timeout", runner_text.replace("/usr/bin/timeout --signal=TERM --kill-after=60s 21600s", ":"), filelist_text, rtl_text),
    ("publish_before_seal", runner_text.replace('seal_dir "${WORK}"\nmv -T -- "${WORK}" "${RESULT}"', 'mv -T -- "${WORK}" "${RESULT}"\nseal_dir "${RESULT}"'), filelist_text, rtl_text),
    ("remove_failure_quarantine", runner_text.replace('mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$.quarantine"', ": # removed"), filelist_text, rtl_text),
    ("loosen_bootstrap_sha", runner_text.replace("3f0791c8c38447275968806360703faa95ef6a45ae53bd3502d09a6c535049e1", "0" * 64), filelist_text, rtl_text),
    ("last_slack", runner_text.replace("return min(float(value) for value in matches)", "return float(matches[-1])"), filelist_text, rtl_text),
    ("remove_tim209_gate", runner_text.replace("grep -Fxq 'TIM-209=0'", "grep -Fq 'TIM-209'"), filelist_text, rtl_text),
    ("remove_opt150_gate", runner_text.replace("grep -Fxq 'OPT-150=0'", "grep -Fq 'OPT-150'"), filelist_text, rtl_text),
    ("loosen_candidate_gate", runner_text.replace("area_ratio <= 1.10 and port_equal and setup_met", "area_ratio <= 1.10"), filelist_text, rtl_text),
    ("claim_g48_dynamic", runner_text.replace("'production_g48_dynamically_verified': False", "'production_g48_dynamically_verified': True"), filelist_text, rtl_text),
    ("claim_exact_cycle", runner_text.replace("'exact_cycle_ratio': False", "'exact_cycle_ratio': True"), filelist_text, rtl_text),
    ("claim_system_speedup", runner_text.replace("'system_speedup': False", "'system_speedup': True"), filelist_text, rtl_text),
    ("claim_same_area", runner_text.replace("'same_area': False", "'same_area': True"), filelist_text, rtl_text),
    ("claim_hold_closed", runner_text.replace("'hold_closed': False", "'hold_closed': True"), filelist_text, rtl_text),
    ("claim_cross_layer_flush", runner_text.replace("'cross_layer_flush_or_rebind_implemented': False", "'cross_layer_flush_or_rebind_implemented': True"), filelist_text, rtl_text),
    ("hide_state_array_mapping", runner_text.replace("'state_arrays_synthesized_as_standard_cells': True", "'state_arrays_synthesized_as_standard_cells': False"), filelist_text, rtl_text),
    ("hide_matched_state_bias", runner_text.replace("'both_axes_own_bundle4_candidate_state': True", "'both_axes_own_bundle4_candidate_state': False"), filelist_text, rtl_text),
    ("upgrade_premodel", runner_text.replace("'cpu_premodel_speedup_not_upgraded_to_rtl': True", "'cpu_premodel_speedup_not_upgraded_to_rtl': False"), filelist_text, rtl_text),
    ("rtl_g12_default", runner_text, filelist_text, rtl_text.replace("parameter int SOURCE_GROUPS = 48", "parameter int SOURCE_GROUPS = 12")),
]
mutation_results = []
for name, rtext, ftext, stext in mutations:
    try:
        static_audit(rtext, ftext, stext)
    except Exception as exc:
        mutation_results.append({"name": name, "rejected": True, "reason": str(exc)})
    else:
        mutation_results.append({"name": name, "rejected": False, "reason": "mutation escaped"})

if not all(item["rejected"] for item in mutation_results):
    raise AssertionError("one or more mutations escaped")

print(json.dumps({
    "status": "PASS_M2001_INDEPENDENT_STATIC_AND_MUTATION_HAMMER",
    "static_checks_pass": True,
    "mutation_count": len(mutation_results),
    "mutations_rejected": sum(item["rejected"] for item in mutation_results),
    "mutations": mutation_results,
    "identity": EXPECTED,
    "eda_launched": False,
    "license_query_launched": False,
}, indent=2, sort_keys=True))
