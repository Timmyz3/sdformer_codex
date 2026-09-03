#!/usr/bin/env python3
"""Independent static/mutation hammer for the exact M1992 matched DC source.

This program intentionally launches no EDA process and performs no license
query.  It binds the reviewed runner/filelist/source identities, checks the
matched two-axis structure and claim boundary, and verifies that independent
mutations of every admission-critical field are rejected.
"""

from __future__ import print_function

import hashlib
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "dc_handoff/scripts/run_m1992_m1990_c2_tsbg_b4_matched_two_axis_logic_only_dc_one_shot.sh"
FILELIST = ROOT / "dc_handoff/filelists/iscas_m1992_c2_tsbg_b4_matched_two_axis_logic_only_dc.f"
RTL = ROOT / "rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv"
ADAPTER = ROOT / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
TCL = ROOT / "dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl"
SDC = ROOT / "dc_handoff/constraints/date_m97_m85_logic_only_3ns.sdc"
M1990 = ROOT / "reviews/m1990_m1986_c2_tsbg_b4_parseable_vcs_result_hammer_r1_20260902/review.json"
M1866 = ROOT / "reviews/m1866_tsbg_ep34_same_io_b2_b4_b8_quickkill_independent_hammer_r1_20260902/review.json"
DOCS359 = ROOT / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    "runner": "4d7db586133997c0722a941c42c4d84a37b03eda1e688c035f814565fec3ad5f",
    "filelist": "e50027edea9470bda92e5f34f590c1c13f236e6f46b836ef4b5028465fe94f4c",
    "rtl": "8524f6a7a6d09e1aaab55ee91515bd1fce9ea57fa2a478a9817f637685299a05",
    "adapter": "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    "tcl": "c9da61c9a483487b3d1157538481a6c940d7277534e2acef634c4b1a1ff7adbe",
    "sdc": "808307c496bd67843907b727acdfe18ea3b48565798f97cb55e689c70c1183f5",
    "m1990": "e2935ed23f2e2b24798ea6b6ab1f098fcd356e1969e31279793a063c9b07b80c",
    "m1866": "6560b3660d247440691d31dea7cccd0ca0294cd203c7f2d957a183116eb81830",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(condition, message):
    if not condition:
        raise AssertionError(message)


def exactly(text, token, count=1):
    need(text.count(token) == count,
         "expected %r exactly %d time(s), found %d" %
         (token, count, text.count(token)))


def static_accept(runner, filelist, rtl):
    """Strict independent predicate for the reviewed source snapshot."""
    need("axis_names=(ordinary_lru4 tsbg_b4)" in runner,
         "two axis names are not exact")
    need("axis_modes=(0 1)" in runner, "axis modes are not 0/1")
    need("for index in 0 1; do" in runner, "two-run loop missing")
    exactly(runner, 'ELAB_PARAMETERS="SCHEDULE_MODE=${mode}"')
    exactly(runner, 'DESIGN=m1880_c2_tsbg_b4_real_channel_signed_frontend')
    need("[[ \"$(wc -l <\"${FILELIST}\")\" -eq 2 ]]" in runner,
         "two-line filelist gate missing")
    need(filelist.splitlines() == [
        "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv",
        "rtl_m1880/m1880_c2_tsbg_b4_real_channel_signed_frontend.sv",
    ], "filelist contents/order differ")

    # Production geometry remains the RTL default; the DC runner overrides
    # schedule mode only.  No G12 directed override is allowed here.
    for token in [
            "parameter int SCHEDULE_MODE = 1",
            "parameter int BUNDLE = 4",
            "parameter int SOURCE_GROUPS = 48",
            "parameter int SOURCES_PER_GROUP = 16",
            "parameter int OUTPUT_SLICES = 6",
            "parameter int CACHE_ROWS = 4",
            "parameter int LANES = 16"]:
        need(token in rtl, "production default missing: " + token)
    exactly(rtl, "if (SCHEDULE_MODE == 0)")
    need(rtl.count("SCHEDULE_MODE") == 4,
         "schedule parameter has an unexpected additional use")

    # The exact reviewed inputs, upstream VCS admission, and immutable freeze
    # document are all pinned before the attempt is consumed.
    for digest in [EXPECTED["filelist"], EXPECTED["tcl"], EXPECTED["sdc"],
                   EXPECTED["docs359"], EXPECTED["rtl"],
                   EXPECTED["adapter"], EXPECTED["m1990"],
                   EXPECTED["m1866"]]:
        need("sha_exact " + digest in runner,
             "missing exact input pin " + digest)
    need("verify_dir_seal \"${M1990_DIR}\"" in runner,
         "upstream M1990 double-seal gate missing")
    need("verify_dir_seal \"${M1866_DIR}\"" in runner,
         "upstream M1866 double-seal gate missing")
    need("verify_dir_seal \"${SOURCE_REVIEW_DIR}\"" in runner,
         "M1993 double-seal gate missing")
    need("M1992_EXPECTED_RUNNER_SHA256" in runner and
         "M1992_EXPECTED_REVIEW_SHA256" in runner,
         "caller-side exact runner/review pins missing")
    need("assert r['authorization'] == {'dc_shell_runs': 2, 'all_other_eda_runs': 0}" in runner,
         "authorization gate is not exact")
    need("json.loads(m1866.read_text())['status'].startswith('PASS_INDEPENDENT_REPLAY')" in runner,
         "M1866 independent premodel status gate missing")
    need("'m1866_cpu_premodel_review_sha256': sha(m1866)" in runner,
         "M1866 review identity is not written into the raw DC receipt")

    # One consumed attempt, quarantine on any incomplete run, and atomic
    # seal-before-publication are mandatory.
    need(runner.index('mkdir -- "${LOCK}" "${ATTEMPT}" "${WORK}"') <
         runner.index('"${LMUTIL}" lmstat'),
         "attempt is not consumed before license preflight")
    need("FAILED_OR_INCOMPLETE_DO_NOT_CITE" in runner and
         'mv -T -- "${WORK}" "${RESULT}.failed_or_incomplete.$$' in runner,
         "failure quarantine missing")
    need(runner.index('seal_dir "${WORK}"') <
         runner.index('mv -T -- "${WORK}" "${RESULT}"'),
         "result is not sealed before atomic publication")
    need("retry=false" in runner, "no-retry marker missing")
    exactly(runner,
            "/usr/bin/timeout --signal=TERM --kill-after=60s 21600s")

    # Exact bootstrap whitelist and the structural/timing artifact gates.
    need("${#error_lines[@]}\" -eq 1" in runner,
         "exact one-error bootstrap gate missing")
    need("3f0791c8c38447275968806360703faa95ef6a45ae53bd3502d09a6c535049e1" in runner,
         "bootstrap block hash is not exact")
    need("other_error_fatal_tim209_opt150_count=0" in runner,
         "non-bootstrap error/TIM-209/OPT-150 gate missing")
    for artifact in ["reports/area.rpt", "reports/qor.rpt",
                     "reports/timing_setup.rpt",
                     "reports/timing_hold_diagnostic.rpt",
                     "reports/constraint_setup.rpt",
                     "reports/precompile_loop_gate.rpt",
                     "reports/constraint_max_capacitance.rpt",
                     "reports/constraint_max_transition.rpt",
                     "reports/constraint_max_fanout.rpt",
                     "reports/port_count.txt"]:
        need(artifact in runner, "artifact gate missing: " + artifact)
    need(runner.count("This design has no violated constraints.") == 3,
         "max-cap/transition/fanout gates are incomplete")

    # Parse the worst path rather than the last printed path.  The receipt is
    # raw/pending review and cannot upgrade directed or CPU opportunity data.
    need("return min(float(value) for value in matches)" in runner,
         "WNS parser does not select the minimum slack")
    need("setup_met = base['setup_wns_ns'] >= 0.0 and tsbg['setup_wns_ns'] >= 0.0" in runner,
         "matched setup gate missing")
    need("area_ratio <= 1.10 and port_equal and setup_met" in runner,
         "candidate gate is not area/ports/setup conjunction")
    need("PASS_RAW_M1992_C2_TSBG_B4_MATCHED_DC_PENDING_INDEPENDENT_RESULT_REVIEW" in runner,
         "result is not explicitly raw/pending independent review")

    required_boundaries = [
        "'hold_closed': False",
        "'exact_rtl_cycle_speedup': False",
        "'power': False",
        "'energy': False",
        "'same_area': False",
        "'system_speedup': False",
        "'paper_ppa_ready': False",
        "'both_axes_own_bundle4_candidate_state': True",
        "'physical_schedule_ablation_not_conventional_baseline_ppa': True",
        "'full_conventional_baseline_area_priced': False",
        "'state_arrays_synthesized_as_standard_cells': True",
        "'layer_private_cache_domain': True",
        "'weight_domain_transition_requires_reset_or_rebind': True",
        "'cross_layer_flush_or_rebind_implemented': False",
        "'production_g48_dynamically_verified': False",
        "'exact_cycle_ratio': False",
    ]
    for token in required_boundaries:
        need(token in runner, "claim boundary missing: " + token)


def main():
    paths = {
        "runner": RUNNER, "filelist": FILELIST, "rtl": RTL,
        "adapter": ADAPTER, "tcl": TCL, "sdc": SDC,
        "m1990": M1990, "m1866": M1866, "docs359": DOCS359,
    }
    observed = {name: sha(path) for name, path in paths.items()}
    need(observed == EXPECTED, "reviewed identity mismatch: %r" % observed)
    runner = RUNNER.read_text()
    filelist = FILELIST.read_text()
    rtl = RTL.read_text()
    static_accept(runner, filelist, rtl)

    mutations = [
        ("same_schedule_modes",
         runner.replace("axis_modes=(0 1)", "axis_modes=(1 1)", 1),
         filelist, rtl),
        ("extra_source_group_override",
         runner.replace('ELAB_PARAMETERS="SCHEDULE_MODE=${mode}"',
                        'ELAB_PARAMETERS="SCHEDULE_MODE=${mode},SOURCE_GROUPS=12"', 1),
         filelist, rtl),
        ("wrong_top",
         runner.replace("DESIGN=m1880_c2_tsbg_b4_real_channel_signed_frontend",
                        "DESIGN=m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter", 1),
         filelist, rtl),
        ("filelist_third_source", runner,
         filelist + "rtl_m1880/m1880_sva.sv\n", rtl),
        ("remove_source_review_seal",
         runner.replace('verify_dir_seal "${SOURCE_REVIEW_DIR}"',
                        ': # removed source review seal', 1), filelist, rtl),
        ("remove_m1866_seal",
         runner.replace('verify_dir_seal "${M1866_DIR}"',
                        ': # removed M1866 seal', 1), filelist, rtl),
        ("remove_m1866_receipt_identity",
         runner.replace("'m1866_cpu_premodel_review_sha256': sha(m1866)",
                        "'m1866_cpu_premodel_review_sha256': 'unbound'", 1),
         filelist, rtl),
        ("loosen_review_authorization",
         runner.replace("{'dc_shell_runs': 2, 'all_other_eda_runs': 0}",
                        "{'dc_shell_runs': 3, 'all_other_eda_runs': 0}", 1),
         filelist, rtl),
        ("remove_dc_wall_timeout",
         runner.replace("/usr/bin/timeout --signal=TERM --kill-after=60s 21600s",
                        "/usr/bin/true", 1), filelist, rtl),
        ("loosen_dc_wall_timeout",
         runner.replace("--kill-after=60s 21600s",
                        "--kill-after=600s 43200s", 1), filelist, rtl),
        ("last_path_not_wns",
         runner.replace("return min(float(value) for value in matches)",
                        "return float(matches[-1])", 1), filelist, rtl),
        ("loosen_area_gate",
         runner.replace("area_ratio <= 1.10 and port_equal and setup_met",
                        "area_ratio <= 1.20 and port_equal and setup_met", 1),
         filelist, rtl),
        ("claim_exact_cycle",
         runner.replace("'exact_cycle_ratio': False",
                        "'exact_cycle_ratio': True", 1), filelist, rtl),
        ("claim_g48_dynamic",
         runner.replace("'production_g48_dynamically_verified': False",
                        "'production_g48_dynamically_verified': True", 1),
         filelist, rtl),
        ("claim_cross_layer_flush",
         runner.replace("'cross_layer_flush_or_rebind_implemented': False",
                        "'cross_layer_flush_or_rebind_implemented': True", 1),
         filelist, rtl),
        ("hide_baseline_state_bias",
         runner.replace("'both_axes_own_bundle4_candidate_state': True",
                        "'both_axes_own_bundle4_candidate_state': False", 1),
         filelist, rtl),
        ("rtl_g12_default", runner, filelist,
         rtl.replace("parameter int SOURCE_GROUPS = 48",
                     "parameter int SOURCE_GROUPS = 12", 1)),
        ("second_schedule_branch", runner, filelist,
         rtl.replace("assign load_ready =", "if (SCHEDULE_MODE == 0) begin end\n    assign load_ready =", 1)),
    ]
    results = []
    for name, mr, mf, mrtl in mutations:
        rejected = False
        reason = ""
        try:
            static_accept(mr, mf, mrtl)
        except AssertionError as exc:
            rejected = True
            reason = str(exc)
        need(rejected, "mutation escaped independent predicate: " + name)
        results.append({"name": name, "rejected": True, "reason": reason})

    # Explicit WNS adversarial sample: the old matches[-1] parser would accept
    # this report, whereas the reviewed min() parser correctly returns -0.25.
    sample = "slack (VIOLATED) -0.2500\nslack (MET) 0.1000\n"
    matches = re.findall(r"slack \((?:MET|VIOLATED)\)\s+(-?[0-9]+(?:\.[0-9]+)?)", sample)
    need(min(float(value) for value in matches) == -0.25,
         "adversarial WNS sample not rejected")
    need(float(matches[-1]) == 0.1,
         "adversarial sample does not distinguish old parser")

    out = {
        "schema": "m1993_m1992_independent_static_mutation_hammer_r1_v1",
        "status": "PASS_M1993_INDEPENDENT_STATIC_AND_MUTATION_HAMMER",
        "identity": observed,
        "static_checks_pass": True,
        "mutation_count": len(results),
        "mutations_rejected": len(results),
        "mutations": results,
        "adversarial_wns": {
            "sample_slacks_ns": [-0.25, 0.1],
            "reviewed_min_parser_ns": -0.25,
            "superseded_last_match_parser_ns": 0.1,
            "reviewed_parser_rejects_false_setup_met": True,
        },
        "eda_launched": False,
        "license_query_launched": False,
    }
    destination = Path(__file__).with_name("mutation_results.json")
    destination.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(out["status"])


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("FAIL_M1993: %s" % exc, file=sys.stderr)
        raise
