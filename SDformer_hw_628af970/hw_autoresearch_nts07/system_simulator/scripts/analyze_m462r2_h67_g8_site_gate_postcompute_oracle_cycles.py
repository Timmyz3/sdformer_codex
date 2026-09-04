#!/usr/bin/env python3
"""M462R2 correction: only the T10 all-site mask may drive the FFN gate.

M462 R1 correctly produced NO-GO for the observed data, but its summary chose
the maximum over both the token-only and all-site rows.  That selection rule
could falsely GO on a future token-only row, because token eligibility cannot
charge ATLIF.  R2 preserves the frozen R1 arithmetic/receipt implementation,
keeps token results as a separate diagnostic, and makes the 1.15/1.20/1.30
gate population exclusively ``t10_all_spatial_site``.
"""

from __future__ import print_function

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
BASE_PATH = (HW / "system_simulator/scripts/"
             "analyze_m462_h67_g8_ffn_postcompute_oracle_cycles.py")


def load_base():
    spec = importlib.util.spec_from_file_location("m462r2_frozen_r1", str(BASE_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen M462 R1 analyzer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


M = load_base()


def validate_contract(path):
    contract = M.strict_json(path)
    M.require(contract.get("schema") ==
              "m462r2_h67_g8_site_gate_postcompute_oracle_cycle_audit_contract_v1",
              "M462R2 contract schema drift")
    M.require(contract.get("status") ==
              "READY_EXACT_SHA_CPU_SITE_ONLY_GATE_CORRECTION",
              "M462R2 contract status drift")
    identities = {}
    for name, record in contract["identity"].items():
        target = M.resolve_identity(record)
        M.require(target.is_file() and not target.is_symlink(),
                  "M462R2 identity absent/symlink: " + name)
        actual = M.sha256(target)
        M.require(actual == record["sha256"],
                  "M462R2 identity SHA drift: " + name)
        identities[name] = target
    M.require(M.sha256(Path(__file__).resolve()) ==
              contract["identity"]["analyzer_r2"]["sha256"],
              "M462R2 self SHA drift")
    M.require(M.sha256(BASE_PATH) ==
              contract["identity"]["frozen_m462_r1_analyzer"]["sha256"],
              "M462R2 frozen R1 implementation drift")
    M.require(M.SAMPLES == 10 and M.TIMESTEPS == 10,
              "M462R2 literal S10/T10 invariant drift")
    M.require(contract["gate_population"] == {
        "eligible_mask_mode": "t10_all_spatial_site",
        "ineligible_diagnostic_mask_mode": "strict_token_tnhw",
        "reason": "ATLIF opportunity requires all literal T=10 entries at one spatial site to be eligible",
    }, "M462R2 gate population contract drift")
    M.require(contract["cycle_model"] == {
        "lanes": M.LANES,
        "samples": M.SAMPLES,
        "timesteps": M.TIMESTEPS,
        "global_envelope_cycles": M.ENVELOPE,
        "profile100_fc1_cycles": M.FC1_BASELINE,
        "profile100_fc2_cycles": M.FC2_BASELINE,
        "ffn_local_sn1_atlif_cycles": M.SN1_ATLIF_BASELINE,
        "ffn_local_sn2_atlif_cycles": M.SN2_ATLIF_BASELINE,
        "profile_normalization": "per_pair_per_role_integer_floor_B_times_S_div_D",
    }, "M462R2 cycle model drift")
    binding = contract["capture_binding"]
    for key in ("top_manifest_sha256", "top_outer_seal_file_sha256",
                "capture_manifest_sha256", "capture_outer_seal_file_sha256"):
        M.require(M.HEX64.match(binding[key]) is not None,
                  "M462R2 malformed capture binding: " + key)
    for key in ("executable_skip", "delta_aee", "valid825_accuracy",
                "measured_cycle_speedup", "system_speedup", "energy",
                "ppa", "headline"):
        M.require(contract["admission"][key] is False,
                  "M462R2 forbidden admission true: " + key)
    return contract, identities


def execute(contract_path, capture_root, output):
    output = Path(output).resolve()
    M.require(not output.exists(), "refusing to overwrite M462R2 output")
    start_sha = M.sha256(Path(__file__).resolve())
    contract, identities = validate_contract(contract_path)
    capture, _summary, _author = M.validate_r5_root(
        capture_root, contract, identities)
    m159 = M.strict_json(identities["m159"])
    accounted = m159["accounted_compute_cycles_per_frame"]
    M.require(int(accounted["fc1_plus_fc2"]) == M.LINEAR_BASELINE and
              int(accounted["full_ffn_subgraph_excluding_bn_residual"]) ==
              M.FFN_ACCOUNTED and
              int(accounted["global_envelope_cycles"]) == M.ENVELOPE and
              int(accounted["sn1_atlif"]) == M.SN1_ATLIF_BASELINE and
              int(accounted["sn2_atlif"]) == M.SN2_ATLIF_BASELINE,
              "M462R2 M159 accounted cycle drift")
    pairs = M.load_ffn_ledger(identities["ffn_ledger"])
    runtime = M.load_operator_runtime(identities["operator_runtime"], pairs)
    output.mkdir(parents=True)
    per_record_path = output / "m462r2_per_record_tau_mask_audit.csv"
    (denominators, selected_issue, selected_tokens, equal_tokens,
     selected_sites, site_data) = M.validate_and_accumulate(
         capture, pairs, runtime, per_record_path)
    invariants = M.full_mask_invariants(pairs, denominators)
    M.require(invariants == {
        "fc1": M.FC1_BASELINE, "fc2": M.FC2_BASELINE,
        "linear": M.LINEAR_BASELINE,
        "sn1_atlif": M.SN1_ATLIF_BASELINE,
        "sn2_atlif": M.SN2_ATLIF_BASELINE,
        "ffn_accounted": M.FFN_ACCOUNTED,
    }, "M462R2 literal S10/T10 full-mask invariant drift")
    frozen = M.aggregate_frozen_grid(
        pairs, denominators, selected_issue, selected_tokens,
        equal_tokens, selected_sites)
    M.attach_atlif_to_frozen_rows(frozen, per_record_path, pairs)
    for row in frozen:
        eligible = row["mask_mode"] == "t10_all_spatial_site"
        row["eligible_for_full_ffn_opportunity_gate"] = eligible
        row["meets_1p15_opportunity_gate"] = (
            eligible and row["total_accounted_postcompute_oracle_saved_cycles"] >=
            M.GATES[0][1])
    site_rows = [row for row in frozen
                 if row["mask_mode"] == "t10_all_spatial_site"]
    token_rows = [row for row in frozen
                  if row["mask_mode"] == "strict_token_tnhw"]
    M.require(len(site_rows) == len(M.TAU_GRID) and
              len(token_rows) == len(M.TAU_GRID),
              "M462R2 dual-mask row population drift")
    best_site = max(site_rows, key=lambda row:
                    row["total_accounted_postcompute_oracle_saved_cycles"])
    best_token = max(token_rows, key=lambda row:
                     row["total_accounted_postcompute_oracle_saved_cycles"])
    site_max = best_site["total_accounted_postcompute_oracle_saved_cycles"]
    extended = M.extended_cliff_diagnostic(site_data, pairs, denominators)
    frozen_path = output / "m462r2_frozen_tau_dual_mask_cycle_oracle.csv"
    extended_path = output / "m462r2_extended_posthoc_cliff_diagnostic.csv"
    M.write_csv(frozen_path, frozen)
    M.write_csv(extended_path, extended)
    summary = {
        "schema": "m462r2_h67_g8_site_gate_postcompute_oracle_cycle_audit_v1",
        "status": ("NO_GO_FROZEN_TAU_SITE_MASK_BELOW_1P15_OPPORTUNITY_GATE"
                   if site_max < M.GATES[0][1] else
                   "GO_FROZEN_TAU_SITE_MASK_REACHES_1P15_OPPORTUNITY_GATE"),
        "r1_correction": {
            "r1_observed_conclusion_changed": False,
            "r1_observed_status":
                "NO_GO_FROZEN_TAU_GRID_BELOW_1P15_OPPORTUNITY_GATE",
            "corrected_rule":
                "Only t10_all_spatial_site rows may drive the full-FFN opportunity gate.",
            "token_rows": "diagnostic only; FC1/FC2 opportunity, no ATLIF",
        },
        "identity": {
            "contract_sha256": M.sha256(contract_path),
            "analyzer_start_end_sha256": start_sha,
            "frozen_m462_r1_analyzer_sha256": M.sha256(BASE_PATH),
            "capture_top_manifest_sha256": M.sha256(
                Path(capture_root) / "manifest.sha256"),
            "capture_top_outer_seal_file_sha256": M.sha256(
                Path(capture_root) / "manifest.sha256.outer.seal.sha256"),
            "capture_payload_manifest_sha256": M.sha256(
                capture / "manifest.sha256"),
            "capture_payload_outer_seal_file_sha256": M.sha256(
                capture / "manifest.sha256.outer.seal.sha256"),
            "input_sha256": {name: M.sha256(path)
                              for name, path in identities.items()},
        },
        "population": {
            "samples": M.SAMPLES, "timesteps": M.TIMESTEPS,
            "ffn_pairs": M.PAIRS, "sample_pair_records": M.RECORDS,
            "tokens": M.TOKENS, "spatial_sites": M.SITES,
            "frozen_tau_points": len(M.TAU_GRID),
            "site_gate_rows": len(site_rows),
            "token_diagnostic_rows": len(token_rows),
        },
        "cycle_model": {
            "profile_normalization":
                "per_pair_per_role_integer_floor_saved_equals_B_times_S_div_D",
            "global_envelope_cycles": M.ENVELOPE,
            "required_savings": {name: value for name, value in M.GATES},
            "literal_s10_t10_full_mask_invariants": invariants,
        },
        "frozen_tau_site_gate_result": {
            "best_site_row": best_site,
            "maximum_site_mask_accounted_postcompute_oracle_saved_cycles": site_max,
            "meets_1p15_opportunity_gate": site_max >= M.GATES[0][1],
        },
        "frozen_tau_token_diagnostic": {
            "best_token_row": best_token,
            "eligible_for_full_ffn_opportunity_gate": False,
        },
        "extended_posthoc_cliff_diagnostic": {
            "part_of_frozen_capture_tau_receipts": False,
            "delta_aee_available": False,
            "admitted": False,
            "rows": extended,
        },
        "source_receipt_checks": {
            "r5_double_seal_and_author_binding": True,
            "npz_byte_logical_dtype_shape_receipts": True,
            "rho_tau_source_work_recomputed": True,
            "operator_runtime_s10_input_active_exact": True,
            "profile100_full_linear_mask_exact": M.LINEAR_BASELINE,
            "full_site_atlif_mask_exact": M.ATLIF_BASELINE,
        },
        "admission": {
            "sealed_checkpoint_bound_s10_postcompute_oracle_audit": True,
            "corrected_site_only_opportunity_gate": True,
            "frozen_tau_opportunity_counts": True,
            "extended_posthoc_cliff_diagnostic": True,
            "executable_skip": False, "delta_aee": False,
            "valid825_accuracy": False, "measured_cycle_speedup": False,
            "system_speedup": False, "energy": False, "ppa": False,
            "headline": False,
        },
        "files": {
            "per_record_tau_mask_audit": per_record_path.name,
            "frozen_tau_dual_mask_cycle_oracle": frozen_path.name,
            "extended_posthoc_cliff_diagnostic": extended_path.name,
        },
        "claim_boundary": (
            "M462R2 is a corrected, frozen H67-ep35/no-running S10 "
            "post-compute oracle opportunity audit. Only T10 all-site rows "
            "drive its opportunity gate. It admits no executable skip, "
            "Delta-AEE, valid825 accuracy, measured cycle speedup, system "
            "speedup, energy, PPA, or headline."),
    }
    summary_path = output / "m462r2_h67_g8_site_gate_cycle_oracle_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    M.require(M.sha256(Path(__file__).resolve()) == start_sha,
              "M462R2 analyzer changed during execution")
    M.require(M.sha256(identities["docs359"]) ==
              contract["identity"]["docs359"]["sha256"],
              "protected docs/359 changed during M462R2")
    manifest, outer = M.seal_output(output)
    print(json.dumps({
        "status": summary["status"],
        "maximum_frozen_tau_site_oracle_saved_cycles": site_max,
        "maximum_frozen_tau_token_diagnostic_saved_cycles":
            best_token["total_accounted_postcompute_oracle_saved_cycles"],
        "meets_1p15_opportunity_gate": site_max >= M.GATES[0][1],
        "manifest_sha256": M.sha256(manifest),
        "outer_seal_file_sha256": M.sha256(outer),
        "postcompute_oracle_only": True,
        "executable_skip": False, "system_speedup": False,
    }, sort_keys=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--capture-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    execute(args.contract.resolve(), args.capture_root.resolve(),
            args.output_dir.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
