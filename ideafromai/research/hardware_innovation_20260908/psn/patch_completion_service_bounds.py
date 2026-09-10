"""Paid-count sensitivities only: no hardware cycles or PPA.

Consumes the saved four-frame P4/H8 sample counts. Scalar issue credits assume
perfect packing, no feedback stalls and unit Conv/PSN operation costs; they are
not a schedule. Logical W uses can be merged by real caches/broadcasts and are
not mapped SRAM transactions. No capture or network evaluation is rerun.
"""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'algorithm/patch_probe/partial_completion'


def main():
    probe = json.loads((BASE/'probe_results.json').read_text())
    cal = json.loads((BASE/'controller_calibration.json').read_text())
    exact = probe['axes']['exact']['splits']['valid']
    names = ['exact', 'individual_g3', 'individual_g2',
             'whole_word_g2', 'whole_word_g1.5']
    fields = ['lane_gated_conv1_active_terms', 'conv1_active_terms',
              'logical_W_vector_uses',
              'PSN_evaluations_excluding_known_raw_zero_columns',
              'confidence_comparisons', 'gates', 'issued_batches',
              'produced_time_columns']
    rows = []
    for name in names:
        s = probe['axes'][name]['splits']['valid']
        a = s['lane_gated_conv1_active_terms']
        p = s['PSN_evaluations_excluding_known_raw_zero_columns']
        q = s['confidence_comparisons']
        w = s['logical_W_vector_uses']
        assert q % 2 == 0
        core_credits = math.ceil((a+p)/32)
        cmp_points = []
        for rate in (2, 8, 32):
            check_credits = math.ceil(q/rate)
            cmp_points.append(dict(logical_comparisons_per_slot=rate,
                comparison_slot_lower_bound=check_credits,
                ideal_independent_stage_max=max(core_credits, check_credits)))
        rows.append(dict(axis=name, counts={f:s[f] for f in fields},
            ratios_to_exact5={f:s[f]/exact[f] for f in fields
                             if exact[f] and f != 'gates'},
            W_over_exact10Y_all_time_batch=w/exact['dense_time_batched_W_uses'],
            illustrative_32_scalar_core_credits=core_credits,
            comparison_sensitivity=cmp_points,
            comparisons_per_slot_to_hide_under_ideal_core=q/core_credits,
            nonexact_predicate_evaluations=q//2,
            factor_compensation_multiply_assignment_range_if_uncached={
                'P4_broadcast_lower':math.ceil(q/8),
                'no_P_reuse_upper':q//2,
                'condition':'all b_h*sum_A_R need multiplication; one ephemeral constant per context/check/t/h; global memoization or zero constants can reduce these numbers'},
            naive_full_864_directory_visits=s['issued_batches']*864))

    # Full per-channel table folds exact base threshold, residual mean and
    # b_h*sum(A_remaining); gamma*SD is common across H and precompiled.
    memory = []
    for width in (24,32,64):
        byt = width//8
        m = dict(width_bits=width,
            context_P4_H8_5Y_plus_U_bytes=32*6*byt,
            context_P4_H8_10Y_plus_U_bytes=32*11*byt,
            extra_5Y_per_context_bytes=32*5*byt,
            ordinary_raw_zero_base_t_h_threshold_bytes=10*96*byt,
            channel_offset_plus_shared_gamma_sd_table_bytes=(112*96+112)*byt,
            factorized_total_bytes=(10*96+96+3*112)*byt,
            unresolved_gate_bitmap_bytes=320//8,
            all_T10_answer_bitmap_bytes=320//8,
            five_Y_lane_valid_bits=5*32,
            common_time_dependency_bits=100)
        m['extra_channel_table_over_ordinary_base_bytes'] = (112*96+112-960)*byt
        m['extra_factorized_table_over_ordinary_base_bytes'] = (96+3*112)*byt
        m['extra_table_vs_more_Y_equal_capacity_contexts'] = {
            'channel_table':m['extra_channel_table_over_ordinary_base_bytes']/m['extra_5Y_per_context_bytes'],
            'factorized_table':m['extra_factorized_table_over_ordinary_base_bytes']/m['extra_5Y_per_context_bytes']}
        memory.append(m)

    ordinary = cal['final_controls']['independent_neuron_cost']['valid']
    shared = cal['final_controls']['shared_production_cost']['valid']
    da = shared['lane_gated_conv1_active_terms']-ordinary['lane_gated_conv1_active_terms']
    dp = shared['PSN_evaluations_excluding_known_raw_zero_columns']-ordinary['PSN_evaluations_excluding_known_raw_zero_columns']
    dq = shared['confidence_comparisons']-ordinary['confidence_comparisons']
    saved_w = ordinary['logical_W_vector_uses']-shared['logical_W_vector_uses']
    crossover = [dict(PSN_cost_over_Conv_add=rho_p,
                       comparison_cost_over_Conv_add=rho_q,
                       W_use_cost_over_Conv_add_must_exceed=(da+rho_p*dp+rho_q*dq)/saved_w)
                 for rho_p,rho_q in [(1,0.25),(1,1),(4,1)]]
    result = dict(
        scope='four fixed validation frames, each 64 P4 groups and all 12 H8 groups; 983040 sampled T10 gates, not full layer',
        sources=dict(probe=str(BASE/'probe_results.json'), calibration=str(BASE/'controller_calibration.json')),
        exact10Y_control=dict(logical_W_vector_uses=exact['dense_time_batched_W_uses'],
            lane_Conv_active_terms=exact['lane_gated_conv1_active_terms'],
            PSN_raw_zero_terms=exact['PSN_evaluations_excluding_known_raw_zero_columns'],
            interpretation='same exact row34 function, each required raw Y time produced once and all T share W delivery; other retention/ports not modeled'),
        axes=rows, storage_sensitivities=memory,
        raw_zero_formula={
            'BN':'Z_s,h = a_h Yraw_s,h + b_h',
            'partial_predictor':'bias_t + b_h sum_all A_t + a_h sum_seen A_t,s Yraw_s,h + mean_R - b_h sum_R A_t',
            'exact_base':'K_t,h = theta - bias_t - b_h sum_all A_t',
            'factor_check':'d = a_h sum_seen A_t,s Yraw_s,h - K_t,h + mean_R - b_h sum_R A_t; compare d with +/- gamma_t SD_R',
            'gamma':'fixed gamma or calibrated per-t gamma may be folded offline into SD; no mandatory runtime gamma multiplication',
            'table_access':'same seen and tested t across P4/H8: eight channel-offset reads + one shared SD broadcast can serve 32 predicates; factoring needs at most eight channel compensation multiplications shared across P4',
            'exact_precision':'Float64 reference over captured FP32 BN inputs; no 24/32-bit deployment or raw-zero reformulation equivalence is admitted'},
        calibrated_shared_increment=dict(extra_lane_Conv=da,extra_PSN_raw=dp,
            extra_comparisons=dq,saved_logical_W=saved_w,
            relative=cal['shared_vs_independent']['valid'],
            additive_cost_formula='delta=A_delta+rho_P*P_delta+rho_Q*Q_delta-rho_W*saved_W; excludes extra controller/source costs',
            break_even=crossover),
        exclusions=[
            'Issue credits are perfect-packing arithmetic lower bounds, not executable RTL cycles; final gate comparison/initialization and BN compensation excluded from core credits',
            'No physical W/source SRAM count; same-time broadcast/cache reuse must be granted to all axes',
            'Full directory scan is only a naive implementation cost, not measured reads; source metadata generation and retained source values are not in the supplied counts',
            'Table capacity does not supply table ports; multiple contexts may need arbitration or replication',
            'More Y may use the same SRAM macro count; no .db or mapped macro claim',
            'Changing P/H/order/capacity/check timing changes the statistical function; old AEE cannot be carried to another schedule',
            'Conv2, BN2, shortcut and pipeline waiting/halo remain outside this service envelope'],
        decision='No new RTL for calibrated shared objective; next useful step is compiler/format and threshold replay against ordinary controls, or a structure training change that produces a larger paid margin.')
    out=Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(rows=[dict(axis=r['axis'],W_over_exact10Y=r['W_over_exact10Y_all_time_batch'],core=r['illustrative_32_scalar_core_credits'],cmp_rate=r['comparisons_per_slot_to_hide_under_ideal_core']) for r in rows],memory32=memory[1],crossover=crossover),ensure_ascii=False))


if __name__ == '__main__':
    main()
