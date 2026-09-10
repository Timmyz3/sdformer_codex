"""Compact attribution from completed numeric/graph results; no new replay."""
from pathlib import Path
import json

HERE=Path(__file__).resolve().parent


def main():
    compiler=json.loads((HERE/'compiled_summary.json').read_text())
    numeric=json.loads((HERE/'numeric_result.json').read_text())
    resources=json.loads((HERE.parent.parent/'bn_state/support_service_resources.json').read_text())
    nf=numeric['completed_frames']
    assert nf==numeric['requested_frames']==10
    old_state=resources['persistent_Y']['rows']*96*24//8+sum(resources['other_common_storage_bytes'].values())
    common_other=old_state-92160-5760-5760-576
    # Same legal narrowing for the conventional MAC and compiled graphs.
    # Output thresholds fit signed25 in this student. Same128-byte output
    # transport and existing source/route/parameter buffering are retained.
    y_bits=compiler['inputs']['legal_common_formats']['Y']
    u_bits=compiler['inputs']['legal_common_formats']['U']
    y_state=320*96*y_bits//8
    tau_state=10*96*25//8
    direct_state=common_other+y_state+10*96*u_bits//8+tau_state+2*96*y_bits//8
    base=dict(frames=nf,
        PSN_MAC_vector_issues=sum(f['baseline_MAC_vector_issues'] for f in numeric['frames']),
        PSN_service_beats_in_old_finite_model=sum(f['baseline_PSN_service_beats'] for f in numeric['frames']),
        FC1_service_beats_same_forced_L16_frontend=sum(f['same_frontend_old_exact_L16']['FC1_beats'] for f in numeric['frames']),
        coefficient_read_words_same_frontend=sum(f['same_frontend_old_exact_L16']['coefficient_read_words'] for f in numeric['frames']),
        old_warm_chain_beats=sum(f['same_frontend_old_exact_L16']['warm_chain_beats'] for f in numeric['frames']),
        known_zero_Y_values=sum(f['source_known_zero_Y_values'] for f in numeric['frames']),
        actual_zero_Y_values=sum(f['real_zero_Y_values'] for f in numeric['frames']),
        original_raw_state_bytes=old_state, common_narrowed_Y_raw_bytes=y_state,
        ordinary_MAC_common_narrowed_raw_state_bytes=direct_state,
        common_retained_non_Y_U_tau_operand_bytes=common_other)
    graphs={}
    for label,desc in compiler['graphs'].items():
        quantities={key:sum(f['graph_arithmetic_counts_not_cycles'][label][key] for f in numeric['frames'])
                    for key in numeric['frames'][0]['graph_arithmetic_counts_not_cycles'][label]}
        states={}
        for order in ('official_topological_order','pressure_aware_ordinary_order'):
            stats=desc[order]
            raw=common_other+y_state+tau_state+stats['temporary_peak_bytes_96lane']+stats['two_operand_register_bits']*12
            states[order]=dict(raw_state_bytes=raw,
                              fits_old_raw_state_budget=raw<=old_state,
                              temporary_RF_bytes=stats['temporary_peak_bytes_96lane'],
                              two_operand_register_bytes=stats['two_operand_register_bits']*12,
                              NOT_a_sized_or_scheduled_macro=True)
        graphs[label]=dict(
            scalar_graph_additions=desc['additions'],
            reduction_vs_complete_separate_CSD_adds=1-desc['additions']/compiler['naive']['separate_CSD_total_adds'],
            observed_counts=quantities,
            known_zero_pruned_adds_per_frame=quantities['known_zero_pruned_vector_adds']/nf,
            arithmetic_issue_ratio_to_word_MAC_not_cycle_ratio=quantities['known_zero_pruned_vector_adds']/base['PSN_MAC_vector_issues'],
            serialized_RF_orders=states)
    result=dict(kind='post-run attribution only; whole-graph compiler completed, global hardware baseline still incomplete',
                base=base,graphs=graphs,
                admission=dict(official_complete_CMVM_algorithm=True,same_integer_matrix_proven=True,
                    numerical_all10_capture_values=True,actual_global_finite_port_timeline=False,
                    bit_demand_strict_termination_implemented=False,ASIC_area_or_power=False),
                unresolved=[
                    'The source-producer, FC2, BN2 and whole network remain outside the inherited S0 block0 boundary.',
                    'Mapping236/263 nodes to the old96MAC slots is a resource decision; an unrolled graph has different area and registers.',
                    'Ordinary known-zero propagation counts omit alias-select/mux control and cannot be called scheduled beats.',
                    'Two-register RF traffic is one legal static order, not an optimized finite-port scheduler; cross-p overlap is not yet scheduled.',
                    'Register alignment for a fully pipelined graph, coefficient/shift control, output backpressure and timing remain to be implemented.',
                    'Ordinary bit-demand strict decision execution remains necessary before evaluating a new threshold-related X.',
                    'Raw source-state savings from13/29-bit narrowing belong equally to ordinary MAC and compiled graphs.',
                    'Compiler bit-cost, width sums and RF traffic are not energy or PPA.'])
    (HERE/'attribution.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
