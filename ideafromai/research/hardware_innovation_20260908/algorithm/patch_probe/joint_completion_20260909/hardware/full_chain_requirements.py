"""Whole-r1 requirements and resource-demand bounds from existing real data.

This does NOT synthesize a whole-layer completion time from 192 local slices.
Full spatial sn2 gates and W2 are absent.  The Conv2 vector/port interval is
conditional on the dense-W2 convention of the existing GPU counter.
"""
from pathlib import Path
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
PATCH = HERE.parents[1]
C = PATCH/'partial_completion'
T, CHANNELS, HEIGHT, WIDTH, P, HG = 10, 96, 240, 320, 4, 8
PREFIX=(2,3,7)


def read(p):return json.loads(p.read_text())


def main():
    batches=read(PATCH/'dependency/source_batch_histogram.json')
    elements=read(PATCH/'dependency/source_element_histogram.json')
    ev={r['file']:r for r in elements['frames']}
    bv={r['file']:r for r in batches['frames']}
    files=[r['file'] for r in batches['frames'] if r['split']=='valid']
    c2={r['file']:r for r in read(C/'integer_exact_valid825/common3_diagonal_34_exact_frames.json')}
    dep=np.load(C/'integer_deployment/common3_diagonal_34.npz')
    w=dep['weight_int8'].reshape(CHANNELS,864)
    a=dep['temporal_int16']
    any_w=np.any(w.reshape(12,8,864)!=0,axis=1)
    assert any_w.all() # Makes whole-P4/H8 column request counts exact.
    pop=np.array([i.bit_count() for i in range(1024)],dtype=np.int64)
    pm=sum(1<<t for t in PREFIX)
    pp=np.array([(i&pm).bit_count() for i in range(1024)],dtype=np.int64)
    prefix_live=np.array([bool(i&pm) for i in range(1024)])
    dense_values=T*CHANNELS*HEIGHT*WIDTH
    positions=HEIGHT*WIDTH
    n_groups=positions//P
    frame_rows=[]
    for file in files:
        bh=np.asarray(bv[file]['hist'],dtype=np.int64)
        eh=np.asarray(ev[file]['hist'],dtype=np.int64)
        # Histograms describe whole frames. Multiplication by Hgroups is exact
        # because EVERY real H8 W vector has at least one nonzero coefficient.
        row=dict(file=file,
            Conv1_nonzero_source_terms_before_W_zero=int(eh@pop)*CHANNELS,
            prefix_nonzero_source_terms_before_W_zero=int(eh@pp)*CHANNELS,
            Conv1_P4_H8_k_t_vector_updates=int(bh@pop)*12,
            prefix_P4_H8_k_t_vector_updates=int(bh@pp)*12,
            Conv1_W64_uses_one_full_T_reuse_epoch=int(bh[1:].sum())*12,
            prefix_W64_uses_one_reuse_epoch=int(bh[prefix_live].sum())*12,
            nonempty_P4_K_source_words=int(bh[1:].sum()),
            Conv2_active_source_scalars=c2[file]['conv2_source_active_scalars'],
            Conv2_valid_fanout_weighted_add_terms_dense_W2=c2[file]['conv2_active_terms'],
            PSN_nonzero_A_products_before_known_zero_Y_skip=int(np.count_nonzero(a))*positions*CHANNELS,
            prefix_PSN_products_before_known_zero_Y_skip=int(np.count_nonzero(a[:,PREFIX]))*positions*CHANNELS,
            exact_gate_decisions=dense_values,
            fixed_BN2_element_evaluations=dense_values,
            shortcut_FP32_additions=dense_values,
        )
        # For a dense H8 W2 vector and P4 masks: 1..4 active p, hence 8..32
        # scalar terms per vector update. Location data are needed to tighten.
        row['Conv2_P4_H8_k_t_vector_update_lower']=math.ceil(row['Conv2_valid_fanout_weighted_add_terms_dense_W2']/32)
        row['Conv2_P4_H8_k_t_vector_update_upper']=row['Conv2_valid_fanout_weighted_add_terms_dense_W2']//8
        frame_rows.append(row)
    mean={k:float(np.mean([r[k] for r in frame_rows])) for k in frame_rows[0] if k!='file'}
    conv_reduction=1-mean['prefix_P4_H8_k_t_vector_updates']/mean['Conv1_P4_H8_k_t_vector_updates']
    weight_reduction=1-mean['prefix_W64_uses_one_reuse_epoch']/mean['Conv1_W64_uses_one_full_T_reuse_epoch']
    # Valid regular Conv1 halo pixels when independently staging one horizontal
    # P4 group; these are repeated physical source-tile loads, not unique pixels.
    halo_y=sum(min(HEIGHT,y+2)-max(0,y-1) for y in range(HEIGHT))
    halo_x=sum(min(WIDTH,x+P+1)-max(0,x-1) for x in range(0,WIDTH,P))
    halo_pixels=halo_y*halo_x
    source_bytes=dense_values//8
    output_bytes=dense_values*4
    result=dict(
        scope='same current common3 integer student, full 10 fixed validation frames for shape/support/Conv2 aggregates; full schedule unavailable',
        not_claimed=['whole-layer model completion steps','RTL cycles','PPA','same precision if Conv2 were changed to INT8','full-network speed'],
        identity=dict(
            Conv1='source theta and fixed norm1 gain folded in real per-H dyadic W8; output Yi24',
            PSN='compiled current A16 Q14, Ui48, all original T10 labels and continuous theta*g retained',
            Conv2='original model FP32 W2/compute, followed by fixed norm2 and continuous identity add; no local W2 export',
            new_training='local train16/valid4 FP32 norm1 students are separate numerical identities; their gates are not inserted into this complete-chain table',
            source_histogram='same four-fixed-BN parent before r1.conv1; changing downstream r1 conv1/sn2 cannot change this source',
        ),
        graph_boundary='sn1(theta*g) ready plus independent continuous identity ready -> Conv1 -> folded norm1/PSN -> theta*g -> Conv2 -> fixed norm2 -> +identity -> dense block output',
        actual_shapes=dict(T=T,C=CHANNELS,H=HEIGHT,W=WIDTH,P4=n_groups,H8_groups=12,kernel_terms=864),
        means_per_frame=mean,frames=frame_rows,
        numeric_bounds=dict(Y_abs_max=int(dep['Y_abs_bound'].max()),U_abs_max=int(dep['U_abs_bound'].max()),
            Y_storage_bits=24,U_work_bits=48,source_theta=float(dep['theta_source']),output_theta=float(dep['theta_output']),
            W1_nonzero=int(np.count_nonzero(w)),W1_elements=int(w.size),all_H8_W1_vectors_have_nonzero=True,
            W2_zero_pattern_and_accumulation_range='unavailable; dense-W2 activity count is an upper logical term count, not an admitted integer bound'),
        resource_point=dict(
            inherited=dict(P=4,H=8,Conv1='32 INT24 add lanes',PSN='8 signed24x16/48 MAC lanes',compare='8 lanes',W_port_bits=64,state_port_bits=64,W_pool_bytes=131072,state_pool_bytes=24576,payload_budget_bytes=2048),
            Conv2_extension_required='FP32 add and fixed-BN affine units/latency must be specified for both axes; inherited integer datapath cannot execute original FP32 W2 as-is',
            simultaneous_weight_policy='phase reuse of same 128KiB coefficient pool: Conv1 W8+bitmap; Conv2 H32 W32 stripe fits 110592B, three stripes. No free full-W2 residency.',
            optional_mapping='A Conv2 H8 coefficient vector is 32B/four 64-bit beats; one active t/P4 update reuses it over four spatial lanes. Reuse across times/positions needs paid state/caching and is common to all controls.',
        ),
        finite_resource_demand_bounds=dict(
            Conv1_time_major_W64_read_beats=mean['Conv1_P4_H8_k_t_vector_updates'],
            Conv1_time_major_Conv32_issues=mean['Conv1_P4_H8_k_t_vector_updates'],
            prefix_time_major_W64_read_beats=mean['prefix_P4_H8_k_t_vector_updates'],
            prefix_time_major_Conv32_issues=mean['prefix_P4_H8_k_t_vector_updates'],
            full_T_reuse_W64_read_beats=mean['Conv1_W64_uses_one_full_T_reuse_epoch'],
            PSN8_issues_before_zero_Y_and_CSE=mean['PSN_nonzero_A_products_before_known_zero_Y_skip']/8,
            compare8_issues_for_final_gates=dense_values/8,
            Conv2_vector_issue_interval_dense_W2=[mean['Conv2_P4_H8_k_t_vector_update_lower'],mean['Conv2_P4_H8_k_t_vector_update_upper']],
            Conv2_W64_beats_if_time_major_one_vector_buffer_dense_W2=[4*mean['Conv2_P4_H8_k_t_vector_update_lower'],4*mean['Conv2_P4_H8_k_t_vector_update_upper']],
            interpretation='Finite port/unit demand, not sums of completion cycles. MAC/vector issues are before documented PSN zero-Y/CSE optimization. Source/halo port conflicts and backpressure need full masks; different units may overlap.',
            PSN_strong_control='ordinary zero-Y skipping, shared coefficients, one-U operation, constant matrix compilation and same-P resource reorder remain legal. No full-layer assumption that every nominal PSN product is issued.',
        ),
        ordinary_halo_and_memory=dict(
            packed_source_tensor_bytes=source_bytes,
            packed_intermediate_gate_tensor_bytes=source_bytes,
            full_Y24_tensor_not_required_bytes=dense_values*3,
            dense_FP32_identity_bytes=output_bytes,dense_FP32_block_output_bytes=output_bytes,
            independent_P4_Conv1_source_halo_pixel_uses=halo_pixels,
            independent_P4_source_halo_payload_bytes=halo_pixels*CHANNELS*T//8,
            local_Conv1_P4_input_shape=[3,6,96,10],
            local_Conv1_input_payload_bytes=3*6*96*10//8,
            local_Conv1_source_row_layout='one c/kh word contains six spatial positions x T10 = 60 bits in one 64-bit word; 288 words = 2304B. Three kw masks use the same fetched row.',
            max_NRV64_bytes=864*8,
            source_buffer_note='2304B input and at most6912B NRV can reuse one8192B buffer with reverse construction; no full10-plane source line buffer is assumed',
            full_T_three_row_gate_linebuffer_bytes=3*WIDTH*CHANNELS*T//8,
            single_T_three_row_gate_linebuffer_bytes=3*WIDTH*CHANNELS//8,
            simple_complete_control='materialize packed sn2 gates once (9.216MB); Conv2 processes H32 coefficient stripes with an ordinary per-T three-row gate buffer (11.52KB), fixed offsets and correct border masks. It avoids Conv1 halo recomputation.',
            optional_materialized_boundary_bytes=dict(gate_write=source_bytes,gate_reads_three_H32_stripes=3*source_bytes,identity_read=output_bytes,output_write=output_bytes,W1_cold=82944,W2_cold_FP32=331776),
            traffic_limit='Identity/output traffic is a declared materialized endpoint, not inevitable DRAM traffic in every fused system. Fused/streamed alternatives must pay lifetime/backpressure and are allowed equally.',
            halo_strong_control='Do not force each Conv2 P4 to recompute18 sn2 pixels; ordinary stored gate bits/line buffers prevent that 4.5x producer expansion.',
        ),
        ideal_free_tail_ceiling=dict(
            assumption='After real three-prefix Conv1 columns, an oracle supplies all still-unknown final gates with no error/cost. This is an unattainable screening ceiling, not a measured predictor.',
            Conv1_scalar_source_term_saving_before_W_zero=1-mean['prefix_nonzero_source_terms_before_W_zero']/mean['Conv1_nonzero_source_terms_before_W_zero'],
            Conv1_vector_issue_saving=conv_reduction,
            Conv1_best_full_T_W_reuse_request_saving=weight_reduction,
            direct_PSN_term_saving_if_prefix_contributions_still_evaluated=1-int(np.count_nonzero(a[:,PREFIX]))/int(np.count_nonzero(a)),
            same_student_Conv2_BN2_shortcut_saving=0,
            whole_chain_formula='S <= f_Conv1*(1-r_Conv1)+f_PSN*(1-r_PSN) minus prediction/index/state tax, on an actually measured common full-chain schedule. Stage fractions unavailable; byte/issue ratios cannot replace them.',
            minimum_Conv1_service_fraction_for_15pct_if_no_other_saving=0.15/conv_reduction,
            minimum_W_service_fraction_for_15pct_if_only_full_T_reuse_W_saving=0.15/weight_reduction,
            gate_result='whole-chain >=15% opportunity not established or disproved by current data; full layer masks/W2/port timeline required',
        ),
        minimum_additional_capture=dict(
            frames='first existing diverse validation frame, then the same four already used; no new selection',
            common_input='packed source gate[T10,C96,240,320], theta, plus tensor layout; 9.216MB/frame',
            per_student_output='packed complete sn2 gate in same layout, plus predictor accepted/needed-column masks if conditional; 9.216MB/frame for gate',
            parameters='real r1 Conv2 W96x96x3x3 FP32 and any bias; norm2 gamma/beta/mean/var/eps; actual sn1/source and output theta',
            numeric_windows='one interior and one border output rectangle, full T10/C96; save corresponding Conv2-input halo, identity, Conv2 preBN and final block output. Avoid full294.912MB identity capture.',
            upstream_identity='shape/dtype/range and whether producer retains or materializes shortcut; original continuous identity is not recoverable from sn1 bits',
        ),
    )
    (HERE/'full_chain_requirements.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(means=mean,ceiling=result['ideal_free_tail_ceiling']),ensure_ascii=False))


if __name__=='__main__':main()
