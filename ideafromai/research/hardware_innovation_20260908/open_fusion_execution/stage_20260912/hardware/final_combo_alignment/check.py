"""Bind final GPU R24+onepass exports to already executed local hardware rows.

No simulator replay, GPU, training, hash, or output substitution is performed.
Logged whole-domain BN stats are used only to verify the out-of-scope boundary,
never as free measured hardware inputs or to add separate timing tables.
"""
from pathlib import Path
import json
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
HARDWARE=HERE.parent
sys.path.insert(0,str(HARDWARE))
import rank_fusion
stage=rank_fusion.stage
common=rank_fusion.common
ALGORITHM=HARDWARE.parent/'algorithm'
EXPORT=ALGORITHM/'hardware_exports'


def compare(a,b):
    a,b=np.asarray(a),np.asarray(b)
    r=dict(shape_a=list(a.shape),shape_b=list(b.shape),dtype_a=str(a.dtype),dtype_b=str(b.dtype),values=int(a.size))
    if a.shape!=b.shape or a.dtype!=b.dtype:
        r.update(same_shape_dtype=False,bitwise_equal=False)
        return r
    aa=np.ascontiguousarray(a).tobytes();bb=np.ascontiguousarray(b).tobytes()
    diff=np.frombuffer(aa,np.uint8).reshape(a.size,a.dtype.itemsize)!=np.frombuffer(bb,np.uint8).reshape(b.size,b.dtype.itemsize)
    r.update(same_shape_dtype=True,bitwise_equal=aa==bb,
        bitwise_element_differences=int(np.count_nonzero(diff.any(axis=1))))
    if a.dtype.kind in 'biuf':
        delta=a.astype(np.float64)-b.astype(np.float64)
        r.update(value_differences=int(np.count_nonzero(a!=b)),max_abs=float(np.max(np.abs(delta))) if a.size else 0.0)
    return r


def fields(old,new):
    result=dict(missing_in_new=sorted(set(old)-set(new)),added_in_new=sorted(set(new)-set(old)),fields={})
    for k in sorted(set(old)&set(new)):result['fields'][k]=compare(old[k],new[k])
    result['all_fields_bitwise_equal']=not result['missing_in_new'] and not result['added_in_new'] and all(r['bitwise_equal'] for r in result['fields'].values())
    return result


def local_student(axis):
    chain=common.FULL.parents[2]
    return (chain/'temporal_structured_recovery/stage128x256/identity_permuted_base.npz' if axis=='ordinary'
            else chain/'fast_temporal_recovery_lifting40/stage320/fast_raw_diagonal.npz')


def main():
    index=json.loads((EXPORT/'index.json').read_text())
    prior=json.loads((HARDWARE/'rank_fusion.json').read_text())
    report=dict(scope=__doc__,axes={},hardware_rerun=False,new_AEE=False,
        overall_status='partial',local_cut='I24 -> source/preview/sn2 -> updated I24 -> projection gate and PED U24/V96. Native projection/global onepass BN/final ADD are downstream and unpriced here.')
    for axis,mode in [('ordinary','original_ordered24'),('lifting_raw','activation_whitened24')]:
        path=EXPORT/axis
        if axis not in index or not (path/'000_zurich_city_09_a_0001.npz').exists():continue
        oldpath=common.FULL/'capture'/axis
        old=common.read_npz(oldpath/'000_zurich_city_09_a_0001.npz')
        oldq=common.read_npz(oldpath/'parameters.npz');oldp=common.read_npz(oldpath/'live_parameters.npz')
        new=common.read_npz(path/'000_zurich_city_09_a_0001.npz')
        newq=common.read_npz(path/'deployed_constants.npz');newp=common.read_npz(path/'live_parameters.npz')
        bn=common.read_npz(path/'BN_parameters.npz')
        original_student=common.read_npz(local_student(axis));exported_student=common.read_npz(path/'student_parameters.npz')
        generatedq,_,rank,_=rank_fusion.parameters_and_gold(old,oldq,'corner',axis,mode)
        ar=dict(mode=mode,rank=rank,export_directory=str(path),
            original_student_path=str(local_student(axis)),
            student_archive=fields(original_student,exported_student),
            deployed_constants=fields(generatedq,newq),live_parameters=fields(oldp,newp),
            BN_parameter_comparison=fields(dict(gamma=oldp['proj_bn_gamma'],beta=oldp['proj_bn_beta'],eps=oldp['proj_bn_eps']),bn),
            metadata={},windows={},matching_service_rows=[],new_AEE=False)
        for name in ['frame_name','window_geometry_json','bit_order','tensor_order','spatial_origin_yx','producer_order']:
            ar['metadata'][name]=compare(old[name],new[name])
        program=json.loads((HARDWARE/'source_rtl_inputs'/f'{axis}_program.json').read_text())
        source_program=json.loads((common.FULL.parent/'two_stage_writeback'/f'{axis}_fused_program.json').read_text())
        ar['source_program']=dict(instructions=len(program),same_original_compiled_instruction_fields=program==source_program,
            source_student_and_exported_constants_compared=True)
        for label in ['corner','interior']:
            rq,candidate,rank,delta=rank_fusion.parameters_and_gold(old,oldq,label,axis,mode)
            relevant=['I24','sn1_gate','preview_Z_shared','preview_shared_raw','preview_tail_raw','preview_BN1_Y','sn2_gate','updated_I24','proj_gate']
            wr=dict(local_input_and_intermediates={key:compare(old[label+'_'+key],new[label+'_'+key]) for key in relevant},
                PED_archive_containers=dict(independent_gold=str(candidate[label+'_continuous_q24'].dtype),GPU_capture=str(new[label+'_continuous_q24'].dtype),wire_format='signed24; dtype storage difference is not a numerical change'))
            # The oracle stores signed integers in int64, the GPU archive int32.
            # The wire format is signed24; compare that common exact container
            # and record the storage distinction without calling it a change.
            wr['actual_GPU_PED_signed24_values']=compare(candidate[label+'_continuous_q24'].astype(np.int32),new[label+'_continuous_q24'])
            wr['PED_wire_bytes_same']=stage.integer_chain.pack24(candidate[label+'_continuous_q24'])==stage.integer_chain.pack24(new[label+'_continuous_q24'])
            wr['native_projection_input_branch_same']=compare(old[label+'_proj_conv_fp32'],new[label+'_proj_conv_fp32'])
            stats=new['proj_bn_onepass_statistics']
            scale,bias=stats[3][None,:,None,None],stats[4][None,:,None,None]
            norm=np.float32(np.float32(new[label+'_proj_conv_fp32']*scale)+bias)
            continuous=np.float32(new[label+'_continuous_q24'].astype(np.float32)*np.float32(2.0**-14))
            final=np.float32(new[label+'_proj_norm_fp32']+continuous)
            wr['unpriced_downstream_boundary']=dict(
                logged_full_domain_shape=new['proj_bn_full_input_shape'].tolist(),domain_elements_per_channel=int(np.prod(new['proj_bn_full_input_shape'])//96),
                function=str(new['proj_bn_function']),statistics_shape=list(stats.shape),
                native_conv_to_logged_onepass_affine=compare(norm,new[label+'_proj_norm_fp32']),
                normalized_plus_actual_q24_PED=compare(final,new[label+'_ped_output_fp32']),
                onepass_norm_vs_old_CUDA_norm=compare(old[label+'_proj_norm_fp32'],new[label+'_proj_norm_fp32']),
                final_combination_vs_old_R32_CUDA=compare(old[label+'_ped_output_fp32'],new[label+'_ped_output_fp32']),
                interpretation='Logged full-domain stats are observational evidence only. These two local arithmetic checks add no hardware service and do not execute global statistics or native projection.')
            row=next(r for r in prior['rows'] if (r['axis'],r['window'],r['mode'])==(axis,label,mode))
            wr['retained_hardware_checks']=row['checks']
            wr['retained_preview_FP_boundary']=row['producer']['checks']
            wr['comparison_to_final_GPU_not_new_simulation']=True
            ar['windows'][label]=wr
            ar['matching_service_rows'].append(dict(source_file='../rank_fusion.json',axis=axis,window=label,mode=mode,
                service_slots=row['service_slots'],producer_end=row['producer_end'],consumer_begin=row['consumer_begin'],consumer_end=row['consumer_end']))
        conditions=[ar['student_archive']['all_fields_bitwise_equal'],ar['deployed_constants']['all_fields_bitwise_equal'],
            ar['live_parameters']['all_fields_bitwise_equal'],ar['BN_parameter_comparison']['all_fields_bitwise_equal'],
            all(v['bitwise_equal'] for v in ar['metadata'].values()),program==source_program]
        for wr in ar['windows'].values():
            conditions += [all(v['bitwise_equal'] for v in wr['local_input_and_intermediates'].values()),
                wr['actual_GPU_PED_signed24_values']['bitwise_equal'],wr['PED_wire_bytes_same'],
                wr['native_projection_input_branch_same']['bitwise_equal'],
                wr['unpriced_downstream_boundary']['native_conv_to_logged_onepass_affine']['bitwise_equal'],
                wr['unpriced_downstream_boundary']['normalized_plus_actual_q24_PED']['bitwise_equal']]
        ar['same_local_deployment_identity']=all(conditions)
        ar['rerun_required']=not ar['same_local_deployment_identity']
        # Accuracy linkage is factual, and can remain in progress while the
        # interface check already closes. Do not inherit earlier R32 AEE.
        ar['accuracy']={}
        for split in ['diverse10','valid825']:
            f=ALGORITHM/'combinations'/axis/split/'combo_summary.json'
            if f.exists():
                s=json.loads(f.read_text())
                ar['accuracy'][split]=dict(source=str(f),frames=s['frames'],complete=s.get('complete',False),
                    AEE_frame_mean=s['AEE_frame_mean'],valid_pixels=s['valid_pixels'])
        report['axes'][axis]=ar
        print(axis,'same_local_identity',ar['same_local_deployment_identity'],'rerun',ar['rerun_required'],flush=True)
    report['overall_status']='LOCAL_INTERFACE_ALIGNMENT_COMPLETE' if len(report['axes'])==2 and all(a['same_local_deployment_identity'] for a in report['axes'].values()) else 'needs_followup'
    report['both_final_combo_valid825_complete']=len(report['axes'])==2 and all(a['accuracy'].get('valid825',{}).get('complete',False) and a['accuracy']['valid825']['frames']==825 for a in report['axes'].values())
    (HERE/'alignment.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    assert all(a['same_local_deployment_identity'] for a in report['axes'].values()), 'Changed identity requires inspection and possibly new simulation.'

if __name__=='__main__':main()
