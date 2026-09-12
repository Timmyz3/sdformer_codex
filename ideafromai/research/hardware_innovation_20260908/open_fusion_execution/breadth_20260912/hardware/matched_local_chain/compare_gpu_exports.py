"""Bind the new stage320 GPU endpoint capture to already measured local rows.

No new simulation, no GPU invocation, no substituted gold or inherited825.
FP preview differences remain explicit even if downstream integer gates agree.
"""
from pathlib import Path
import importlib.util, json
import numpy as np

HERE=Path(__file__).resolve().parent
BREADTH=HERE.parents[1]
EXPORT=BREADTH/'algorithm/hardware_exports'
spec=importlib.util.spec_from_file_location('matched_chain_for_binding',HERE/'run.py')
chain=importlib.util.module_from_spec(spec);spec.loader.exec_module(chain)


def compare(a,b,wire=None):
    a,b=np.asarray(a),np.asarray(b)
    r=dict(shape_a=list(a.shape),shape_b=list(b.shape),dtype_a=str(a.dtype),dtype_b=str(b.dtype),values=int(a.size))
    if a.shape!=b.shape:
        return dict(**r,shape_equal=False,equal=False)
    if wire=='signed24':
        assert a.min()>=-(1<<23) and a.max()<(1<<23) and b.min()>=-(1<<23) and b.max()<(1<<23)
        a,b=a.astype(np.int32),b.astype(np.int32)
    elif wire=='gate':
        assert np.all((a==0)|(a==1)) and np.all((b==0)|(b==1))
        a,b=a.astype(np.uint8),b.astype(np.uint8)
    if a.dtype.kind=='f' and b.dtype.kind=='f':
        if a.dtype!=b.dtype:
            return dict(**r,shape_equal=True,dtype_equal=False,equal=False)
        av=np.ascontiguousarray(a).view(np.uint8).reshape(a.size,a.dtype.itemsize)
        bv=np.ascontiguousarray(b).view(np.uint8).reshape(b.size,b.dtype.itemsize)
        n=int(np.count_nonzero(np.any(av!=bv,axis=1)))
    else:
        n=int(np.count_nonzero(a!=b))
    r.update(shape_equal=True,wire_format=wire,element_differences=n,equal=n==0)
    if a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':
        r['max_abs']=float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if a.size else 0.0
    return r


def main():
    index=json.loads((EXPORT/'index.json').read_text())
    old=chain.common.read_npz(chain.common.FULL/'capture/ordinary/000_zurich_city_09_a_0001.npz')
    oldp=chain.common.read_npz(chain.common.FULL/'capture/ordinary/live_parameters.npz')
    axes={}
    preview_keys=['preview_u','preview_v','preview_A','preview_b','preview_theta_source','preview_theta_output',
        'bn1_gamma','bn1_beta','bn1_mean','bn1_var','bn1_eps']
    for structure in ['dense','contiguous34','lifting40']:
        folder=EXPORT/structure
        capture=folder/'000_zurich_city_09_a_0001.npz'
        if not capture.exists():
            axes[structure]=dict(status='CAPTURE_PENDING');continue
        gpu=chain.common.read_npz(capture)
        q=chain.common.read_npz(BREADTH/'algorithm/matched_training'/structure/'stage320/deployed_constants.npz')
        gq=chain.common.read_npz(folder/'deployed_constants.npz')
        gp=chain.common.read_npz(folder/'live_parameters.npz')
        assert set(q)==set(gq)
        parameters={k:compare(q[k],gq[k]) for k in q}
        preview_parameters={k:compare(oldp[k],gp[k]) for k in preview_keys}
        ar=dict(structure=structure,parameters=parameters,unchanged_preview_parameters=preview_parameters,
            capture_structure=str(gpu['trained_structure']),capture_steps=int(gpu['cumulative_new_GT_steps']),windows={},
            capture_path=str(capture),same_first_frame_AEE=index['axes'][structure]['exact_same_trained_endpoint_AEE'])
        for label in ['corner','interior']:
            cpu=chain.common.read_npz(HERE/(structure+'_'+label+'_cpu_endpoints.npz'))
            cost=json.loads((HERE/(structure+'_'+label+'.json')).read_text())
            keys={'source_gate':('sn1_gate','gate'),'preview_gate':('sn2_gate','gate'),
                'updated':('updated_I24','signed24'),'projection_gate':('proj_gate','gate'),'PED':('continuous_q24','signed24')}
            wr=dict(original_I24=compare(old[label+'_I24'],gpu[label+'_I24'],'signed24'),
                endpoints={name:compare(cpu[name],gpu[label+'_'+gname],wire) for name,(gname,wire) in keys.items()},
                FP_preview={name:compare(cpu[name],gpu[label+'_'+gname]) for name,gname in
                    [('cpu_preview_Z','preview_Z_shared'),('cpu_preview_raw','preview_shared_raw'),('cpu_preview_BN1','preview_BN1_Y')]},
                service_slots=cost['service_slots'],original_service_receipt=structure+'_'+label+'.json',
                original_CPU_checks=cost['producer']['checks'])
            wr['fixed_integer_endpoints_match']=wr['original_I24']['equal'] and all(v['equal'] for v in wr['endpoints'].values())
            wr['FP_preview_all_bitwise_equal']=all(v['equal'] for v in wr['FP_preview'].values())
            ar['windows'][label]=wr
        ar['status']='FIXED_INTEGER_ENDPOINTS_AND_PARAMETERS_MATCH' if (
            all(v['equal'] for v in parameters.values()) and all(v['equal'] for v in preview_parameters.values())
            and ar['capture_structure']==structure and ar['capture_steps']==320
            and all(w['fixed_integer_endpoints_match'] for w in ar['windows'].values())) else 'NEEDS_NUMERICAL_FOLLOWUP'
        stats=gpu['proj_bn_onepass_statistics']
        ar['unpriced_downstream_boundary']=dict(onepass_domain_shape=gpu['proj_bn_full_input_shape'].tolist(),
            onepass_statistics_shape=list(stats.shape),function=str(gpu['proj_bn_function']),
            included_in_service=False,native_or_BN_tables_added=False)
        ar['quality']={}
        path=BREADTH/'algorithm/valid825'/structure/(structure+'_summary.json')
        if path.exists():
            s=json.loads(path.read_text());ar['quality']['valid825']=dict(source=str(path),
                frames=s.get('frames'),complete=s.get('complete'),AEE_frame_mean=s.get('AEE_frame_mean'),valid_pixels=s.get('valid_pixels'))
        axes[structure]=ar
        print(structure,ar['status'],[(k,v['fixed_integer_endpoints_match'],v['FP_preview_all_bitwise_equal']) for k,v in ar['windows'].items()],flush=True)
    report=dict(evidence=__doc__,axes=axes,new_simulation=False,
        status='ALL_THREE_CAPTURED_INTEGER_ENDPOINTS_MATCH' if all(x['status']=='FIXED_INTEGER_ENDPOINTS_AND_PARAMETERS_MATCH' for x in axes.values()) else 'PENDING_OR_DIFFERENT',
        inference_limit='Finite actual GPU endpoint comparison links these fixed windows only; it is not a full-network CPU/RTL equivalence proof or full-frame cycle result.')
    (HERE/'gpu_alignment.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
