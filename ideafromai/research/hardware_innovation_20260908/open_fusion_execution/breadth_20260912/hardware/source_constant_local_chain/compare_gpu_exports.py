"""Compare two measured PoT2 local chains with their actual GPU first-frame halos."""
from pathlib import Path
import importlib.util,json
import numpy as np
HERE=Path(__file__).resolve().parent
BREADTH=HERE.parents[1]
spec=importlib.util.spec_from_file_location('matched_compare',HERE.parent/'matched_local_chain/compare_gpu_exports.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
compare=base.compare

def arrays(path):
    with np.load(path) as f:return {k:f[k] for k in f.files}

def main():
    old=base.chain.common.read_npz(base.chain.common.FULL/'capture/ordinary/000_zurich_city_09_a_0001.npz')
    oldp=base.chain.common.read_npz(base.chain.common.FULL/'capture/ordinary/live_parameters.npz')
    preview_keys=['preview_u','preview_v','preview_A','preview_b','preview_theta_source','preview_theta_output','bn1_gamma','bn1_beta','bn1_mean','bn1_var','bn1_eps']
    axes={}
    for axis in ['dense','lifting40']:
        folder=BREADTH/'algorithm/source_constant_aee'/axis/'hardware_exports'
        gpu=arrays(folder/'000_zurich_city_09_a_0001.npz')
        q=arrays(BREADTH/'source_constant_probe'/axis/'deployed_constants.npz')
        gq=arrays(folder/'deployed_constants.npz');gp=arrays(folder/'live_parameters.npz')
        assert set(q)==set(gq)
        ar=dict(parameters={k:compare(q[k],gq[k]) for k in q},unchanged_preview_parameters={k:compare(oldp[k],gp[k]) for k in preview_keys},windows={})
        for label in ['corner','interior']:
            cpu=arrays(HERE/(axis+'_'+label+'_cpu_endpoints.npz'))
            keys={'source_gate':('sn1_gate','gate'),'preview_gate':('sn2_gate','gate'),'updated':('updated_I24','signed24'),'projection_gate':('proj_gate','gate'),'PED':('continuous_q24','signed24')}
            wr=dict(original_I24=compare(old[label+'_I24'],gpu[label+'_I24'],'signed24'),
                endpoints={k:compare(cpu[k],gpu[label+'_'+v],wire) for k,(v,wire) in keys.items()},
                FP_preview={k:compare(cpu[k],gpu[label+'_'+v]) for k,v in [('cpu_preview_Z','preview_Z_shared'),('cpu_preview_raw','preview_shared_raw'),('cpu_preview_BN1','preview_BN1_Y')]})
            wr['fixed_integer_endpoints_match']=wr['original_I24']['equal'] and all(v['equal'] for v in wr['endpoints'].values())
            wr['FP_preview_all_bitwise_equal']=all(v['equal'] for v in wr['FP_preview'].values())
            ar['windows'][label]=wr
        ar['fixed_integer_endpoints_and_parameters_match']=all(v['equal'] for v in ar['parameters'].values()) and all(v['equal'] for v in ar['unchanged_preview_parameters'].values()) and all(w['fixed_integer_endpoints_match'] for w in ar['windows'].values())
        ar['fresh_diverse10_quality']=json.loads((folder.parent/'quality.json').read_text())
        axes[axis]=ar
        print(axis,ar['fixed_integer_endpoints_and_parameters_match'],flush=True)
    report=dict(axes=axes,all_fixed_integer_endpoints_and_parameters_match=all(a['fixed_integer_endpoints_and_parameters_match'] for a in axes.values()),
        scope='Only two actual first-frame halos of each new PoT2 function, not whole-network equivalence or825; FP preview differences explicitly retained.',new_simulation=False)
    (HERE/'gpu_alignment.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
