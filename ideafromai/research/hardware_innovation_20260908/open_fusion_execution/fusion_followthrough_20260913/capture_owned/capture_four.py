"""Four independent matched-dense forwards; observer-only two-halo capture.

S48 is copied from the actual As accumulator passed to helper.observe, before
the existing write24. No reference array supplies any network operand.
"""
from pathlib import Path
import argparse
import json
import sys
import time

import numpy as np


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=lambda x:x.item() if isinstance(x,np.generic) else x.tolist())+'\n')


def read_npz(path):
    with np.load(path) as z:
        return {k:z[k] for k in z.files}


def validate_capture(path, constants, previous_first=None):
    """Post-capture CPU checks only; all S/g comparisons are independent."""
    a=read_npz(path)
    rows={}
    for label in ('corner','interior'):
        x=a[label+'_I24'].astype(np.int64)
        observed=a[label+'_source_S48']
        expected=(constants['As_q16'].astype(np.int64)@x.reshape(10,-1)).reshape(x.shape)
        assert observed.dtype==np.int64 and np.array_equal(observed,expected),label
        assert np.all(observed>=-(1<<47)) and np.all(observed<(1<<47))
        shift=int(constants['As_exponent'])
        quo,rem=np.divmod(expected,1<<shift)
        state=np.clip(quo+((2*rem>(1<<shift))|((2*rem==(1<<shift))&((quo&1)!=0))),-(1<<23),(1<<23)-1)
        shape=(10,)+(1,)*(x.ndim-1)
        cutoff=constants['source_threshold'].reshape(shape)
        direction=constants['source_direction'].reshape(shape)
        constant=constants['source_constant'].reshape(shape)
        gate=np.where(constant>=0,constant.astype(bool),np.where(direction>0,state>=cutoff,state<=cutoff))
        assert np.array_equal(gate,a[label+'_sn1_gate'])
        fields=('I24','sn1_gate','sn2_gate','updated_I24','proj_gate','continuous_q24')
        for f in fields:
            assert label+'_'+f in a
        rows[label]=dict(source_shape=list(x.shape),S48_values=observed.size,
                         S48_differences=0,source_gate_differences=0,
                         maximum_abs_S48=int(np.abs(observed).max()))
    if previous_first is not None:
        old=read_npz(previous_first)
        fields=[k for k in old if k.startswith(('corner_','interior_')) and
                (np.issubdtype(old[k].dtype,np.integer) or old[k].dtype==np.bool_)]
        assert fields
        checked={}
        for k in fields:
            assert k in a and np.array_equal(a[k],old[k]),('old_first_integer_endpoint',k)
            checked[k]=int(a[k].size)
        assert np.array_equal(a['window_geometry_json'],old['window_geometry_json'])
        rows['old_first_frame']=dict(checked_fields=checked,differences=0,
                                    reference=str(previous_first),exact_geometry=True)
    return rows


def main():
    sys.dont_write_bytecode=True
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path(__file__).resolve().parent)
    args=p.parse_args()
    args.root=args.root.resolve();args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=True)
    op=args.root/'open_fusion_execution'
    algorithm=op/'breadth_20260912/algorithm'
    sys.path.insert(0,str(algorithm))
    from parent_network import ParentNetwork,arrays
    import torch
    assert torch.cuda.is_available()
    manifest=json.loads((args.root/'motion/capture/frames.json').read_text())
    frames=manifest['frames'][:4]
    names=[f['file'] for f in frames]
    assert names==[f'zurich_city_09_a_{i:04d}.npy' for i in range(1,5)]
    assert all(a['timestamp_end_us']==b['timestamp_start_us'] for a,b in zip(frames,frames[1:]))
    parent=algorithm/'matched_training/dense/stage320/deployed_constants.npz'
    first=algorithm/'hardware_exports/dense/000_zurich_city_09_a_0001.npz'
    constants=arrays(parent)
    assert first.exists(), 'Existing matched dense first-frame reference is required'
    report=dict(complete=False,structure='dense',parent=str(parent),frames=[],
                frame_protocol=frames,optimizer_updates=0,independent_inference_frames=True,
                history_used_by_forward=False,old_native_ep34_gate_input=False,
                source_S48='Actual helper.observe accumulator_ranges/As_I_Q24 tensor, before RNE/sat24; gold only for later Machine checks.',
                full_model_endpoint='Original preds.2 coarse head; complete forward only, no 825 run.',
                device=dict(name=torch.cuda.get_device_name(),torch=torch.__version__,
                            free_total_bytes=list(torch.cuda.mem_get_info())),
                parameters='deployed_constants.npz',live_parameters='live_parameters.npz',
                layout='Per frame: corner_/interior_ fields from SmallCapture; source_S48 is T,C,H,W int64, same source geometry as I24/sn1_gate.')
    index=args.output/'index.json';dump(index,report)
    np.savez_compressed(args.output/'deployed_constants.npz',**constants)
    net=ParentNetwork(args)
    full=net.lift/'schedule_compare_same_port/full_chain'
    sys.path.insert(0,str(full));sys.path.insert(0,str(full.parent))
    from fixed_structure import LiteralForward
    from endpoint_observer import SmallCapture
    from evaluate_branch_control import evaluate_axis
    net.install('ordinary');net.helper.restore()
    helper=LiteralForward(net.controller,net.pair.temporal.theta,constants,'dense');net.helper=helper
    latest={};original_statistics=net.math.statistics;original_observe=helper.observe

    class FourCapture(SmallCapture):
        def save_frame(self):
            assert self.arrays is not None
            for label in ('corner','interior'):
                assert label+'_source_S48' in self.arrays
            self.arrays['source_S48_provenance']=np.asarray(report['source_S48'])
            f=frames[self.index]
            for k in ('timestamp_start_us','timestamp_end_us'):
                self.arrays[k]=np.asarray(f[k],np.int64)
            super().save_frame()
            row=dict(self.rows[-1],frame_index=self.index,
                     timestamp_start_us=f['timestamp_start_us'],timestamp_end_us=f['timestamp_end_us'])
            row['checks']=validate_capture(args.output/row['capture'],constants,first if self.index==0 else None)
            report['frames'].append(row);dump(index,report)

    def statistics(x,gamma,beta,eps):
        st=original_statistics(x,gamma,beta,eps);latest['stats']=st.copy();return st

    observer=FourCapture(net.model,net.modules,helper,net.pair.temporal.theta,names,args.output,
                         True,structure='dense',latest=latest,function_label='matched_dense_stage320_four_independent_frames')
    def observe(category,name,value):
        original_observe(category,name,value)
        if category=='accumulator_ranges' and name=='As_I_Q24':
            observer.save_value('source_S48',value,'source')
            for label in ('corner','interior'):
                key=label+'_source_S48';actual=observer.arrays[key]
                assert np.all(np.isfinite(actual)) and np.array_equal(actual,np.rint(actual))
                assert np.all(actual>=-(1<<47)) and np.all(actual<(1<<47))
                observer.arrays[key]=actual.astype(np.int64)
            observer.order.append('source_As_actual_preRNE_S48')

    helper.observe=observe;net.math.statistics=statistics
    np.savez_compressed(args.output/'BN_parameters.npz',gamma=net.bn_gamma,beta=net.bn_beta,eps=np.asarray(net.bn.eps))
    dump(args.output/'deployment_metadata.json',helper.metadata)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    try:
        summary=evaluate_axis(args,net.model,net.current,names,'dense',progress_tag='FOLLOWTHROUGH_CAPTURE')
        rows=json.loads((args.output/'dense_frames.json').read_text())
        old=json.loads((algorithm/'matched_training/dense/stage320/dense_frames.json').read_text())[0]
        assert rows[0]['file']==old['file'] and rows[0]['valid_pixels']==old['valid_pixels']
        assert rows[0]['aee_sum']==old['aee_sum'],('first_frame_AEE_sum',rows[0]['aee_sum'],old['aee_sum'])
        report.update(complete=True,forward_summary=summary,
                      first_frame_AEE_sum_exact=True,peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      wall_seconds=time.monotonic()-started,
                      new_quality_population=False,new_training=False,frames_complete=len(report['frames']))
        assert report['frames_complete']==4
        dump(index,report)
    finally:
        observer.restore();helper.observe=original_observe;net.math.statistics=original_statistics;net.close()
    print('FOUR_FRAME_CAPTURE_COMPLETE',json.dumps(dict(frames=4,output=str(args.output),peak_allocated_bytes=report['peak_allocated_bytes'])),flush=True)


if __name__=='__main__':main()
