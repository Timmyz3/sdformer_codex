"""Read-only CPU reconstruction of four already audited archived event tensors.

Run on the archive host. No files, splits, checkpoints, or existing arrays change.
The normalized-float64-time arm is a diagnostic counterfactual, not a new dataset.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
import hashlib, importlib.util, json, pathlib, sys, time
sys.dont_write_bytecode=True
import numpy as np
import torch
torch.set_num_threads(1)

ROOT=pathlib.Path('/root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC')
SOURCE=pathlib.Path('/root/private_data/work/sdformer_codex/SDformer')
path=SOURCE/'tools/prepare_dsec_single_sequence.py'
spec=importlib.util.spec_from_file_location('archived_preprocessor',path)
prep=importlib.util.module_from_spec(spec);spec.loader.exec_module(prep)
import h5py

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def compare(a,b):
    diff=np.abs(a.astype(np.float64)-b.astype(np.float64))
    return {'array_equal':bool(np.array_equal(a,b)),'different_elements':int(np.count_nonzero(a!=b)),
            'max_absolute_difference':float(diff.max()),'mean_absolute_difference':float(diff.mean()),
            'relative_L1':float(diff.sum()/max(np.abs(b.astype(np.float64)).sum(),1e-30))}

def main():
    records=[];started=time.monotonic()
    for seq,idx in [('zurich_city_01_a',50),('zurich_city_11_c',411),('thun_00_a',22),('zurich_city_11_b',484)]:
        name=seq+'_'+format(idx,'04d')+'.npy'
        stamps=ROOT/'train_optical_flow'/seq/'flow/forward_timestamps.txt'
        beg,end=np.loadtxt(stamps,delimiter=',',dtype=np.int64)[idx-1]
        eventdir=ROOT/'train_events'/seq/'events/left'
        archived=ROOT/'saved_flow_data/event_tensors/10bins/left'/seq/name
        with h5py.File(eventdir/'events.h5','r') as f,h5py.File(eventdir/'rectify_map.h5','r') as r:
            slicer=prep.EventSlicer(f);slicer.t_offset=np.int64(slicer.t_offset)
            rectify=r['rectify_map'][()]
            cpu=prep.make_voxel_chunk(slicer,rectify,prep.VoxelGrid((10,480,640)),int(beg),int(end),torch.device('cpu'))
            old=np.load(archived,allow_pickle=False)
            ev=slicer.get_events(int(beg),int(end));xr,yr=prep.rectify_events(ev['x'],ev['y'],rectify).T
            valid=(xr>=0)&(xr<640)&(yr>=0)&(yr<480)
            ts=ev['t'][valid].astype(np.float64)
            normalized=(ts-ts[0])/(ts[-1]-ts[0])
            rounded=ts.astype(np.float32);normalized_old=(rounded-rounded[0])/(rounded[-1]-rounded[0])
            precise_events={
                'p':torch.from_numpy(ev['p'][valid].astype(np.float32)),
                't':torch.from_numpy(normalized.astype(np.float32)),
                'x':torch.from_numpy(xr[valid].astype(np.float32)),
                'y':torch.from_numpy(yr[valid].astype(np.float32))}
            precise=prep.VoxelGrid((10,480,640)).convert_CHW(precise_events).numpy()
            records.append({'target_id':name[:-4],'timestamp_pair_us':[int(beg),int(end)],
                'archived_npy_sha256':sha(archived),'rectify_sha256':sha(eventdir/'rectify_map.h5'),
                'timestamps_sha256':sha(stamps),'raw_events_in_interval':int(ev['t'].size),
                'in_image_events':int(valid.sum()),'archived_generator_cpu_vs_archived_npy':compare(cpu,old),
                'relative_time_before_float32_diagnostic_vs_archived_npy':compare(precise,old),
                'absolute_timestamp_float32_ulp_us':float(np.spacing(rounded[0])),
                'normalized_time_max_rounding_difference':float(np.max(np.abs(normalized-normalized_old))),
                'archived_tensor_finite':bool(np.isfinite(old).all())})
    print(json.dumps({'scope':'four-sample read-only CPU reconstruction, not whole-dataset proof or AEE',
        'torch':str(torch.__version__),'numpy':np.__version__,'cuda_initialized':torch.cuda.is_initialized(),
        'helper_path':str(path),'helper_sha256':sha(path),
        'event_representation_sha256':sha(SOURCE/'third_party/SDformerFlow/DSEC_dataloader/event_representations.py'),
        'counterfactual':'Subtract/normalize timestamps in float64 before float32 conversion; original arrays unchanged',
        'seconds':time.monotonic()-started,'records':records},indent=2))

if __name__=='__main__':main()
