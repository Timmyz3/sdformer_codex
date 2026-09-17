"""Python 3.12 NumPy-only replay from the original torch-save archive.

No Torch import/install, GPU, or training. The declared output is regenerated.
The shared read_torch already accepts BytesIO via zipfile; it is unchanged.
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import argparse
import io
import json
import sys
import tarfile
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(BASE / 'bn_state'))
from support_service_model import read_torch


def nearest(gate, D):
    s = gate.reshape(-1, 6, 16).astype(np.int16)
    d = D.astype(np.int16)
    distance = s.sum(-1)[..., None] + d.sum(-1)[None] - 2*np.einsum('ngc,gkc->ngk', s, d)
    idx = distance.argmin(-1)
    return (d[np.arange(6)[None], idx].reshape(gate.shape).astype(np.uint8),
            idx.reshape(gate.shape[0], gate.shape[1], 6).astype(np.uint8), distance.min(-1))


def quant(x, f, width):
    q = np.rint(np.asarray(x, dtype=np.float64) * 2**f).astype(np.int64)
    assert q.min() >= -(1 << (width-1)) and q.max() < (1 << (width-1))
    return q


def opportunity(g, D):
    _, _, dist = nearest(g, D)
    pc = g.reshape(-1, 6, 16).sum(-1)
    exact = dist == 0
    useful = exact & (pc >= 2)
    return dict(groups=int(pc.size), active_bits=int(pc.sum()), exact_groups=int(exact.sum()),
                useful_exact_groups=int(useful.sum()),
                exact_saved_adds_unpriced=int(np.where(useful, pc-1, 0).sum()),
                projection_hamming_bits=int(dist.sum()))


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12 for this supported regeneration path.')
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, default=BASE/'algorithm/support_training.tar.gz')
    ap.add_argument('--student-root', type=Path, default=BASE/'algorithm/support_training')
    ap.add_argument('--output', type=Path, default=HERE/'source_cases.npz')
    ap.add_argument('--compare-to', type=Path,
                    help='Optional existing NPZ for a field comparison; not an input dependency.')
    ap.add_argument('--compare-statistics', type=Path,
                    help='Optional historical source_statistics.json comparison.')
    args = ap.parse_args()
    with tarfile.open(args.archive, 'r:gz') as tar:
        cache = read_torch(io.BytesIO(tar.extractfile('support_training/teacher_cache.pt').read()))
    p = read_torch(args.student_root/'forced_code_parameters.pt')
    run = json.loads((args.student_root/'run.json').read_text())
    old = json.loads(args.compare_statistics.read_text()) if args.compare_statistics else None
    original = None
    if args.compare_to:
        with np.load(args.compare_to) as z:
            original = {k:np.array(z[k],copy=True) for k in z.files}
    assert [r['file'] for r in cache] == run['train_files'] and len(cache) == 32
    A, bias, center, theta = [np.array(p[k], copy=True) for k in ['A','bias','center','theta']]
    D = np.asarray(p['dictionary'], dtype=np.uint8)
    assert np.array_equal(D, np.load(args.student_root/'dictionary.npy'))
    Aq = quant(A, 12, 16).astype(np.int16)
    tr = (theta.astype(np.float64)+center.astype(np.float64)-bias.astype(np.float64)).reshape(10)
    threshold = quant(tr, 28, 48)
    records, cases = [], []
    prefix_max = 0
    for i, row in enumerate(cache):
        X = np.asarray(row['x'], dtype=np.float32)
        assert X.shape == (10,512,96)
        Xq = quant(X,16,24).astype(np.int32)
        h = (A @ X.reshape(10,-1) + bias) - center
        raw = (h >= theta).reshape(X.shape).astype(np.uint8)
        margin = (h-theta).reshape(X.shape)
        proj, idx, _ = nearest(raw,D)
        accum = np.zeros(X.shape,dtype=np.int64)
        for s in range(10):
            accum += Aq[:,s,None,None].astype(np.int64)*Xq[s].astype(np.int64)[None]
            prefix_max = max(prefix_max,int(np.abs(accum).max()))
        assert np.array_equal(accum,np.einsum('ts,spc->tpc',Aq.astype(np.int64),Xq.astype(np.int64)))
        imargin=accum-threshold[:,None,None]
        raw_i=(imargin>=0).astype(np.uint8)
        proj_i,idx_i,_=nearest(raw_i,D)
        result=dict(file=row['file'],raw_gate_differences=int((raw!=raw_i).sum()),
            projected_gate_differences=int((proj!=proj_i).sum()),
            projected_code_differences=int((idx!=idx_i).sum()),
            teacher_gate_differences=int((raw!=row['gate']).sum()),
            raw_fp32=opportunity(raw,D),raw_integer=opportunity(raw_i,D),
            projected_fp32=opportunity(proj,D),projected_integer=opportunity(proj_i,D))
        if old is not None:
            for key,value in result.items():
                assert value == old['frames'][i][key], (i,key,value,old['frames'][i][key])
        records.append(result)
        if i<2:
            cases.append(dict(X_fp32=X[:,:32],X_q16=Xq[:,:32],raw_g_fp32=raw[:,:32],raw_g_int=raw_i[:,:32],
                projected_g_fp32=proj[:,:32],projected_g_int=proj_i[:,:32],fp32_margin=margin[:,:32],
                int_accum_q28=accum[:,:32],int_margin_q28=imargin[:,:32],
                code_index_fp32=idx[:,:32].transpose(1,0,2),code_index_int=idx_i[:,:32].transpose(1,0,2)))
        print('NUMPY312',i,row['file'],result['raw_gate_differences'],flush=True)
    data={k:np.stack([c[k] for c in cases]) for k in cases[0]}
    data.update(A_fp32=A,A_q12=Aq,bias_fp32=bias,center_fp32=center,theta_fp32=theta,
        threshold_real=tr,threshold_q28=threshold,D=D,frame_file=np.array([r['file'] for r in cache[:2]]),
        sample_index=np.arange(32,dtype=np.int32),
        reconstructed_spatial_index=np.rint(np.linspace(0,19199,512,dtype=np.float32)).astype(np.int64)[:32],
        spatial_index_status=np.array('reconstructed from fixed 19200-position source shape; not stored in cache'),
        source_boundary=np.array('cached pre-source FP32 X from 32 training frames; forced student source parameters'),
        dut_input_fields=np.array(['X_q16','A_q12','threshold_q28','D']),
        oracle_fields=np.array(['X_fp32','raw_g_fp32','raw_g_int','projected_g_fp32','projected_g_int',
            'code_index_fp32','code_index_int','fp32_margin','int_accum_q28','int_margin_q28']))
    exact_fields=[]
    if original is not None:
        for key,value in data.items():
            assert np.array_equal(value,original[key]),key
            exact_fields.append(key)
    assert prefix_max<2**47
    if old is not None:
        assert prefix_max==old['observed']['integer_prefix_abs_max']
    assert 'torch' not in sys.modules
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**data)
    reread=np.load(args.output)
    for k,v in data.items():assert np.array_equal(v,reread[k]),k
    report=dict(python=sys.version.split()[0],numpy=np.__version__,torch_imported=False,
        archive=str(args.archive),cache_member='support_training/teacher_cache.pt',
        parameter_source=str(args.student_root/'forced_code_parameters.pt'),
        no_training=True,no_clipping=True,no_new_AEE=True,
        historical_gpu_replay=dict(python='3.10',interpreter='/opt/anaconda3/envs/pytorch310/bin/python',
            source='prepare_sources_gpu_legacy.py',log='prepare_sources.log',
            note='Original run violated requested Python 3.12; recorded, not relabelled.'),
        independent_numpy_fp32_formula='(A @ X + bias) - center >= theta',
        original_fixture_exact_fields=exact_fields,
        comparison_input=str(args.compare_to) if args.compare_to else None,
        comparison_statistics=str(args.compare_statistics) if args.compare_statistics else None,
        fp32_margin_nonidentical_values=int(np.count_nonzero(data['fp32_margin']!=original['fp32_margin'])) if original is not None else None,
        fp32_margin_abs_difference_max=float(np.abs(data['fp32_margin']-original['fp32_margin']).max()) if original is not None else None,
        full_cache_per_frame_counts_and_opportunities_match_original=True if old is not None else None,
        note='Full historical raw-gate arrays were not retained; full-cache agreement is per-frame counts/opportunities, selected P32 gates compare elementwise.',
        integer_prefix_abs_max=prefix_max,
        totals={k:sum(r[k] for r in records) for k in ['raw_gate_differences','projected_gate_differences','projected_code_differences','teacher_gate_differences']},
        frames=records)
    args.output.with_name('source_statistics_numpy312.json').write_text(json.dumps(report,indent=2)+'\n')
    print('PASS Python',report['python'],'NumPy-only; 32 frames; output',str(args.output),
          'optional exact comparison fields',len(exact_fields),flush=True)


if __name__=='__main__':main()
