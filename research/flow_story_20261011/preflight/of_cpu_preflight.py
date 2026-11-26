#!/usr/bin/env python3
"""Read-only official-code provenance, split, dense-op and tile dependency audit.

The NumPy toy validates nearest-resize/valid-convolution indexing, NOT OF accuracy.
No PyTorch, checkpoints, CUDA, training, or modifications to source repositories.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import numpy as np

HERE = Path(__file__).resolve().parent
OFF = HERE / 'of_official_source'
SD = Path('/home/zhumd/work/sdformer_codex/SDformer')


def split(path, target_column):
    rows = [r for r in csv.reader(path.open()) if r]
    targets = [r[target_column] for r in rows]
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'rows': len(rows), 'unique_targets': len(set(targets)),
            'sequence_counts': dict(Counter(x.rsplit('_', 1)[0] for x in targets))}, set(targets), rows


def nearest_valid_preimage(lo, hi, src_n, out_n, kernel=7):
    """Exclusive output box -> inclusive/exclusive source interval, PyTorch nearest rule."""
    return (lo * src_n // (out_n + kernel - 1),
            (hi + kernel - 2) * src_n // (out_n + kernel - 1) + 1)


def deps(box, source_hw, output_hw):
    return (*nearest_valid_preimage(box[0], box[1], source_hw[0], output_hw[0]),
            *nearest_valid_preimage(box[2], box[3], source_hw[1], output_hw[1]))


def separable_box(x_roi, source_hw, roi_origin, output_hw, box, dw, pw, spike=False):
    y0, y1, x0, x1 = box
    ys = np.arange(y0, y1 + 6) * source_hw[0] // (output_hw[0] + 6)
    xs = np.arange(x0, x1 + 6) * source_hw[1] // (output_hw[1] + 6)
    patch = x_roi[ys[:, None] - roi_origin[0], xs[None, :] - roi_origin[1], :]
    depthwise = np.zeros((y1-y0, x1-x0, patch.shape[-1]))
    for ky in range(7):
        for kx in range(7):
            depthwise += patch[ky:ky+y1-y0, kx:kx+x1-x0] * dw[:, ky, kx]
    out = depthwise @ pw
    return (out >= 0).astype(float) if spike else out


def toy_tail():
    rng = np.random.default_rng(11)
    coarse_hw, full_hw = (12, 16), (24, 32)
    coarse, bottom = rng.normal(size=(*coarse_hw, 2)), rng.normal(size=(*full_hw, 2))
    dw1, pw1 = rng.normal(size=(2, 7, 7)), rng.normal(size=(2, 3))
    dwp, pwp = rng.normal(size=(5, 7, 7)), rng.normal(size=(5, 2))
    full = (0, 24, 0, 32)
    dec = separable_box(coarse, coarse_hw, (0, 0), full_hw, full, dw1, pw1, True)
    reference = separable_box(np.concatenate((dec, bottom), -1), full_hw, (0, 0), full_hw, full, dwp, pwp)
    boxes = [(0, 8, 0, 8), (8, 16, 8, 24), (16, 24, 24, 32), full]
    checks = []
    for box in boxes:
        db = deps(box, full_hw, full_hw)
        dec_roi = separable_box(coarse, coarse_hw, (0, 0), full_hw, db, dw1, pw1, True)
        z = np.concatenate((dec_roi, bottom[db[0]:db[1], db[2]:db[3]]), -1)
        got = separable_box(z, full_hw, (db[0], db[2]), full_hw, box, dwp, pwp)
        ref = reference[box[0]:box[1], box[2]:box[3]]
        err = float(np.max(np.abs(ref - got)))
        assert err < 1e-10
        checks.append({'output_box': box, 'decoder_required_box': db, 'max_abs_error': err})
    return {'scope': 'Independent NumPy toy, official network NOT executed; geometry only', 'checks': checks}


def dense_ops():
    stages = []
    def add(name, mac, shape): stages.append({'module': name, 'dense_mac': mac, 'output_shape_cthw': shape})
    h, w, t, c = 480, 640, 21, 2
    t -= 4
    add('bottom', t*h*w*2*32*5*7*7, [32,t,h,w]); c=32
    for i, outc in enumerate([64,128,256,512], 1):
        t -= 4
        add(f'conv{i}.dw_pw_before_pool', t*h*w*(c*5*7*7+c*outc), [outc,t,h//2,w//2])
        h//=2; w//=2; c=outc
    add('res1.two_dw_pw', h*w*2*(512*7*7+512*512), [512,1,h,w])
    for name, inc, outc, oh, ow in [('deconv4',512,256,60,80),('deconv3',512,128,120,160),('deconv2',256,64,240,320),('deconv1',128,32,480,640)]:
        add(name, oh*ow*(inc*7*7+inc*outc), [outc,1,oh,ow])
    for name, inc in [('pred4',512),('pred3',256),('pred2',128),('pred1',64)]:
        add(name,480*640*(inc*7*7+inc*2),[2,1,480,640])
    total=sum(x['dense_mac'] for x in stages)
    tail=sum(x['dense_mac'] for x in stages if x['module'] in ['deconv1','pred1'])
    wide_tail=sum(x['dense_mac'] for x in stages if x['module'] in ['deconv3','deconv2','deconv1','pred3','pred2','pred1'])
    residual=next(x['dense_mac'] for x in stages if x['module']=='res1.two_dw_pw')
    return {'scope': 'Unweighted dense MAC opportunity only; NOT SOP, PE cycles, energy, or measured speedup. Excludes pooling/IF/resize/control/memory.',
            'stages': stages,'total_dense_mac':total,'final_tail_dense_mac':tail,
            'final_tail_fraction':tail/total,'wide_tail_dense_mac':wide_tail,
            'wide_tail_fraction':wide_tail/total,'residual_fraction':residual/total}


def main():
    seq=OFF/'data/dataset/saved_flow_data/sequence_lists'
    a, train, _=split(seq/'train_split_doubleseq.csv',1)
    b, valid, vr=split(seq/'valid_split_doubleseq.csv',1)
    c, local, _=split(SD/'data/Datasets/DSEC/saved_flow_data/sequence_lists/valid_split_seq.csv',0)
    selected=(192,224,256,288)
    decoder=deps(selected,(480,640),(480,640))
    earlier=deps(decoder,(240,320),(480,640))
    r={'official_repository':'https://github.com/J-Cuadrado/OF_EV_SNN',
       'official_commit':subprocess.check_output(['git','-C',str(OFF),'rev-parse','HEAD'],universal_newlines=True).strip(),
       'splits':[a,b,c], 'comparison':{'official_train_valid_target_overlap':len(train&valid),
        'local_targets_in_official_valid':len(local&valid),'local_targets_in_official_train':len(local&train),
        'local_targets_outside_official_target_sets':len(local-(train|valid)),
        'local_outside_examples':sorted(local-(train|valid))[:30],
        'official_valid_targets_missing_locally':len(valid-local),
        'official_pairs_adjacent_same_sequence':all(x[0].rsplit('_',1)[0]==x[1].rsplit('_',1)[0] and int(x[1].rsplit('_',1)[1].split('.')[0])-int(x[0].rsplit('_',1)[1].split('.')[0])==1 for x in vr)},
       'geometry_example':{'final_output_box_y0_y1_x0_x1':selected,'deconv1_required_box':decoder,'deconv1_input_required_box':earlier},
       'dense_opportunity':dense_ops(),'toy_dependency_check':toy_tail()}
    out=HERE/'of_cpu_preflight_results.json'
    out.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({'output':str(out),'comparison':r['comparison'],'geometry_example':r['geometry_example'],
       'final_tail_fraction_dense':r['dense_opportunity']['final_tail_fraction'],
       'residual_fraction_dense':r['dense_opportunity']['residual_fraction'],
       'toy_max_errors':[x['max_abs_error'] for x in r['toy_dependency_check']['checks']]},indent=2))


if __name__=='__main__': main()
