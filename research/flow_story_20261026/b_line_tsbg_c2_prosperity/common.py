"""Shared loaders for the B-line TSBG/C2 vs official Prosperity study (CPU only)."""
import os, json, struct, zlib, hashlib
from pathlib import Path
import numpy as np

HOME = Path.home()
OUT = HOME / 'work/sdformer_codex/research/flow_story_20261026/b_line_tsbg_c2_prosperity'
SNAP = HOME / 'work/sdformer_codex/research/flow_story_20261026/c1_prosperity_denominator/official_sim_snapshot'
HW = HOME / 'work/sdformer_codex/SDformer/hw_autoresearch_nts07'
CAP = HW / 'results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901'
FRAME_MAGIC = b'M1558F01'
FRAME_HEADER = struct.Struct('<8sHH11I')   # identical to capture_m1558 source
FC_FRAMES_SHA = 'dceb6c0c80b9c5898d10b4ad813fbcd7683fa80191b54b78eadaadda04a818b1'
INDEX = OUT / 'fc_frame_index.json'


def layers():
    L = json.load(open(CAP / 'layers.json'))['layers']
    return {int(l['layer_id']): l for l in L if l['target'] in ('FC1', 'FC2')}


def build_index():
    idx = {}
    with open(CAP / 'fc_frames.bin', 'rb') as f:
        while True:
            pos = f.tell()
            pre = f.read(FRAME_HEADER.size)
            if not pre:
                break
            v = FRAME_HEADER.unpack(pre)
            (magic, ver, hsz, lid, sid, fidx, fstart, ntok, ch, bitrow, nnz, rawb, compb, crc) = v
            assert magic == FRAME_MAGIC and hsz == FRAME_HEADER.size
            idx.setdefault(f'{sid}:{lid}', []).append([pos, fidx, fstart, ntok, ch, bitrow, nnz, rawb, compb, crc])
            f.seek(compb, 1)
    for k in idx:
        idx[k].sort(key=lambda r: r[2])
    INDEX.write_text(json.dumps(idx))
    return idx


def get_index():
    if not INDEX.exists():
        return build_index()
    return json.loads(INDEX.read_text())


def load_codes_support(sample, layer, idx=None):
    """Return support bool [tokens, C] in canonical capture token order, plus counters."""
    idx = idx or get_index()
    L = layers()[layer]
    fr = idx[f'{sample}:{layer}']
    tokens = int(L['tokens_per_call']); C = int(L['input_channels'])
    S = np.zeros((tokens, C), dtype=bool)
    neg = nonunit = nnz_tot = 0
    expect = 0
    with open(CAP / 'fc_frames.bin', 'rb') as f:
        for (pos, fidx, fstart, ntok, ch, bitrow, nnz, rawb, compb, crc) in fr:
            assert fstart == expect and ch == C
            f.seek(pos + FRAME_HEADER.size)
            raw = zlib.decompress(f.read(compb))
            assert len(raw) == rawb and (zlib.crc32(raw) & 0xffffffff) == crc
            mb = ntok * bitrow
            mats = []
            for i in range(3):
                p = np.frombuffer(raw[i * mb:(i + 1) * mb], dtype=np.uint8).reshape(ntok, bitrow)
                mats.append(np.unpackbits(p, axis=1, bitorder='little')[:, :C].astype(bool))
            sup, sgn, nun = mats
            assert int(sup.sum()) == nnz
            S[fstart:fstart + ntok] = sup
            neg += int(sgn.sum()); nonunit += int(nun.sum()); nnz_tot += nnz
            expect += ntok
    assert expect == tokens
    shp = L['input_shape']   # [T,1,H,W,C]
    T = int(shp[0]); HWn = tokens // T
    return S, dict(tokens=tokens, C=C, T=T, HW=HWn, nnz=nnz_tot, negative=neg, nonunit=nonunit,
                   N=int(L['output_channels']), name=L['module_name'], target=L['target'])


def import_official():
    import sys, types
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.modules.setdefault('prosparsity_engine', types.ModuleType('prosparsity_engine'))
    if str(SNAP) not in sys.path:
        sys.path.insert(0, str(SNAP))
    import torch
    torch.set_num_threads(1)
    import simulator as simmod
    from networks import FC
    from accelerator import Accelerator
    return torch, simmod, FC, Accelerator
