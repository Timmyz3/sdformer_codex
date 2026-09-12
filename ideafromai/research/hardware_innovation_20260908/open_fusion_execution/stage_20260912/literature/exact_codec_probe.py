"""Read-only captured FP32 opportunity audit. No model changes, GPU, or EDA.

Reference codecs, not faithful complete EBPC/ZipServ implementations. All byte
counts include per-block mode/width, masks, bases, and byte padding. Statistics
are from the current tensor (optimistic encode-time scan, never a free online
oracle). Sample blocks are actually serialized and decoded bit-exactly.
"""
import json
import math
import struct
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
CAP = Path('/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain/capture')
B = 64


def pack(values, width):
    acc = 0
    for i, v in enumerate(values):
        acc |= int(v) << (width * i)
    return acc.to_bytes((len(values) * width + 7) // 8, 'little')


def unpack(data, width, count):
    acc = int.from_bytes(data, 'little')
    mask = (1 << width) - 1
    return np.array([(acc >> (i * width)) & mask for i in range(count)], dtype=np.uint64)


def exp_encode(words, base):
    e = (words >> 23) & 255
    good = (e >= base) & (e < base + 7)
    code = np.where(good, e - base + 1, 0)
    # Exact all 24 sign/mantissa bits; escaping words retain all 32 bits.
    sm = (words[good] & 0x7fffff) | ((words[good] >> 31) << 23)
    return pack(code, 3) + pack(sm, 24) + words[~good].astype('<u4').tobytes()


def exp_decode(data, count, base):
    nc = (count * 3 + 7) // 8
    code = unpack(data[:nc], 3, count)
    good = code > 0
    ng = int(good.sum())
    sm = unpack(data[nc:nc+ng*3], 24, ng)
    out = np.empty(count, dtype=np.uint32)
    out[good] = ((sm & 0x7fffff) | ((sm >> 23) << 31) | ((code[good] + base - 1) << 23)).astype(np.uint32)
    out[~good] = np.frombuffer(data[nc+ng*3:], dtype='<u4')
    return out


def encode(block, default, base, mode):
    if mode == 0:
        return bytes([0]) + block.astype('<u4').tobytes()
    if mode == 1:
        neq = block != default
        return bytes([1]) + pack(neq, 1) + block[neq].astype('<u4').tobytes()
    if mode == 2:
        xor = block[1:] ^ block[0]
        width = int(xor.max(initial=0)).bit_length()
        return bytes([2, width]) + struct.pack('<I', int(block[0])) + pack(xor, width)
    if mode == 3:
        return bytes([3]) + exp_encode(block, base)
    if mode == 4:
        neq = block != default
        return bytes([4]) + pack(neq, 1) + exp_encode(block[neq], base)
    if mode == 5:
        delta = block[1:].astype(np.int64) - int(block[0])
        zz = np.where(delta >= 0, 2*delta, -2*delta-1).astype(np.uint64)
        width = int(zz.max(initial=0)).bit_length()
        return bytes([5,width])+struct.pack('<I',int(block[0]))+pack(zz,width)
    raise ValueError(mode)


def decode(data, default, base):
    mode = data[0]
    if mode == 0:
        return np.frombuffer(data[1:], dtype='<u4')
    if mode in [2,5]:
        width = data[1]
        first = struct.unpack('<I', data[2:6])[0]
        v = unpack(data[6:], width, B-1)
        if mode == 2:
            v = v ^ first
        else:
            d = (v >> 1).astype(np.int64)
            d = np.where(v & 1, -d-1, d)
            v = d + first
        return np.concatenate([np.array([first], dtype=np.uint32),v.astype(np.uint32)])
    if mode == 3:
        return exp_decode(data[1:], B, base)
    neq = unpack(data[1:9], 1, B).astype(bool)
    out = np.full(B, default, dtype=np.uint32)
    out[neq] = np.frombuffer(data[9:], dtype='<u4') if mode == 1 else exp_decode(data[9:], int(neq.sum()), base)
    return out


def probe(x, label):
    assert x.dtype == np.float32 and x.ndim == 4
    # Native capture T,C,H,W; contiguous blocks within each spatial plane.
    t, c, h, w = x.shape
    words = np.ascontiguousarray(x).view(np.uint32)
    results = {'label': label, 'shape': list(x.shape), 'dtype': str(x.dtype), 'elements': int(x.size), 'raw_bytes': int(x.nbytes), 'full_xor_representation_roundtrip': True}
    sums = np.zeros(7, dtype=np.int64)
    selected = np.zeros(6, dtype=np.int64)
    tested = 0
    mismatches = 0
    sample_bytes = 0
    exponent_escape = 0
    default_words = 0
    entropy = []
    # Six views: raw; fill/raw; xor/raw; exponent/raw; fill/exponent/raw;
    # universal selector of all four ordinary codecs (a strong simple control).
    for ci in range(c):
        flat = words[:,ci].reshape(-1)
        pad = (-flat.size) % B
        a = np.pad(flat,(0,pad),constant_values=0).reshape(-1,B)
        values, counts = np.unique(a, return_counts=True)
        default = int(values[counts.argmax()])
        exp = (a >> 23) & 255
        eh = np.bincount(exp.ravel(), minlength=256)
        base = int(np.convolve(eh, np.ones(7, dtype=np.int64), 'valid').argmax())
        good = (exp >= base) & (exp < base+7)
        neq = a != default
        default_words += int((~neq).sum())
        exponent_escape += int((~good).sum())
        p = eh[eh>0]/float(a.size)
        entropy.append(float(-(p*np.log2(p)).sum()))
        n = a.shape[0]
        raw = np.full(n, 1+B*4, dtype=np.int64)
        fill = 1+8+neq.sum(axis=1)*4
        xo = a[:,1:] ^ a[:,:1]
        xmax = xo.max(axis=1).astype(np.float64)
        width = np.where(xmax > 0, np.floor(np.log2(np.maximum(xmax, 1)))+1, 0).astype(np.int64)
        xor = 2+4+((B-1)*width+7)//8
        d = a[:,1:].astype(np.int64)-a[:,:1].astype(np.int64)
        zz = np.where(d>=0,2*d,-2*d-1)
        zm = zz.max(axis=1).astype(np.float64)
        zw = np.where(zm>0,np.floor(np.log2(np.maximum(zm,1)))+1,0).astype(np.int64)
        delta = 2+4+((B-1)*zw+7)//8
        ex = 1+(B*3+7)//8+good.sum(axis=1)*3+(~good).sum(axis=1)*4
        remaining = neq.sum(axis=1)
        fg = (neq & good).sum(axis=1)
        fx = 1+8+(remaining*3+7)//8+fg*3+(remaining-fg)*4
        costs = np.stack([raw,fill,xor,ex,fx,delta])
        sums += np.array([raw.sum(),np.minimum(raw,fill).sum(),np.minimum(raw,xor).sum(),np.minimum(raw,ex).sum(),costs[[0,1,3,4]].min(axis=0).sum(),costs.min(axis=0).sum(),np.minimum(raw,delta).sum()])
        modes = costs.argmin(axis=0)
        selected += np.bincount(modes, minlength=6)
        # Actual byte serialization across a deterministic uniform channel sample.
        ids = np.unique(np.linspace(0,n-1,min(n,8),dtype=np.int64))
        for bi in ids:
            block = a[bi]
            for mode in range(6):
                blob = encode(block, default, base, mode)
                rebuilt = decode(blob, default, base)
                assert len(blob) == int(costs[mode,bi]), (mode,len(blob),costs[mode,bi])
                mismatches += int(np.count_nonzero(rebuilt != block))
                tested += 1
                sample_bytes += len(blob)
        # Full representation checks for common transforms, including NaN payloads.
        assert np.array_equal(np.column_stack([a[:,0], (xo ^ a[:,:1])]), a)
    # Per-channel default word (4B), exponent base (1B), 16B tensor header.
    metadata = np.array([16,16+4*c,16,16+c,16+5*c,16+5*c,16])
    names = ['raw_block_framed','default_fill_or_raw','xor_base_or_raw','exponent_window_or_raw','default_plus_exponent_or_raw','ordinary_all_modes','word_signed_delta_or_raw']
    results['codec_bytes'] = dict(zip(names, map(int,sums+metadata)))
    results['ratio_encoded_over_raw'] = {k:v/x.nbytes for k,v in results['codec_bytes'].items()}
    results.update({'padded_values':c*((-t*h*w)%B),'default_fraction_including_padding':default_words/(x.size+c*((-t*h*w)%B)),'exponent_escape_fraction_including_padding':exponent_escape/(x.size+c*((-t*h*w)%B)),'mean_channel_exponent_entropy_bits':float(np.mean(entropy)),'selected_blocks':dict(zip(['raw','default','xor','exponent','default_exponent','signed_word_delta'],map(int,selected))),'actual_serialized_blocks_tested':tested,'actual_serialized_sample_bytes':sample_bytes,'bit_mismatches':mismatches})
    return results


def main():
    start=time.time()
    rows=[]
    keys=['proj_bn_full_input_fp32','proj_bn_full_output_fp32','corner_preview_Z_shared','interior_preview_Z_shared','corner_preview_BN1_Y','interior_preview_BN1_Y']
    for arm in ['ordinary','lifting_raw']:
        p=CAP/arm/'000_zurich_city_09_a_0001.npz'
        with np.load(p,allow_pickle=False) as f:
            for key in keys:
                r=probe(f[key],arm+'/'+key)
                r['source_path']=str(p)
                rows.append(r)
                print(r['label'],r['ratio_encoded_over_raw'],flush=True)
    out={'scope':'1 frame, 2 arms, full projection BN input/output + corner/interior preview; exact FP32 bits; CPU codec sizing and sample serializer only','limitations':['No GPU, RTL, PPA, full consumer timing, or AEE rerun.','Same-frame per-channel statistics require scan/histogram state not charged as execution cycles.','Reference modes are incomplete EBPC/ZipServ/Shannonic implementations; universal selection is a simple strong control, not a novelty claim.','Compression preserves source words but does not authorize reordering floating-point arithmetic or changing BN.','FP32 sign and all 23 mantissa bits retained; this is not BF16 conversion.'],'python':sys.version,'numpy':np.__version__,'block_values':B,'elapsed_cpu_process_wall_s':time.time()-start,'rows':rows}
    (HERE/'exact_codec_results.json').write_text(json.dumps(out,indent=2)+'\n')


if __name__ == '__main__': main()
