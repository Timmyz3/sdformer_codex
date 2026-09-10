#!/usr/bin/env python3.12
"""Read first predeclared sample, all 12 FC1 domains. Statistics only, no EDA."""
from pathlib import Path
import hashlib
import json
import struct
import zlib
import numpy as np

BASE = Path(__file__).resolve().parents[1]
CAP = Path('/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_binary_capture_s40_r1_20260901')
EXPECTED = {'layers.json':'bd40c213f075ea3198f7145d25e9c96988701f46d5572c1e40d36e008feab08a',
            'fc_frames.bin':'dceb6c0c80b9c5898d10b4ad813fbcd7683fa80191b54b78eadaadda04a818b1',
            'sample_order.json':'d4f1f6e140b531b972d53b48aa64e5f0aa5497b79d460616a0b3f89139a4f773'}
HEADER = struct.Struct('<8sHH11I')
POP = np.array([v.bit_count() for v in range(256)],dtype=np.uint8)


def main():
    output=BASE/'records/v2_sample0_moments.json'
    if output.exists():
        raise RuntimeError('Output already exists; preserve this research result')
    for name,digest in EXPECTED.items():
        d=hashlib.sha256()
        with (CAP/name).open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):d.update(block)
        assert d.hexdigest()==digest, name
    layers={r['layer_id']:r for r in json.loads((CAP/'layers.json').read_text())['layers']}
    selected={k:v for k,v in layers.items() if v['target']=='FC1'}
    assert len(selected)==12
    supports={k:[] for k in selected}; counts={k:0 for k in selected}; frame_counts={k:0 for k in selected}
    with (CAP/'fc_frames.bin').open('rb') as f:
        while True:
            hdr=f.read(HEADER.size)
            assert len(hdr)==HEADER.size
            magic,version,hs,lid,sid,fi,start,n,C,br,nnz,rb,cb,crc=HEADER.unpack(hdr)
            assert magic==b'M1558F01' and version==1 and hs==HEADER.size
            if sid>0:break
            assert sid==0 and lid in layers
            if lid not in selected:
                f.seek(cb,1);continue
            spec=selected[lid]
            assert C==spec['input_channels'] and br==(C+7)//8 and 0<n<=4096
            assert fi==frame_counts[lid] and start==counts[lid]
            dec=zlib.decompressobj(); raw=dec.decompress(f.read(cb))+dec.flush()
            assert dec.eof and not dec.unused_data and not dec.unconsumed_tail
            assert len(raw)==rb and zlib.crc32(raw)&0xffffffff==crc
            mb=n*br; assert rb==3*mb+2*n+nnz
            bits=np.unpackbits(np.frombuffer(raw[:mb],dtype=np.uint8).reshape(n,br),axis=1,bitorder='little')
            assert not bits[:,C:].any()
            S=bits[:,:C]
            assert not any(raw[mb:3*mb]), 'nonbinary/nonpositive source'
            row_counts=np.frombuffer(raw[3*mb:3*mb+2*n],dtype='<u2')
            assert np.array_equal(row_counts,S.sum(1)) and int(row_counts.sum())==nnz
            assert np.all(np.frombuffer(raw[3*mb+2*n:],dtype=np.int8)==1)
            supports[lid].append(S);counts[lid]+=n;frame_counts[lid]+=1
    rows=[]
    for lid,spec in selected.items():
        N,C,H=spec['tokens_per_call'],spec['input_channels'],spec['output_channels']
        assert counts[lid]==N
        S=np.concatenate(supports[lid],axis=0)
        assert S.shape==(N,C)
        d=S.sum(1,dtype=np.int64);k=S.sum(0,dtype=np.int64)
        columns=np.packbits(S,axis=0,bitorder='little').T.copy()
        G=np.zeros((C,C),dtype=np.int64)
        for i in range(C):
            G[i,i:]=POP[np.bitwise_and(columns[i],columns[i:])].sum(1,dtype=np.int64)
            G[i:,i]=G[i,i:]
        assert np.array_equal(G.diagonal(),k)
        assert np.all(G<=np.minimum(k[:,None],k[None,:]))
        tri=G[np.triu_indices(C,1)]
        pair_updates=int((d*(d-1)//2).sum())
        assert int(tri.sum())==pair_updates
        assert int(G.sum())==int((d*d).sum())
        # Independently check bit packing and AND-popcount against direct integer
        # multiplication on a fixed small prefix, without treating it as timing.
        small=S[:127,:min(C,32)].astype(np.int64)
        ref=small.T@small
        packed=np.packbits(small.astype(np.uint8),axis=0,bitorder='little').T
        actual=np.array([[int(POP[a&b].sum()) for b in packed] for a in packed])
        assert np.array_equal(ref,actual)
        nonzero=int(np.count_nonzero(tri));total=C*(C-1)//2
        row={'sample_id':0,'module':spec['module_name'],'N':N,'C':C,'H':H,'frames':frame_counts[lid],
             'active_elements':int(d.sum()),'mean_active_per_row':float(d.mean()),'mean_square_active_per_row':float((d*d).mean()),
             'max_active_per_row':int(d.max()),'row_active_histogram':np.bincount(d,minlength=C+1).tolist(),
             'source_one_count_updates':int(d.sum()),'source_distinct_pair_count_updates':pair_updates,
             'nonzero_offdiagonal_gram_entries':nonzero,'offdiagonal_gram_entries':total,'offdiagonal_gram_nonzero_fraction':nonzero/total,
             'upper_gram_count_payload_bytes':(C*(C+1)//2*N.bit_length()+7)//8,
             'full_quadratic_contraction_terms':C*(C+1)//2*H,
             'normal_fc1_active_scalar_weight_terms':int(d.sum())*H,
             'note':'Counts use distinct unordered i<j pairs; diagonal k separately. Scalar terms are not cycles/energy.'}
        rows.append(row)
        print(spec['module_name'].split('layers.')[1], 'mean_active',round(row['mean_active_per_row'],3),'pair_updates',pair_updates,'G_density',round(nonzero/total,5),flush=True)
    result={'status':'CPU_SOURCE_STATISTICS_ONLY','sample_selection':'predeclared sample_id=0; all 12 FC1 layers; each full T×B×H×W domain, no frame/row subsampling',
            'sources':{str(CAP/k):v for k,v in EXPECTED.items()},'layers':rows,
            'checks':'source hashes; all selected frames CRC/shape/order/count/code checks; Gram diagonal and pair-count identities; fixed prefix integer matrix reference',
            'claim_boundary':{'RTL_SPEEDUP_ADMISSION':0,'PPA_ADMISSION':0,'accuracy_claim':False,'frozen_FP_bit_exact':False,'new_GPU_capture':False,'scope_generalization':'not S40 or valid825 distributions'},
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
