"""Append actual trained sparse-W cases without changing the original fixtures."""
from pathlib import Path
import json
import struct
import numpy as np
from prepare_cases import ROOT,HERE,make


def encode(c):
    n=c['name'].encode();out=bytearray(struct.pack('<H',len(n))+n)
    out.extend(bytes([len(c['decode']),c['metadata']['real'],c['metadata']['route']=='packed_time']))
    out.extend(c['theta'].tobytes())
    table=np.sum(c['decode'].T.astype(np.uint64)<<np.arange(len(c['decode']),dtype=np.uint64),axis=1).astype(np.uint8)
    out.extend(table.tobytes());out.extend(struct.pack('<H',len(c['program'])));out.extend(c['program'].tobytes())
    for key,dtype in [('coeff','<i4'),('tau','<i4'),('W','i1'),('codes','u1'),
                      ('S','<i4'),('present','u1'),('gate_words','<u2')]:
        out.extend(np.asarray(c[key],dtype=dtype).tobytes())
    return bytes(out)


def main():
    books=np.load(ROOT/'algorithm/stage2_temporal_codes/codebooks.npz')
    basis=np.load(ROOT/'algorithm/stage2_temporal_codes/signed_basis.npz')
    D=books['s2b3_dictionary'].astype(np.int64)
    rows=[];seen=set()
    for t in range(10):
        key=tuple(D[:,t]);
        if any(key) and key not in seen:rows.append(t);seen.add(key)
    matrix=D[basis['s2b3_selected_code_indices']].T[rows]
    inverse=np.rint(np.linalg.inv(matrix)).astype(np.int64)
    assert np.array_equal(matrix@inverse,np.eye(6,dtype=np.int64))
    extra=[];prune=ROOT/'algorithm/pruning_probe'
    for variant in ('row_2of4_trained','broadcast8_C16_half_trained'):
        with np.load(prune/(variant+'.npz')) as f:
            W=f['weight_int8'];tau=f['threshold_int32'];B=f['B_int8'].astype(np.int64);E=f['E_int8'].astype(np.int64).T
            assert W.shape==(1536,384) and np.all(f['hidden_keep'])
        Y=D[:,rows].T;A=B@inverse;B7=(B@E)[:,1:]
        assert np.array_equal(A@Y,B@E)
        for i,seq in enumerate(('zurich_city_09_a_0001','zurich_city_02_c_0011')):
            path=prune/'codes'/variant/seq/'s2b3.npz'
            p=408+56*i;hs=[633+97*i,1366+97*i]
            with np.load(path) as f:codes=f['codes'][p:p+16]
            same=[]
            for route,decode,coeff in [('packed_class',np.eye(8,dtype=np.int64)[1:],B7),('packed_time',Y,A)]:
                c=make(f'pruned_{variant}_{seq}_s2b3_p{p}_h{hs[0]}_{hs[1]}_{route}',codes,W[hs],decode,coeff,tau[:,hs],[1,1],
                    dict(real=True,route=route,capture=str(path.relative_to(ROOT)),block=3,
                         positions=list(range(p,p+16)),hidden_rows=hs,theta_source=1.0,weight_variant=variant,
                         weight_density=[float(np.mean(W[h]!=0)) for h in hs],
                         identity='Saved trained sparse W, matching capture and B/E/tau; not a synthetic pruning mask'))
                extra.append(c);same.append(c['gate_words'])
            assert np.array_equal(*same)
    original=(HERE/'cases.bin').read_bytes();assert original[:4]==b'GPS1'
    original_count=struct.unpack('<I',original[4:8])[0]
    (HERE/'intersection_cases.bin').write_bytes(b'GPS1'+struct.pack('<I',original_count+len(extra))+original[8:]+b''.join(map(encode,extra)))
    summary=json.loads((HERE/'cases.json').read_text())
    summary['total_cases']+=len(extra);summary['real_cases']+=len(extra)
    summary['additional_actual_sparse_captures']=4
    summary['cases'] += [dict(name=c['name'],program_length=len(c['program']),**c['metadata']) for c in extra]
    summary['comparison']='same case, dense W value lookup versus actual byte-addressed sorted index/value traversal; class/time and scalar/member remain independent controls'
    (HERE/'intersection_cases.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(f'{original_count}+{len(extra)} cases; appended real row2of4/C16 trained W with their own captures')


if __name__=='__main__':main()
