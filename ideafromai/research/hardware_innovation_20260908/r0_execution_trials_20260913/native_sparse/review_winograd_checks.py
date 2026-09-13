#!/usr/bin/env python3
"""Bounded independent math/format/counter audit; does not run or modify RTL."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
D=HERE.parent/'winograd'
BT=np.array([[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]],dtype=np.int64)
AT=np.array([[1,1,1,0],[0,1,-1,-1]],dtype=np.int64)
G2=np.array([[2,0,0],[1,1,1],[1,-1,1],[0,0,2]],dtype=np.int64)
L=np.kron(AT,AT);R=np.kron(BT,BT)

def support(path,count):
    words=[int(s,16) for s in path.read_text().split()]
    return np.array([(words[i//64]>>(2*(i%64)))&3 for i in range(count)],dtype=np.int64)

def successor_visits(sup,keys):
    visits=0
    for g in range(6):
        k=0;seen=[]
        while k<keys:
            seen.append(k);visits+=1
            end=min((k//64+1)*64,keys)
            later=[j for j in range(k+1,end) if sup[g*keys+j]]
            k=later[0] if later else end
        assert set(np.flatnonzero(sup[g*keys:(g+1)*keys])).issubset(seen)
        assert len(seen)==len(set(seen))
    return visits

def audit():
    summary={}
    # Exhaust the finite source alphabet; independent of any real fixture.
    bitmaps=((np.arange(65536,dtype=np.uint32)[:,None]>>np.arange(16))&1).astype(np.int64)
    vals=bitmaps@R.T
    assert vals[:,5].min()==0 and vals[:,5].max()==4
    assert np.all(np.abs(np.delete(vals,5,axis=1))<=2)
    assert np.all(np.argwhere(vals==3)[:,1]==5)
    summary['binary_alphabet_patterns']=65536
    summary['V_min_by_coordinate']=vals.min(axis=0).tolist()
    summary['V_max_by_coordinate']=vals.max(axis=0).tolist()
    # Kernel/input basis proves the integer bilinear identity (9 x 16 bases).
    for k in range(9):
        w=np.eye(9,dtype=np.int64)[k].reshape(3,3)
        u=G2@w@G2.T
        expanded=L@np.diag(u.reshape(-1))@R
        direct=np.zeros((4,16),dtype=np.int64)
        for py in range(2):
            for px in range(2):direct[py*2+px,(py+k//3)*4+px+k%3]=1
        assert np.array_equal(expanded,4*direct)
    summary['integer_bilinear_basis_pairs']=144
    # Same linear in-place overwrite order as the documented schedule, checked
    # against matrix multiplication on all 16 basis memories.
    for b in range(16):
        original=np.eye(16,dtype=np.int64)[b].reshape(4,4)
        mem=original.reshape(-1).copy()
        for xi in range(8):
            r,c=divmod(xi,4)
            if r==0:mem[xi]=mem[c]+mem[4+c]+mem[8+c]
            else:mem[xi]=mem[4+c]-mem[8+c]-mem[12+c]
        for xi in range(4):
            r,c=divmod(xi,2)
            if c==0:mem[xi]=mem[r*4]+mem[r*4+1]+mem[r*4+2]
            else:mem[xi]=mem[r*4+1]-mem[r*4+2]-mem[r*4+3]
        assert np.array_equal(mem[:4],(AT@original@AT.T).reshape(-1))
    summary['inplace_inverse_basis_cases']=16
    for n in range(-8192,8193):
        q=n>>2;r=n&3
        rtl=q+int(r>2 or (r==2 and ((n>>2)&1)))
        assert rtl==round(n/4)
    summary['signed_RNE_quarter_checks']=16385
    kinds={'direct':(864,2),'wino':(1536,3),'masked_wino':(1536,3),'masked_expanded':(6144,3)}
    format_checks={}
    f0=np.load(D/'fixtures/real_00/fixture.npz',allow_pickle=False)
    matrices={'direct':f0['Wq'].reshape(96,864),'wino':f0['U4'].reshape(96,1536),
      'masked_wino':(f0['U4']*f0['mask'].repeat(8,axis=0)[:,None,:]).reshape(96,1536),
      'masked_expanded':f0['E4'].reshape(96,6144)}
    for kind,(keys,cb) in kinds.items():
        blob=(D/f'fixtures/real_00/{kind}.bin').read_bytes()
        sup=support(D/f'fixtures/real_00/{kind}.meta',keys*6)
        vectors=matrices[kind].reshape(6,16,keys).transpose(0,2,1).reshape(-1,16)
        decoded_count=0;cross_half=0;alignments=set()
        for vi,vec in enumerate(vectors):
            flag=int(sup[vi]);expected=int(any(vec[:8]))+2*int(any(vec[8:]))
            assert flag==expected
            if not flag:continue
            base=vi*16*cb
            first=(base+(0 if flag&1 else 8*cb))//32
            last=(base+(16 if flag&2 else 8)*cb-1)//32
            assert 0<=last-first<=1
            assembled=blob[first*32:(last+1)*32].ljust(64,b'\0')
            if flag!=3:alignments.add((flag,base%32,last-first+1));cross_half+=int(last>first)
            for lane in range(16):
                if not (flag&(1<<(lane//8))):continue
                off=base+lane*cb-first*32
                assert 0<=off and off+cb<=64
                assert int.from_bytes(assembled[off:off+cb],'little',signed=True)==int(vec[lane])
                decoded_count+=1
        format_checks[kind]={'decoded_live_coefficients':decoded_count,'half_only_cross_line_vectors':cross_half,
            'half_flag_base_offset_rows':sorted(alignments),'static_successor_SCAN_cycles':successor_visits(sup,keys)}
    summary['signed_coefficient_formats']=format_checks
    neg_tie_even=0;neg_tie_odd=0;positive_ties=0
    all_rows=json.loads((D/'rtl_results.json').read_text())
    for tile in range(8):
        name=f'real_{tile:02d}';z=np.load(D/f'fixtures/{name}/fixture.npz',allow_pickle=False)
        S=z['S'];V=np.einsum('ij,tcjk,lk->tcil',BT,S,BT).reshape(10,96,16)
        raw=np.einsum('ocps,tcs->top',z['E4'],S.reshape(10,96,16))
        ties=(raw%4==2);negative=ties&(raw<0)
        neg_tie_even+=int(np.count_nonzero(negative&((raw//4)%2==0)))
        neg_tie_odd+=int(np.count_nonzero(negative&((raw//4)%2!=0)))
        positive_ties+=int(np.count_nonzero(ties&(raw>=0)))
        for kind,(keys,cb) in kinds.items():
            sup=support(D/f'fixtures/{name}/{kind}.meta',keys*6).reshape(6,keys)
            halves=(sup&1)+(sup>>1)
            if kind=='direct':
                events=np.array([int(S[:,c,ky:ky+2,kx:kx+2].sum()) for c in range(96) for ky in range(3) for kx in range(3)])
            elif kind=='masked_expanded':events=np.repeat(S.reshape(10,96,16).sum(axis=0)[:,None,:],4,axis=1).reshape(-1)
            else:events=np.count_nonzero(V,axis=0).reshape(-1)
            expected_aac=int((halves*events[None,:]).sum())
            for row in [r for r in all_rows if r['fixture']==name and r['kind']==kind]:
                st=row['state_cycles']
                assert st[11]==expected_aac
                assert st[5]==format_checks[kind]['static_successor_SCAN_cycles']
    for row in all_rows:
        st=row['state_cycles'];req=row['weight_requests'];stress=row['stress']
        assert sum(st)==row['cycles'] and row['mismatches']==0 and row['values_checked']==3840
        assert row['weight_responses']==req and row['cr_bytes']==32*req
        assert st[7]==req+row['request_stall'] and st[8]==req*(4 if stress else 1)
        assert st[17]==480+row['output_stall'] and st[1]==120+row['input_stall']
        assert st[0]==row['configuration_cycles']+1
    summary['real_negative_ties_floor_even']=neg_tie_even
    summary['real_negative_ties_floor_odd']=neg_tie_odd
    summary['real_positive_ties']=positive_ties
    summary['RTL_JSON_rows_audited']=len(all_rows)
    summary['real_AAC_and_SCAN_rows_audited']=64
    summary['status']='all bounded checks passed; no RTL rerun, no author file mutation'
    (HERE/'review_winograd_checks.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':audit()
