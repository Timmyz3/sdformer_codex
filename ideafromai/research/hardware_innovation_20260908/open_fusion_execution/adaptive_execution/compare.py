"""Isolated full-K ProSparsity residuals on the existing column-broadcast U engine.

Early executable prototype, NOT the official Prosperity/GustavSNN simulator.
The parent rule follows Prosperity (HPCA 2025), MIT, Chiyue Wei 2024.
Full K864, actual theta-absorbed dyadic coefficients, no training/quantization.
The scalar match controller is deliberately explicit; a parallel CAM remains
an untested implementation, not an implicit free matcher in these results.
"""
from pathlib import Path
import argparse
import json
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
FULL = BASE / 'algorithm/patch_probe/residual_consumer_probe/projection_chain/fast_temporal_recovery_lifting40/schedule_compare_same_port/full_chain'
sys.path.insert(0, str(FULL / 'preview_sn2_chain'))
from run_chain import Machine, coefficients, build_nrv, execute_u, NRV, Z, SRC
from consumer_service import read_npz
from numerical_reference import difference

KEY, PARENTS = 98304, 100608


def parent_rule(keys):
    """Same max-popcount subset / earliest equal-pattern parent as official code."""
    counts = [x.bit_count() for x in keys]
    parent = [-1] * len(keys)
    for i, key in enumerate(keys):
        if counts[i] < 2:
            continue
        best = 0
        for j, candidate in enumerate(keys):
            if candidate & ~key or (candidate == key and j >= i):
                continue
            if counts[j] > best:
                best, parent[i] = counts[j], j
    return parent, counts


def extract_keys(m, n, rows):
    # Controller's one 64-bit key-word latch; keys reside in the common SRAM.
    m.phase = 'fullK_match_key_clear'
    for off in range(0, rows * 112, 8):
        m.advance(write=(KEY + off, bytes(8)), tag='key_clear')
    m.phase = 'fullK_match_key_build'
    for record in range(n):
        raw = m.read_word(NRV + record * 8)
        k, mask = int.from_bytes(raw[:4], 'little'), int.from_bytes(raw[4:], 'little')
        while mask:
            bit = mask & -mask
            tp = bit.bit_length() - 1
            address = KEY + tp * 112 + k // 64 * 8
            word = int.from_bytes(m.read_word(address), 'little')
            word |= 1 << (k % 64)
            m.advance(tag='key_bit_insert')
            m.advance(write=(address, word.to_bytes(8, 'little')), tag='key_update')
            mask ^= bit
    # Functional copies below are verification/controller reference only.
    # Pair comparisons are charged below from real SRAM key reads.
    return [int.from_bytes(m.state[KEY+i*112:KEY+(i+1)*112], 'little') for i in range(rows)]


def match(m, n, rows):
    start = m.time
    keys = extract_keys(m, n, rows)
    counts = []
    m.phase = 'fullK_popcount'
    for i in range(rows):
        count = 0
        for word in range(14):
            count += int.from_bytes(m.read_word(KEY+i*112+word*8), 'little').bit_count()
            m.advance(tag='scalar_popcount')
        counts.append(count)
    m.phase = 'fullK_parent_match'
    parent = [-1] * rows
    for i in range(rows):
        if counts[i] < 2:
            continue
        best = 0
        for j in range(rows):
            m.advance(tag='parent_count_filter')
            if counts[j] <= best or counts[j] > counts[i] or (counts[j] == counts[i] and j >= i):
                continue
            subset = True
            for word in range(14):
                a = int.from_bytes(m.read_word(KEY+i*112+word*8), 'little')
                b = int.from_bytes(m.read_word(KEY+j*112+word*8), 'little')
                m.advance(tag='scalar_subset_compare')
                if b & ~a:
                    subset = False
                    break
            if subset:
                best, parent[i] = counts[j], j
        # 20-entry counts/parent controller table, 4B per entry, common SRAM.
    for i in range(0, rows, 2):
        values = [(counts[j] << 16) | (parent[j]+1) for j in range(i, min(i+2, rows))]
        payload = b''.join(v.to_bytes(4, 'little') for v in values).ljust(8, b'\0')
        m.advance(write=(PARENTS+i*4, payload), tag='parent_table_store')
    expected, expected_counts = parent_rule(keys)
    assert parent == expected and counts == expected_counts
    return keys, parent, counts, m.time - start


def rewrite_residuals(m, n, rows, parent):
    m.phase = 'fullK_residual_masks'
    removed, live_records = 0, 0
    for record in range(n):
        raw = m.read_word(NRV+record*8)
        k, original = int.from_bytes(raw[:4], 'little'), int.from_bytes(raw[4:], 'little')
        mask = original
        for i in range(rows):
            if parent[i] < 0:
                continue
            address = PARENTS + i*4
            entry = int.from_bytes(m.read_word(address)[address%8:address%8+4], 'little')
            p = (entry & 65535)-1
            m.advance(tag='parent_mask_filter')
            if original & (1 << p):
                assert original & (1 << i)
                mask &= ~(1 << i)
        removed += original.bit_count()-mask.bit_count()
        if mask:
            payload = k.to_bytes(4, 'little')+mask.to_bytes(4, 'little')
            m.advance(write=(NRV+live_records*8, payload), tag='residual_NRV_write')
            live_records += 1
    return live_records, removed


def forest_u(m, n, base, rows, parent, counts):
    m.phase = 'preview_U32_residual_GP'
    for r in range(rows*4):
        m.advance(op=('clear',r,None),tag='U_clear')
    for record in range(n):
        raw = m.read_word(NRV+record*8)
        k, mask = int.from_bytes(raw[:4],'little'), int.from_bytes(raw[4:],'little')
        m.advance(tag='NRV_select')
        for hg in range(4):
            m.coefficient(base['U']+(k*32+hg*8)*4)
            for tp in range(rows):
                if mask & (1<<tp):
                    dst=tp*4+hg
                    m.wait_reg(dst)
                    m.advance(op=('FMA',dst,(None,None)),tag='U_active_issue')
    m.phase='fullK_parent_RF_propagation'
    for tp in sorted(range(rows),key=lambda i:(counts[i],i)):
        if parent[tp] < 0:
            continue
        address=PARENTS+tp*4
        entry=int.from_bytes(m.read_word(address)[address%8:address%8+4],'little')
        p=(entry&65535)-1
        for hg in range(4):
            dst,src=tp*4+hg,p*4+hg
            m.wait_reg(dst);m.wait_reg(src)
            m.advance(op=('add_reg',dst,src),tag='parent_vector_add')
    m.drain()
    z=np.empty((rows,32),np.float32)
    for hg in range(4):
        for tp in range(rows):
            dst=tp*4+hg
            z[tp,hg*8:hg*8+8]=m.rf[dst]
            m.advance(op=('TF32',dst,None),tag='Z_conversion')
            m.store_reg(dst,Z+(tp*32+hg*8)*4)
    return z


def run(data, params, label, mode, stress):
    geo=json.loads(str(data['window_geometry_json']))[label]
    source=data[label+'_sn1_gate']
    words=sum(source[t].astype(np.uint16)<<t for t in range(10))
    blob,base,info=coefficients(params,False)
    m=Machine(stress)
    m.phase='common_coefficient_fill';m.dma_input(blob,0,True)
    m.phase='common_source_input';m.dma_input(words.astype('<u2').tobytes(),SRC)
    output=[];reports=[]
    h,w=geo['gate_shape'];oy,ox=geo['gate_origin']
    for y in range(h):
        for x in range(0,w,2):
            positions=[(oy+y,ox+x+p) for p in range(min(2,w-x))]
            rows=10*len(positions);start=m.time
            n,occ,live=build_nrv(m,words,geo['source_origin'],positions)
            parents=removed=0;match_slots=0;selected='GP'
            if mode=='gp':
                z=execute_u(m,n,base,False,len(positions))
            else:
                keys,parent,counts,match_slots=match(m,n,rows)
                parents=sum(p>=0 for p in parent)
                saved=sum(counts[p] for p in parent if p>=0)
                # Observable pre-execution bound, no timed-both-paths oracle.
                # Even optimistic saved issue slots must pay actual matching,
                # scalar mask filtering, residual writes, and parent adds.
                threshold=match_slots+n*max(1,parents)+4*parents
                choose=mode=='forest' or saved*4>threshold
                if choose:
                    nn,removed=rewrite_residuals(m,n,rows,parent)
                    z=forest_u(m,nn,base,rows,parent,counts);selected='forest_GP'
                else:
                    z=execute_u(m,n,base,False,len(positions))
            output.append(z)
            reports.append(dict(y=y,x=x,source_occurrences=occ,NRV_rows=n,parents=parents,
                removed_occurrences=removed,match_slots=match_slots,selected=selected,service=m.time-start))
    return np.concatenate(output),dict(service_slots=m.time,stages=dict(m.stages),counts=dict(m.count),
        groups=reports,coefficient_bytes=info['used_bytes'],state_capacity=131072,
        RF_words=96,accumulator_words=80,matcher_state_bytes=2320,
        selector='visible metadata only; no free trial of both schedules')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stress',action='store_true');args=ap.parse_args()
    result=dict(scope='Actual two preselected windows, full K864 preview U32 only; CPU payload/slot prototype, not RTL/PPA or whole-layer cycles.',
        provenance='Prosperity subset rule, column-broadcast residual execution, one full-K parent RF propagation. Not official full-architecture replication.',
        axes={})
    for axis in ('ordinary','lifting_raw'):
        cap=FULL/'capture'/axis
        data=read_npz(cap/'000_zurich_city_09_a_0001.npz');p=read_npz(cap/'live_parameters.npz')
        result['axes'][axis]={}
        for label in ('corner','interior'):
            ref=None;entry={}
            for mode in ('gp','forest','adaptive'):
                z,report=run(data,p,label,mode,args.stress)
                if ref is None: ref=z
                report['difference_vs_GP']=difference(z,ref)
                assert report['difference_vs_GP']['differences']==0
                entry[mode]=report
                print(axis,label,mode,report['service_slots'],flush=True)
            base=entry['gp']['service_slots']
            for mode in ('forest','adaptive'):
                entry[mode]['net_service_reduction']=1-entry[mode]['service_slots']/base
            result['axes'][axis][label]=entry
    out=HERE/('stress.json' if args.stress else 'ready.json')
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__': main()
