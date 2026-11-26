"""Independent raw-input/count/tree/trace check; no hardware implementation import."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np

B = Path(__file__).resolve().parent
P = B.parent/'round37_cheap_selector'
H = P/'hardware/actual_v1'
A = P/'algorithm/actual_v1'


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    hardware = json.loads((H/'CPU_EXECUTION_V1.json').read_text())
    pred = json.loads((A/'PREDICTIONS_BEFORE_QUALITY_V1.json').read_text())
    assert hardware['status'] == 'PASS' and hardware['fit_identities'] == 64
    ids = pred['ids']
    assert set(ids) == {r['target_id'] for r in hardware['rows']}
    byid = {}
    inputs = {}
    for row in hardware['rows']:
        tid = row['target_id']
        path = Path(row['source'])
        assert digest(path) == row['input_sha256']
        z = np.load(path, allow_pickle=False)
        assert z.shape == (10,480,640) and z.dtype == np.float32 and np.isfinite(z).all()
        block = z.reshape(10,4,120,4,160)
        pos = np.sum(block > 0, axis=(0,2,4), dtype=np.uint64)
        neg = np.sum(block < 0, axis=(0,2,4), dtype=np.uint64)
        counts = np.stack([pos,neg],axis=-1).reshape(-1)
        actual = np.fromfile(H/(tid+'.u32'), dtype='<u4')
        backing = np.fromfile(H/(tid+'.u32.rf'), dtype='<u4')
        assert np.array_equal(counts, actual) and np.array_equal(actual,backing[:32])
        assert np.array_equal(actual,pred['counts'][ids.index(tid)])
        assert np.array_equal(actual,row['count'])
        assert row['service']['feature_ready_ticks'] == 192000*12+188*32+32+2+16+32
        byid[tid] = actual
        inputs[tid] = row['input_sha256']
    for replay in hardware['oof_replays']:
        tid, name, f = replay['target_id'], replay['policy'], replay['fold']
        fold = pred['folds'][f]
        assert tid in fold['held_ids'] and tid not in fold['fit']['fit_ids']
        packed = np.fromfile(H/f'fold{f}_{name}.u32', dtype='<u4')
        node, visits = int(packed[12]), 0
        while not node & 0x80000000:
            assert node < 3 and visits < 2
            feature, threshold, left, right = map(int, packed[node*4:node*4+4])
            assert feature < 32 and 0 <= threshold <= 192000
            node = left if byid[tid][feature] <= threshold else right
            visits += 1
        action = node & 7
        assert action == replay['action'] == pred['actions'][name][ids.index(tid)]
        assert replay['visited_nodes'] == visits
        assert replay['head_ticks'] == 2+7*visits
        assert replay['full_prefix_ticks'] == 2310098+2+7*visits
        end = 0
        for event in replay['head_events']:
            assert event['start'] == end
            assert event['end']-event['start'] == 1
            end = event['end']
            if event['op'] == 'read':
                assert event['address']%4 == 0 and 0 <= event['address'] <= 188
        # The event list covers reads/compare/branch. The paid final action
        # output is a separate one-tick service, not a register event.
        assert end+1 == replay['head_ticks']
    trace_dtype = np.dtype([('start','<u8'),('end','<u8'),('address','<u4'),
                           ('words','<u4'),('memory','u1'),('write','u1'),('vector','u1'),('reserved','u1')])
    trace = np.fromfile(H/'first.trace',dtype=trace_dtype)
    first = hardware['rows'][0]['service']
    assert len(trace) == first['trace_records']
    assert np.all(trace['end']-trace['start'] == 1)
    assert np.all(trace['start'][1:] >= trace['end'][:-1])
    assert np.all(trace['address']%4 == 0) and set(trace['memory']) == {0,1}
    assert np.all((trace['write'] <= 1) & (trace['reserved'] == 0))
    for mid, size in [(0,128),(1,192)]:
        t = trace[trace['memory'] == mid]
        assert np.all(t['address']+4*t['words'] <= size)
        assert np.all(t['words'] == (16 if mid == 0 else 1))
        for write, field in [(0,'read_bytes'),(1,'write_bytes')]:
            assert int(t[t['write']==write]['words'].sum())*4 == first['memory'][mid][field]
    assert int((trace['end']-trace['start']).sum())+sum(first['ops'].values()) == first['action_ready_ticks']
    with (H/'first.trace.dma.csv').open() as f:
        dma = list(csv.DictReader(f))
    assert len(dma) == 189
    assert int(dma[0]['external_offset']) == 12288000 and int(dma[0]['bytes']) == 64
    offset = 0
    for row in dma[1:]:
        assert int(row['external_offset']) == offset
        n = int(row['bytes'])
        assert n%64 == 0 and n <= 65536
        assert int(row['end_startup'])-int(row['start']) == 32
        offset += n
    assert offset == 12288000
    out = B/'COUNT32_RAW_AND_TRACE_INDEPENDENT_V1.json'
    assert not out.exists()
    out.write_text(json.dumps(dict(status='PASS_BOUNDED_CPU_MODEL',
      auditor='root, independent of hardware implementer',
      raw_identities=64,counts=2048,OOF_packed_tree_replays=128,trace_records=len(trace),
      DMA_bursts=189,full_input_bytes=12288000,feature_model_ticks=2310098,
      model_timing_is_not_RTL_or_synthesis=True,whole_network_claim=False,
      final_action_output_tick_paid_separately_not_in_head_event_list=True,
      input_sha256=inputs,source_sha256=digest(__file__),
      bound_files={str(p.relative_to(P)):digest(p) for p in [H/'CPU_EXECUTION_V1.json',H/'first.trace',H/'first.trace.dma.csv',A/'PREDICTIONS_BEFORE_QUALITY_V1.json']}),indent=2)+'\n')
    print('COUNT32_INDEPENDENT_PASS',digest(out))


if __name__=='__main__':
    main()
