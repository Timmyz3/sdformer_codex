"""Check source work against independent all-input CPU graphs; separate selection split."""
from pathlib import Path
from collections import defaultdict
import csv,json

HERE=Path(__file__).resolve().parent
FIELDS=['cycles','channels','scalar_mac','words','xwords','graph_words','prefetch_words','graph_hits','req_stall','out_stall']

def main():
    probe=json.loads((HERE/'probe.json').read_text())['variants']
    out=[];verified=0
    for name,key in [('zero','zero_inclusive_producer_cost'),('l2','zero_inclusive_integer_L2')]:
        for scope,nframes in [('small',2),('expanded',32)]:
            rows=[]
            for raw in csv.DictReader((HERE/f'source_{name}_{scope}.csv').open()):
                r={k:v if k=='case' else int(v) for k,v in raw.items()};rows.append(r)
                assert r['mode'] in (2,3) and r['order']==r['packed32']==r['prefetch']==1
                assert r['scalar_mac']==100*r['channels'] and r['xwords']==2*r['channels']
                assert r['words']==r['xwords']+r['graph_words']+21
                assert r['cycles']+1==sum(r[f'state{i}'] for i in range(14))
            assert len(rows)==(nframes*32+3)*4
            for frame in range(nframes):
                for bp in (0,1):
                    for mode in (2,3):
                        rr=[r for r in rows if r['case'].startswith(f'train{frame}_') and r['mode']==mode and r['bp']==bp]
                        assert len(rr)==32
                        expected=probe['original' if mode==2 else key]['source_jobs_by_order_then_training_frame'][1][frame]
                        assert sum(r['channels'] for r in rr)==expected
            for split,frames in [('selection_frame0',[0]),('unselected_training_frames',list(range(1,nframes))),('all_training_frames',list(range(nframes)))]:
                for bp in (0,1):
                    counts={}
                    for mode in (2,3):
                        rr=[r for r in rows if r['real'] and int(r['case'].split('_')[0][5:]) in frames and r['mode']==mode and r['bp']==bp]
                        counts[mode]={f:sum(r[f] for r in rr) for f in FIELDS}
                    a,b=counts[2],counts[3]
                    out.append(dict(function=name,scope=scope,split=split,frames=frames,bp=bp,code=a,response_class=b,
                                    cycle_savings_percent=100*(1-b['cycles']/a['cycles'])))
            if scope=='expanded':
                small=list(csv.DictReader((HERE/f'source_{name}_small.csv').open()))
                actual={(r['case'],r['mode'],r['bp']):r for r in rows}
                for old in small:
                    new=actual[(old['case'],int(old['mode']),int(old['bp']))]
                    assert all(new[k]==int(old[k]) for k in FIELDS)
            verified+=len(rows)
    result=dict(status='PASS',commands=verified,source_X_MAC_rawgate_code_class_checked_in_CPP=True,
                independent_graph_channel_and_word_accounting=True,first_two_frames_replay_identical=True,
                selection='frame0 only; frame1..31 are unselected TRAINING probes, not validation',
                period='cycle-index difference start to done, matching parent source harness; inclusive histogram sums cycles+1',summaries=out)
    (HERE/'SOURCE_SUMMARY.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({'status':'PASS','commands':verified}))

if __name__=='__main__':main()
