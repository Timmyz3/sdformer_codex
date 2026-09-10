#!/opt/anaconda3/bin/python3.12
"""Same-capture M0 workload ledger; MAC proxies, explicitly not hardware cycles."""
import collections
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAP = Path(json.loads((HERE/'plan.json').read_text())['input_root'])
execution = json.loads((CAP/'execution_trace.json').read_text())
ordered = [json.loads(line) for line in (CAP/'unified_ordered_records.jsonl').read_text().splitlines()]
atlif = json.loads((CAP/'atlif_activity.json').read_text())
dead = {r['name'] for r in atlif if r.get('deployment_dead_result', False)}


def family(name):
    if '.attn.linear_' in name: return 'QK_projection'
    if name.endswith('.attn.proj'): return 'attention_output_projection'
    if '.mlp.fc1' in name: return 'FC1'
    if '.mlp.fc2' in name: return 'FC2'
    if '.downsample.' in name: return 'downsample'
    if '.patch_embed.residual_encoding.' in name: return 'patch_residual_conv3x3'
    if '.patch_embed.' in name: return 'patch_other_convolution'
    if name.startswith('sttmultires_unet.resblocks.'): return 'bottleneck_conv3x3'
    if '.preds.' in name: return 'flow_heads'
    return 'other_dot_operator'


groups = collections.defaultdict(lambda: {'calls':0, 'names':set(), 'dense_macs':0,
                                         'activity_weighted_macs_proxy':0.0})
for r in execution:
    if r['kind'] != 'operator': continue
    g = groups[family(r['name'])]
    g['calls'] += 1; g['names'].add(r['name']); g['dense_macs'] += r['dense_macs']
    g['activity_weighted_macs_proxy'] += r['dense_macs']*r['input_active']/r['input_elements']

# Existing operator_runtime omits ConvTranspose. Derive its MAC opportunities
# from the same calls and actual next-BN output shapes. 3x3, stride2,pad1,opad1:
# sum of valid taps across one spatial axis is 3*input_size-1.
bn_shapes = {(r['global_sample_id'],r['name']):r['input']['shape'] for r in ordered if r['category']=='batch_norm'}
decoder_details=[]
for r in ordered:
    if r['category'] != 'decoder_convtranspose': continue
    inp = r['input']; t,b,ci,h,w = inp['shape']
    base = r['name'].rsplit('.deconv.',1)[0]
    outshape = bn_shapes[(r['global_sample_id'],base+'.norm_layer.norm_layer')]
    co = outshape[2]
    mac = t*b*ci*co*(3*h-1)*(3*w-1)
    dense_including_cropped_taps = t*b*ci*co*h*w*9
    g = groups['decoder_convtranspose']
    g['calls'] += 1; g['names'].add(r['name']); g['dense_macs'] += mac
    g['activity_weighted_macs_proxy'] += mac*inp['active']/inp['elements']
    decoder_details.append({'sample':r['global_sample_id'],'name':r['name'],
                            'input_shape':inp['shape'],'output_shape':outshape,
                            'valid_tap_dense_macs':mac,'dense_including_cropped_taps':dense_including_cropped_taps})

psn=collections.defaultdict(lambda:{'calls':0,'modules':set(),'dense_macs':0,'elements':0})
for r in execution:
    if r['kind']!='atlif': continue
    key=('dead_result_' if r['name'] in dead else 'graph_live_')+'T'+str(r['temporal_steps'])
    a=psn[key];a['calls']+=1;a['modules'].add(r['name']);a['dense_macs']+=r['dense_macs'];a['elements']+=r['output_elements']

attention=collections.defaultdict(lambda:{'calls':0,'pairs':0,'score_tokens':0,'normalization_rows':0})
for r in execution:
    if r['kind']!='attention': continue
    a=attention['S'+str(r['stage'])];a['calls']+=1
    a['pairs']+=r['pair_total'];a['score_tokens']+=r['token_total'];a['normalization_rows']+=r['token_total']//450
score_tokens=sum(a['score_tokens'] for a in attention.values())
psn_live=sum(v['dense_macs'] for k,v in psn.items() if k.startswith('graph_live'))
dot_dense=sum(g['dense_macs'] for g in groups.values())
dot_active=sum(g['activity_weighted_macs_proxy'] for g in groups.values())
gated_k_multiply=score_tokens*32

# lambda is a declared conversion assumption (scalar MAC-equivalent work per
# 32-lane score), not measured latency. Show sensitivity instead of summing
# popcounts and MACs with an unannounced exchange rate.
conversion=[]
for name,base in [('dense_dot_plus_live_PSN',dot_dense+psn_live),
                  ('activity_proxy_dot_plus_live_PSN',dot_active+psn_live)]:
    for lam in [1,32,96,256,1024]:
        score_work=lam*score_tokens
        conversion.append({'denominator':name,'MAC_equivalent_assumption_per_score':lam,
                           'score_share_percent':100*score_work/(base+gated_k_multiply+score_work)})

full_density={}
for stage in range(4):
    full_density['S'+str(stage)]={}
    for which in ['q','k']:
        rows=[r for r in atlif if f'.layers.{stage}.' in r['name'] and f'.attn.sn_{which}.' in r['name']]
        active=sum(r['active'] for r in rows);elements=sum(r['elements'] for r in rows)
        full_density['S'+str(stage)][which]={'active':active,'elements':elements,'density':active/elements,
                                           'negative_outputs':sum(r['neg'] for r in rows)}

table=[]
for name,g in groups.items():
    table.append({'family':name,'operators':len(g['names']),'calls':g['calls'],
                  'dense_macs_per_frame':g['dense_macs']/40,
                  'active_macs_proxy_per_frame':g['activity_weighted_macs_proxy']/40,
                  'dense_dot_share_percent':100*g['dense_macs']/dot_dense})
table.sort(key=lambda r:-r['dense_macs_per_frame'])
with (HERE/'m0_operator_table.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
result={
 'status':'SAME_EP34_40_SAMPLE_MAC_WORKLOAD_PROXY_NOT_CYCLE_SHARE',
 'samples':40,'input_capture':str(CAP),'operator_table':table,
 'dot_operators':sum(len(g['names']) for g in groups.values()),
 'dot_dense_macs_per_frame':dot_dense/40,'dot_active_macs_proxy_per_frame':dot_active/40,
 'PSN':{k:dict(v,modules=sorted(v['modules'])) for k,v in psn.items()},
 'attention_full_call_work':dict(attention),
 'score_tokens_per_frame':score_tokens/40,
 'three_popcounts_if_separate_per_token_per_frame':3*score_tokens/40,
 'motion_popcounts_with_obvious_pair_sharing_per_frame':score_tokens/80,
 'gate_times_K_scalar_multiply_opportunities_per_frame':gated_k_multiply/40,
 'BN_input_elements_per_frame':sum(r['input']['elements'] for r in ordered if r['category']=='batch_norm')/40,
 'proxy_conversion_sensitivity':conversion,
 'lambda_needed_for_2percent':{
     name:(base+gated_k_multiply)*.02/.98/score_tokens
     for name,base in [('dense_dot_plus_live_PSN',dot_dense+psn_live),('activity_proxy_dot_plus_live_PSN',dot_active+psn_live)]},
 'full_call_QK_density':full_density,
 'decoder_details':decoder_details,
 'limits':[
   'MAC table covers83 dot operators; original79-row operator_runtime omitted4 ConvTranspose operators.',
   'ConvTranspose counts valid 3x3 scatter taps at stride2,pad1,outputpad1; density proxy assumes activity uniformly distributed over tap counts.',
   'Activity-weighted MAC is a scalar support proxy, not strongest-zero execution, product sparsity or a latency prediction.',
   'Only81 consumed ATLIF modules enter graph-live PSN costs;12 called attn_sn dead results are listed separately.',
   'PSN entries count full temporal matrix MACs; other neuron comparisons/bias/threshold/state traffic are not converted into MACs.',
   'BN/current-domain statistics, Shiftmax pow2/max/sum/log, residual/position adds, interpolation, layout/traffic and scheduling are not silently free; they are outside this MAC denominator.',
   'Score lambda is an explicit unit-conversion sensitivity, not an implementation or measured clock count.',
   'No ep34 whole-system cycle share, RTL speedup, physical PPA, or automatic2percent gate decision follows.'
 ]}
(HERE/'m0_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:result[k] for k in ['status','dot_operators','dot_dense_macs_per_frame','dot_active_macs_proxy_per_frame','score_tokens_per_frame','proxy_conversion_sensitivity','lambda_needed_for_2percent']},ensure_ascii=False,indent=2))
