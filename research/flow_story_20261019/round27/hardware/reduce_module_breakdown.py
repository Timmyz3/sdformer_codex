#!/usr/bin/env python3
"""Lossless operation-family reclassification of frozen arithmetic PE ticks."""
import collections, hashlib, json
from pathlib import Path
HERE=Path(__file__).resolve().parent
p=HERE/'UNIFIED_SERVICE_V1.json';h=hashlib.sha256(p.read_bytes()).hexdigest()
assert h=='f1aca17e9a6f7e3afc6f57260fce313ae8197d61eedb5c5c9c0d65603ac6ad14'
d=json.loads(p.read_text());inventory={x['name']:x for x in d['inventory']['affines']}
family={'conv':'convolution_affine_20','MLP_FC':'MLP_FC_24','attention_projection':'attention_projection_36','patch_merge':'patch_merge_linear_3'}
answer={}
for mode in ('Q8_strong','Q6_serial'):
 result=next(r for r in d['results'] if r['scope']=='MLP12_patch4' and r['implementation']==mode)
 groups=collections.Counter();producer_mean=0
 for r in result['records']:
  pe=r['PE_arithmetic_ticks']
  if r['category']=='affine':
   bn=r['producer_BN_mean_PE_ticks'];producer_mean+=bn
   groups[family[inventory[r['name']]['kind']]]+=pe-bn;groups['current_BN_78_including_producer_mean']+=bn
  elif r['category']=='current_BN':groups['current_BN_78_including_producer_mean']+=pe
  elif r['category']=='PSN':groups['PSN_93_live']+=pe
  elif 'attention_pos_add' in r['name'] or r['category'] in ('attention_reduce','attention_gate'):groups['attention_aux_binary_and_position']+=pe
  elif r['category']=='remap':groups['layout_copy_PE_free_elapsed_not_free']+=pe
  else:groups['other_residual_and_final_output']+=pe
 assert sum(groups.values())==result['summary']['PE_arithmetic_ticks']
 answer[mode]=dict(total_arithmetic_PE_ticks=result['summary']['PE_arithmetic_ticks'],groups={k:dict(PE_arithmetic_ticks=v,fraction=v/sum(groups.values())) for k,v in groups.items()},producer_mean_reclassified_ticks=producer_mean)
out=dict(schema='flow_r27_exclusive_operation_family_PE_breakdown_v1',source_result_sha256=h,
 notes=['Arithmetic PE work is additive: producer BN mean is moved from its affine job into BN once. No arithmetic cost is lost or double counted.',
 'PE reserved/local-port max terms cannot be uniquely reattributed by subtraction; use the original resource records for them. This table does not claim an additive causal wall-time decomposition.',
 'Conv/MLP/attention rows are affine+scale/bias costs, with BN/PSN separately shown. Structural blocks include both and must not be summed again.',
 'All numbers are the frozen assumed machine, not calibrated ASIC cycles; no R27 quality read.'],scenarios=answer)
path=HERE/'MODULE_PE_BREAKDOWN_V1.json';assert not path.exists();path.write_text(json.dumps(out,indent=2)+'\n')
lines=['# 固定主机器：整图算术 PE 分母分解','','只读冻结 `UNIFIED_SERVICE_V1.json`；不跑模型、不读GT、不改变任何资源假设。以下是互斥运算族：BN和PSN单列，因此“MLP FC”不是整个MLP结构块的全部成本。producer内支付的BN均值加法重新归入BN一次，合计严格守恒。','','| 运算族 | 强Q8算术PE tick | 占强Q8整图 | 固定Q6算术PE tick |','|---|---:|---:|---:|']
for name,v in answer['Q8_strong']['groups'].items():
 q=answer['Q6_serial']['groups'][name]['PE_arithmetic_ticks'];lines.append(f"| {name} | {v['PE_arithmetic_ticks']:,} | {v['fraction']*100:.4f}% | {q:,} |")
lines+=['| **合计** | **'+f"{answer['Q8_strong']['total_arithmetic_PE_ticks']:,}"+'** | **100%** | **'+f"{answer['Q6_serial']['total_arithmetic_PE_ticks']:,}"+'** |','','本地端口等待取max，无法像算术工作一样唯一归属后相加，因此不伪造按模块的可加wall time。实际原生105个PSN中的12个诊断死输出已对双方同样剔除；原执行附表见主README。layout的算术PE=0不表示本地搬运或DMA免费。完整BN统计、全部T以及未选中的卷积/注意力均在分母中。']
(HERE/'MODULE_PE_BREAKDOWN_V1.md').write_text('\n'.join(lines)+'\n')
print(json.dumps(answer,indent=2))
