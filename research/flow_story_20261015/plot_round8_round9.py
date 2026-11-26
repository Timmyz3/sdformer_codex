#!/usr/bin/env python3
"""Descriptive full-cohort quality figure; no error bars or speedup claim."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
r8=json.loads((HERE/'layer_repair_round8/analysis.json').read_text())
r9=json.loads((HERE/'fixed_layer_confirmation_round9/analysis.json').read_text())
fig,ax=plt.subplots(1,3,figsize=(14,4.7),gridspec_kw={'width_ratios':[1,1,1.35]})
vals=[r8['summaries']['overall']['all_valid']['relative_to_native_percent']['fine_s2b'+str(i)] for i in range(6)]
ax[0].bar(range(6),vals,color=['#b8c1cc','#b8c1cc','#237a8d','#b8c1cc','#b8c1cc','#b8c1cc'])
ax[0].axhline(3,color='#ad4537',ls='--',lw=1.3,label='AllGT screen (+3%)')
ax[0].set_xticks(range(6),['b0','b1','b2','b3','b4','b5']);ax[0].set_xlabel('One complete stage2 MLP restored')
ax[0].set_ylabel('AllGT AEE change vs native (%)');ax[0].set_title('R8: all six equal-service choices\n8 reused development members',fontsize=11)
ax[0].legend(fontsize=8,loc='lower right')
for i,v in enumerate(vals):ax[0].text(i,v+.07,f'{v:.2f}',ha='center',fontsize=9)
x=np.arange(3)
for offset,arm,label,color in [(-.18,'CC','All coarse','#a7b4c5'),(.18,'fine_s2b2','Fixed b2 fine','#237a8d')]:
    v=[r9['summaries']['overall'][reg]['relative_to_native_percent'][arm] for reg in ('all_valid','B','L')]
    ax[1].bar(x+offset,v,width=.34,color=color,label=label)
    for xi,yi in zip(x+offset,v):ax[1].text(xi,yi+.08,f'{yi:.2f}',ha='center',fontsize=9)
ax[1].axhline(0,color='#697583',lw=.8);ax[1].set_xticks(x,['AllGT','Boundary','Large motion'])
ax[1].set_ylabel('AEE change vs native (%)');ax[1].set_ylim(-.35,3.35)
ax[1].set_title('R9: layer fixed before evaluation\n16 new members, same two sequences',fontsize=11);ax[1].legend(fontsize=8)
v=[f['regions']['all_valid']['CC_minus_fixed_AEE'] for f in r9['frames']]
colors=['#237a8d' if z>0 else '#c06b59' for z in v]
ax[2].bar(np.arange(16),v,color=colors);ax[2].axhline(0,color='#697583',lw=.8);ax[2].axvline(7.5,color='#697583',lw=.8,ls=':')
labels=[f['target_id'].rsplit('_',1)[1] for f in r9['frames']]
ax[2].set_xticks(np.arange(16),labels,rotation=60,fontsize=8);ax[2].set_xlabel('All 16 frame IDs: 01a (left) / 05a (right)')
ax[2].set_ylabel('AEE(CC) - AEE(fixed b2)');ax[2].set_title('R9: paired benefit, including failures\nPositive means fixed b2 improves CC',fontsize=11)
for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',alpha=.15);a.set_axisbelow(True)
fig.text(.01,.01,'Ordinary static controls, not a runtime damage predictor. Backbone-seen train-side sequences; no official-test or hardware-speedup claim.',fontsize=9,color='#4b5563')
fig.tight_layout(rect=(0,.07,1,1));fig.savefig(HERE/'round8_round9_quality.png',dpi=180);fig.savefig(HERE/'round8_round9_quality.pdf')
