#!/usr/bin/env python3
"""Full fixed input-by-supervision contrast; all controls stay in analysis table."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
raw=json.loads((HERE/'analysis_v1/analysis.json').read_text())
summaries=raw['check_summaries']
fig,axes=plt.subplots(1,3,figsize=(12,4.5))
families=('B5','rich37','PCA11','flow11');x=np.arange(4)
for ax,region,title in zip(axes,('all_valid','B','L'),('All valid GT','Motion boundary','Large displacement')):
    base=summaries['coarse_baseline']['regions'][region]['frame_mean_AEE']
    for offset,target,color in ((-.19,'MSE','#a7b4c5'),(.19,'task','#237a8d')):
        values=[100*(summaries[f+'_'+target]['regions'][region]['frame_mean_AEE']/base-1) for f in families]
        ax.bar(x+offset,values,width=.36,color=color,label=target)
    h=100*(summaries['psn1_spatial_neighbor_hamming']['regions'][region]['frame_mean_AEE']/base-1)
    ax.axhline(0,color='#4b5563',lw=.8,label='CC baseline')
    ax.axhline(h,color='#ad4537',lw=1.2,ls='--',label='PSN Hamming')
    ax.set_xticks(x,families);ax.set_title(title);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('AEE change versus CC (%) — lower is better')
axes[0].legend(fontsize=8)
fig.suptitle('R10: fixed inputs × supervision at stage2/block2, actual k1 endpoints',fontsize=12)
fig.text(.01,.01,'8 identity-new members in 2 historically exposed sequences. No native-relative gate or hardware-speedup claim. All 19 rules reported in ANALYSIS.md.',fontsize=8.5,color='#4b5563')
fig.tight_layout(rect=(0,.07,1,.96))
fig.savefig(HERE/'analysis_v1/input_supervision.png',dpi=180)
fig.savefig(HERE/'analysis_v1/input_supervision.pdf')
