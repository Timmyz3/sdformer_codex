#!/usr/bin/env python3
"""Standalone exploratory scientific figure; use complete frozen analysis only."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',type=Path,required=True)
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output-prefix',type=Path,required=True)
    a=p.parse_args();raw=json.loads(a.raw.read_text());analysis=json.loads(a.analysis.read_text())
    if raw['status']!='TILE_INTERVENTIONS_DEV8_COMPLETE':raise ValueError('Incomplete raw input')
    fig,axs=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
    seqs=sorted({f['sequence'] for f in raw['frames']}); colors=['#26749a','#d27636']
    for seq,color in zip(seqs,colors):
        xx=[];yy=[]
        for f in raw['frames']:
            if f['sequence']!=seq:continue
            base=f['arms']['coarse_stage02']['metrics']['regions']['all_valid']['AEE']
            for item in f['offline_fine_coarse_branch_mse']:
                r,c=item['tile'];name=f'fine_r{r}_c{c}'
                val=f['arms'][name]['metrics']['regions']['all_valid']['AEE']
                xx.append(item['MSE']);yy.append(100*(base-val)/base)
        axs[0].scatter(xx,yy,s=25,color=color,alpha=.75,label=seq.replace('zurich_city_',''))
    axs[0].set_xscale('log');axs[0].axhline(0,color='.5',lw=.8)
    axs[0].set_xlabel('Offline fine/coarse branch MSE')
    axs[0].set_ylabel('All-GT AEE improvement from one fine tile (%)')
    axs[0].set_title('(a) Feature change versus actual task effect',loc='left',fontsize=11)
    axs[0].legend(title='Development sequence',frameon=False,fontsize=8)
    names=['LOSO_task_ridge','LOSO_branch_mse_ridge','LOSO_position_task_mean']
    labels=['Task-target\nridge','Feature-target\nridge','Position-only\ntask table']
    base=analysis['k1_summaries']['coarse_baseline']
    for i,(seq,color) in enumerate(zip(seqs,colors)):
        values=[]
        for name in names:
            b=base['per_sequence'][seq]['all_valid']['frame_mean_AEE']
            v=analysis['k1_summaries'][name]['per_sequence'][seq]['all_valid']['frame_mean_AEE']
            values.append(100*(b-v)/b)
        axs[1].bar(np.arange(3)+(i-.5)*.34,values,width=.32,color=color,label=seq.replace('zurich_city_',''))
    axs[1].set_xticks(np.arange(3),labels);axs[1].axhline(0,color='.5',lw=.8)
    axs[1].set_ylabel('All-GT AEE improvement versus coarse (%)')
    axs[1].set_title('(b) Opposite-sequence training, forced k=1',loc='left',fontsize=11)
    axs[1].legend(title='Check sequence',frameon=False,fontsize=8)
    for ax in axs:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Exploratory dev8 only: 96 interventions, 8 frames, 2 sequences',fontsize=12)
    a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png'):
        out=a.output_prefix.with_suffix('.'+ext)
        if out.exists():raise ValueError('Preserve existing figure')
        fig.savefig(out,dpi=220,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()
