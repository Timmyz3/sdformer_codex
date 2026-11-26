#!/usr/bin/env python3
"""All twenty recovery-state/layout combinations; descriptive, no ranking selection."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--analysis',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();z=json.loads(a.analysis.read_text());s=z['summaries']
    assert z['schema']=='flow_round12_fixed_quality_analysis_v2'
    states=('initial','native','FF','CC','fine_s2b2');modes=states[1:]
    fig,axes=plt.subplots(1,3,figsize=(15,6.3))
    for ax,region in zip(axes,('all_valid','B','L')):
        ref=s['initial/native']['overall'][region]['frame_mean_AEE']
        values=np.array([[s[state+'/'+mode]['overall'][region]['frame_mean_AEE'] for mode in modes] for state in states])
        relative=100*(values/ref-1)
        bound=max(float(np.abs(relative).max()),.1)
        im=ax.imshow(relative,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
        ax.set_xticks(range(4),['native','FF','CC','block2 fine'],rotation=30,ha='right')
        ax.set_yticks(range(5),['initial','trained native','trained FF','trained CC','trained block2'])
        ax.set_xlabel('Evaluation layout')
        ax.set_title({'all_valid':'All valid GT','B':'Flow-jump boundary proxy','L':'Large displacement'}[region])
        for y in range(5):
            for x in range(4):
                ax.text(x,y,'%.4f\n%+.2f%%'%(values[y,x],relative[y,x]),ha='center',va='center',fontsize=9,
                        color='white' if abs(relative[y,x])>bound*.65 else 'black')
        fig.colorbar(im,ax=ax,shrink=.72,label='Change from initial native (%)')
    fig.suptitle('R12: all initial and recovered checkpoints across all four layouts\n'
                 'Labels: frame-mean AEE and change from initial native; lower is better',fontsize=13)
    fig.text(.5,.02,'8 train-side members; 20-fit-member / 200-update ordinary recovery pilot. No official-test or hardware-speedup claim.',
             ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.045,1,.91))
    a.output_dir.mkdir(parents=True,exist_ok=True)
    for ext in ('png','pdf'):fig.savefig(a.output_dir/('all_states_all_layouts.'+ext),dpi=170)
    plt.close(fig)
    # Each sequence is retained in a separate complete matrix.
    seqs=[q for q in s['initial/native'] if q!='overall']
    fig,axes=plt.subplots(1,2,figsize=(11,6.1))
    for ax,seq in zip(axes,seqs):
        ref=s['initial/native'][seq]['all_valid']['frame_mean_AEE']
        values=np.array([[s[state+'/'+mode][seq]['all_valid']['frame_mean_AEE'] for mode in modes] for state in states])
        relative=100*(values/ref-1);bound=max(float(np.abs(relative).max()),.1)
        im=ax.imshow(relative,cmap='RdBu_r',vmin=-bound,vmax=bound,aspect='auto')
        ax.set_xticks(range(4),['native','FF','CC','block2 fine'],rotation=30,ha='right')
        ax.set_yticks(range(5),['initial','trained native','trained FF','trained CC','trained block2'])
        ax.set_title(seq.replace('zurich_city_','')+' : all valid GT')
        for y in range(5):
            for x in range(4):
                ax.text(x,y,'%.4f\n%+.2f%%'%(values[y,x],relative[y,x]),ha='center',va='center',fontsize=9,
                        color='white' if abs(relative[y,x])>bound*.65 else 'black')
        fig.colorbar(im,ax=ax,shrink=.75,label='Change from initial native (%)')
    fig.suptitle('R12: both historical sequences retained, four members each')
    fig.tight_layout(rect=(0,0,1,.94))
    for ext in ('png','pdf'):fig.savefig(a.output_dir/('both_sequences_all_layouts.'+ext),dpi=170)
    plt.close(fig)
    (a.output_dir/'plot_receipt.json').write_text(json.dumps({'analysis_sha256':hashlib.sha256(a.analysis.read_bytes()).hexdigest(),
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Descriptive all-combination plots; the frozen V2 gates use additional matched/self references and per-sequence B/L.',
        'files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.output_dir.iterdir()) if p.suffix in ('.png','.pdf')}},indent=2)+'\n')


if __name__=='__main__':main()
