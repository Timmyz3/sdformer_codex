"""Frozen four-policy prospective check; original quality only, scoped costs."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

B=Path(__file__).resolve().parent
P=B.parent
sources={
 'quality':P/'quality/actual_v1/SUMMARY_V1.json',
 'detail':P/'quality/actual_v1/original_EVALUATION_V1.json',
 'cost':P/'hardware_budget/actual_v2/ACTUAL_PREFIX_BUDGET_V1.json',
 'Q4':P.parents[1]/'flow_story_20261021/round33_predictability/hardware/results/SUMMARY_V1.json',
}
d={k:json.loads(p.read_text()) for k,p in sources.items()}
q=d['quality']['results']['original']
q4=next(r for r in d['Q4']['candidates'] if r['action']==6)
cost={'Q4':4*q4['static_affine_BN_ticks'],'static360':1032538860}
for name in ['folded128','ordinary128']:
 cost[name+'_fp32_fixed6']=float(np.mean([r['models'][name]['scoped_total'] for r in d['cost']['rows']]))
labels={'Q4':'Q4','static360':'Static 360','folded128_fp32_fixed6':'Response-selected 151',
        'ordinary128_fp32_fixed6':'Calibrated ordinary 151'}
colors=['#777777','#3e7598','#087f6d','#94619a']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                    'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
fig,(a,b)=plt.subplots(1,2,figsize=(11.7,4.7),gridspec_kw={'width_ratios':[1.25,1]})
points=[]
for i,(key,label) in enumerate(labels.items()):
 x,y=cost[key]/1e6,q[key]['AEE'];points.append(dict(policy=key,cost=cost[key],AEE=y))
 a.scatter(x,y,s=65,color=colors[i],marker='D' if i==2 else 'o',zorder=3)
 a.annotate(label,(x,y),xytext=(7,7) if i==0 else (-10,7),textcoords='offset points',
            ha='left' if i==0 else 'right',color=colors[i])
a.set(xlabel='Prefix + four isolated convs (M model ticks)',ylabel='All-valid-GT AEE',
      xlim=(810,1060),ylim=(1.452,1.526))
a.grid(alpha=.18);a.set_title('Frozen quality / modeled service',loc='left',fontweight='bold')
summ=d['detail']['summaries'];fold='folded128_fp32_fixed6'
seqs=[s for s in summ[fold] if s!='overall']
delta=[summ[fold][s]['all_valid']['pixel_weighted_AEE']-
       summ['static360'][s]['all_valid']['pixel_weighted_AEE'] for s in seqs]
b.barh(range(len(seqs)),delta,color=['#087f6d' if x<0 else '#c28135' for x in delta])
b.set_yticks(range(len(seqs)),[s.replace('zurich_city_','Zurich ').replace('thun_','Thun ') for s in seqs])
b.invert_yaxis();b.axvline(0,color='#555',linewidth=.8)
for i,x in enumerate(delta):
 b.text(x+(.001 if x>=0 else -.001),i,f'{x:+.4f}',ha='left' if x>=0 else 'right',va='center',fontsize=9)
b.set_xlim(min(delta)-.035,max(delta)+.02)
b.set_xlabel('Candidate minus static360 AEE (px)')
b.set_title('3/5 sequences improve',loc='left',fontweight='bold')
fig.suptitle('R38: a frozen candidate survives new identities at the paid budget',
             x=.065,ha='left',fontsize=13,fontweight='bold')
fig.text(.065,.032,'20 new identities in 5 previously used sequences; original outputs. Shared stem and rest of network excluded.',fontsize=8.5)
fig.text(.065,.005,'All five leave-one-sequence comparisons favor the candidate vs static360. Large-motion AEE is nearly tied; one high-support tail remains.',fontsize=8.5)
fig.tight_layout(rect=(0,.085,1,.93))
for ext in ['png','pdf','svg']:
 p=B/('QUALITY_COST_V1.'+ext);assert not p.exists();fig.savefig(p,dpi=180,bbox_inches='tight')
plt.close(fig)
out=B/'FIGURE_SOURCES_V1.json';assert not out.exists()
out.write_text(json.dumps(dict(points=points,sequence_deltas=dict(zip(seqs,delta)),
 sources={k:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for k,p in sources.items()},
 script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
 cost_is_identity_mean_not_pixel_weighted=True,quality_is_pixel_weighted=True,
 new_sequences=False,whole_network_speed_claim=False),indent=2)+'\n')
print('R38_FIGURE_SAVED',points)
