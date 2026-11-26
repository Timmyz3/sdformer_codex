import json,os
os.chdir(os.path.expanduser('~/work/sdformer_codex/SDformer/hw_autoresearch_nts07/results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903'))
print(open('M2057_RUN_COMMANDS.txt').read()[:3000])
d=json.load(open('result.json'))
def show(d,pre='',depth=0):
    for k,v in d.items():
        if isinstance(v,dict) and depth<2 and len(v)<40:
            print(pre+str(k)+':'); show(v,pre+'  ',depth+1)
        else:
            s=json.dumps(v); print(pre+str(k)+': '+s[:400])
show(d)
