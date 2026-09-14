"""Independent consumer/retirement conservation from final existing receipts."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parent;R=HERE.parent/'consumer_packed'

def main():
    out=[];checks=0
    for filename in ('results.json','results_64.json','results_full.json'):
        rows=json.loads((R/filename).read_text())
        for x in rows:
            assert x['identity_input']=='IEEE_binary32'
            F=x.get('tiles',1)
            expected=dict(outputs=3840*F,raw_outputs=3840*F,J_outputs=3840*F,source_load_words=1536*F,
                origin_words=F,output_beats=480*F,retired_tiles=F,consumer_raw_words=480*F,
                consumer_identity_words=480*F,consumer_coefficient_words=24*F,consumer_mul_issues=480*F,
                consumer_add_issues=960*F,consumer_round_issues=480*F,consumer_output_words=480*F,
                consumer_conversion_issues=480*F,
                consumer_cycles=3385*F+x['consumer_join_wait_cycles']+x['consumer_output_stalls'],
                total_cycles=x['consumer_cycles']+x['static_words']+x['parameter_stalls']+1536*F+F+x['source_load_stalls']+2*F+1)
            for field,y in expected.items():assert x[field]==y,(filename,field,x[field],y);checks+=1
            assert x['external_source_words']+x['padding_words']==1536*F;checks+=1
            assert x['static_words']==(1848 if x['command']==0 else 0);checks+=1
            if 'first_tile' in x:
                ext=0
                for tid in range(x['first_tile'],x['first_tile']+F):
                    oy=2*(tid//160)-1;ox=2*(tid%160)-1
                    ext+=96*sum(0<=oy+y<240 and 0<=ox+z<320 for y in range(4) for z in range(4))
                assert x['external_source_words']==ext;checks+=1
            if not x['stall']:
                assert x['core_output_stalls']==2419*F;checks+=1
                assert x['consumer_cycles']-x['core_cycles']==6*F;checks+=1
        out.append(dict(file=filename,jobs=len(rows),outputs_per_checkpoint=sum(r['outputs'] for r in rows)))
    result=dict(complete=True,reran_RTL=False,checks=checks,all_match=True,files=out)
    (HERE/'review_consumer_packed_counts.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
