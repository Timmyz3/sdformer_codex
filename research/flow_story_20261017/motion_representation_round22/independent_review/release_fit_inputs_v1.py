#!/usr/bin/env python3
"""Release only16 fit GT/mask paths after full native audit; no array loads."""
import argparse
from pathlib import Path
import io_common as c

def release(state,freeze_sha):
    rows=[]
    for i,row in enumerate(state['members']):
        tid=row['target_id'];path,r=state['native'][tid]
        out=dict(target_id=tid,sequence=row['sequence'],role='fit' if i<16 else 'development',native_archive=dict(path=str(path.resolve()),sha256=c.sha(path)))
        if i<16:
            for name,kind in (('fit_GT','gt'),('fit_mask','mask')):
                p=state['paths'][tid][kind];out[name]=dict(path=str(p.resolve()),sha256=state['evidence'][tid][kind])
        rows.append(out)
    c.require(sum('fit_GT' in r for r in rows)==16 and all('fit_GT' not in r and 'fit_mask' not in r for r in rows[16:]),'Onlyfit GT/mask release')
    return dict(schema='flow_r22_independent_fit_only_release_v1',status='FIT16_LABEL_PATHS_ONLY_DEV_CLOSED',graph_id=c.GRAPH,
      roles_sha256=c.ROLE_SHA,source_result_sha256=c.sha(state['result_path']),execution_audit_sha256=c.sha(state['audit_path']),IO_freeze_sha256=freeze_sha,
      members=rows,fit_GT_references=16,fit_mask_references=16,development_GT_references=0,sealed_data_references=0,GT_array_values_loaded=False,
      scope='Source-bound fit-only paths; not a quality evaluation. Development GT metrics require frozen projection predictions.')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',type=Path,required=True);p.add_argument('--freeze-sha256',required=True)
    p.add_argument('--result-dir',type=Path,required=True);p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    f=c.preflight(a.freeze,a.freeze_sha256);state=c.source_admission(f,a.result_dir,a.input_root);c.write(a.output,release(state,a.freeze_sha256));print('FIT16_PATHS_RELEASED '+c.sha(a.output))
