#!/usr/bin/env python3
"""Bounded directory-name discovery only; never reads research-file contents."""
import json
import os
from pathlib import Path
import socket

HERE=Path(__file__).resolve().parent
roots=['/data','/datasets','/home/zhumd/data','/mnt']
out={'hostname':socket.gethostname(),'scope':'Directory names only, depth <=4; no symlink traversal; general_snn_codesign pruned. Negative results do not prove machine-wide absence.', 'roots':{}}
for root in roots:
    p=Path(root)
    r={'exists':p.exists(),'is_directory':p.is_dir(),'top_level':[],
       'dsec_named_directories':[],'scan_errors':[]}
    out['roots'][root]=r
    if not p.is_dir():continue
    try:
        r['top_level']=[{'name':x.name,'is_dir':x.is_dir(),'symlink':x.is_symlink()} for x in sorted(p.iterdir())][:100]
    except OSError as e:
        r['scan_errors'].append(str(e));continue
    def record_error(e):r['scan_errors'].append(str(e))
    for here,dirs,files in os.walk(str(p),onerror=record_error,followlinks=False):
        depth=len(Path(here).relative_to(p).parts)
        dirs[:]=sorted(x for x in dirs if x!='general_snn_codesign' and not x.startswith('.'))
        if 'dsec' in Path(here).name.lower():r['dsec_named_directories'].append(here)
        if depth>=4:dirs[:]=[]
path=HERE/'data_root_discovery.json'
path.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,indent=2)+'\n')
print(json.dumps({'hostname':out['hostname'],'output':str(path),
      'matches':{k:v['dsec_named_directories'] for k,v in out['roots'].items()},
      'scope':out['scope']},indent=2))
