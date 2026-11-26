#!/usr/bin/env python3
"""Copy exactly the eight fixed audit inputs as opaque bytes; no quality analysis."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time
sys.dont_write_bytecode=True
from prepare_common_state import prepare,sha,require,write

def main():
    p=argparse.ArgumentParser()
    for key in ['runtime','membership']:
        p.add_argument('--'+key,type=Path,required=True);p.add_argument('--'+key+'-sha256',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();runtime,members,base,b,context,roles=prepare(a)
    native,_,_,_,hashes,*_=context
    a.output_dir.mkdir(parents=True,exist_ok=False)
    entries=[]
    with tarfile.open(str(a.output_dir/'inputs.tar'),'w') as tar:
        for row in members['check_members']:
            for key,relative in native.target_files(row).items():
                rel=Path(relative);require(not rel.is_absolute() and '..' not in rel.parts,'Unsafe relative path')
                path=native.checked_path(b.data_root,relative)
                actual=sha(path);expected=hashes[relative]['sha256'];require(actual==expected,'Input SHA mismatch')
                entries.append({'target_id':row['target_id'],'kind':'events' if key=='input' else key,
                                'relative_path':relative,'expected_sha256':expected,'sha256':actual,'bytes':path.stat().st_size})
                tar.add(str(path),arcname=relative,recursive=False)
    require(len(entries)==24,'Exactly24 fixed input arrays')
    result={'schema':'flow_r13_input_snapshot_v1','utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            'source_root':str(b.data_root),'all_files':entries,'archive_bytes':(a.output_dir/'inputs.tar').stat().st_size,
            'archive_sha256':sha(a.output_dir/'inputs.tar'),'membership_sha256':a.membership_sha256,
            'script_sha256':sha(__file__),'scope':'Opaque bytes from fixed check8; no numerical GT/events inspection or model inference'}
    write(a.output_dir/'receipt.json',result);print(json.dumps({'status':'SNAPSHOT_COMPLETE','files':24,'receipt_sha256':sha(a.output_dir/'receipt.json')}))

if __name__=='__main__':main()
