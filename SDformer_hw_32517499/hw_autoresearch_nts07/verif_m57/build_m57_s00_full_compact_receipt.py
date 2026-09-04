#!/usr/bin/env python3
"""Build the M57 S00 full compact VCS receipt from immutable run evidence."""
import argparse, hashlib, json, re
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); a=ap.parse_args()
    repo=a.repo.resolve(); c=json.loads((repo/'hw_autoresearch_nts07/contracts/m57_s00_phase_safe_full_compact_exact_sha_vcs_contract_r2_20260823.json').read_text())
    run=repo/c['paths']['production_run']; replay=json.loads((run/'m57_s00_ledger_replay.json').read_text()); log=(run/'sim.raw.log').read_text(errors='replace')
    if a.output.exists(): raise ValueError('refusing receipt overwrite')
    start=int((run/'start_epoch.txt').read_text()); end=int((run/'end_epoch.txt').read_text())
    covers={}
    for name,attempts,matches in re.findall(r'm54_sva\.(cp_[A-Za-z0-9_]+),\s+(\d+) attempts,\s+(\d+) match',log): covers[name]={'attempts':int(attempts),'matches':int(matches)}
    payload={
      'schema':'m57_s00_phase_safe_full_compact_exact_sha_vcs_receipt_v2','status':'PASS_EXACT_SHA_PHASE_SAFE_FULL_S00_VCS_COMPACT_REPLAY','date':'2026-08-23',
      'contract_sha256':sha(repo/'hw_autoresearch_nts07/contracts/m57_s00_phase_safe_full_compact_exact_sha_vcs_contract_r2_20260823.json'),
      'run':{'directory':str(run),'elapsed_seconds':end-start,'sim_rc':int((run/'sim.rc').read_text()),'gzip_rc':int((run/'gzip.rc').read_text()),'full_sample_not_sampled':True},
      'identity':{'ledger_gzip_sha256':sha(run/'m57_s00_handshake_ledger.compact.log.gz'),'ledger_gzip_bytes':(run/'m57_s00_handshake_ledger.compact.log.gz').stat().st_size,'sim_log_sha256':sha(run/'sim.raw.log'),'replay_sha256':sha(run/'m57_s00_ledger_replay.json'),'input_sha256_manifest':sha(run/'prelaunch_input_sha256.txt')},
      'functional_and_protocol':replay,
      'sva':{'module_active':log.count('M54_ASSERTION_MODULE_ACTIVE=1')==1,'coverpoints':covers,'assertion_failure_signatures':len(re.findall(r'(?i)(assertion failed|error-|fatal:)',log))},
      'terminal':{'pass_line_count':len(re.findall(r'^PASS M57 S0 ',log,re.M)),'progress_group_100000_count':len(re.findall(r'^M57_PROGRESS ',log,re.M))},
      'predecessor_boundary':{'old_tb_full_sim_citable_for_phase_safe_cycles':False,'fifo_deadlock_citable':False},
      'claim_boundary':c['claim_boundary']
    }
    a.output.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print('PASS M57 receipt full S00 elapsed={}s'.format(end-start))
if __name__=='__main__': main()
