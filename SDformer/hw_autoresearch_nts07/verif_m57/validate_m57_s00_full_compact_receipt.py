#!/usr/bin/env python3
"""Fail-closed validation for the M57 S00 full compact VCS receipt."""
import argparse, hashlib, json, re
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def req(x,m):
    if not x: raise ValueError(m)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--repo',type=Path,required=True); ap.add_argument('--receipt',type=Path,required=True); a=ap.parse_args(); repo=a.repo.resolve()
    cp=repo/'hw_autoresearch_nts07/contracts/m57_s00_phase_safe_full_compact_exact_sha_vcs_contract_r2_20260823.json'; c=json.loads(cp.read_text()); r=json.loads(a.receipt.read_text()); run=repo/c['paths']['production_run']; exp=c['expected']
    req(r['contract_sha256']==sha(cp),'contract identity drift'); req(r['status']=='PASS_EXACT_SHA_PHASE_SAFE_FULL_S00_VCS_COMPACT_REPLAY','bad status')
    paths={'simv':repo/c['paths']['compile_run']/'simv','uncompressed_schedule_stream':repo/'hw_autoresearch_nts07/dc_handoff/runs/m57_diagnostics_20260823/s00_sim_r2/input.bin','schedule_manifest':repo/'hw_autoresearch_nts07/results/m57_h67_k4c16_temporal_vcs_r1_20260823/m57_s00_schedule_manifest.json','rtl':repo/'hw_autoresearch_nts07/rtl_m57/qfit_m57_m53_schedule_bridge.sv','testbench':repo/'hw_autoresearch_nts07/tb_m57/tb_m57_m53_schedule_bridge.sv','assertions':repo/'hw_autoresearch_nts07/verif_m54/qfit_k4_parent_delta_p8_l96_ctx16_assertions.sv','filelist':repo/'hw_autoresearch_nts07/dc_handoff/filelists/date_m57_m53_schedule_bridge_vcs.f','compile_raw_log':repo/c['paths']['compile_run']/'compile.raw.log','vcs_launcher_binary':Path('/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs'),'compact_replayer':repo/'hw_autoresearch_nts07/verif_m57/replay_m57_handshake_ledger.py'}
    for k,p in paths.items(): req(p.is_file() and sha(p)==c['exact_sha256'][k],k+' SHA drift')
    req(int((run/'sim.rc').read_text())==0 and int((run/'gzip.rc').read_text())==0,'nonzero production rc')
    rep=r['functional_and_protocol']; req(rep['accepted_requests']==rep['accepted_responses']==exp['source_issue_cycles'],'request/response total drift'); req(rep['accepted_outputs']==exp['descriptor_commands'],'output total drift'); req(rep['functional_mismatch_count']==0 and rep['metadata_fifo_final_occupancy']==0,'functional/FIFO failure'); req(rep['event_lines']>exp['source_issue_cycles'],'event population incomplete')
    req(0<=rep['maximum_metadata_occupancy']<=16 and 0<=rep['maximum_context_occupancy']<=16 and 0<=rep['maximum_complete_occupancy']<=16,'occupancy bound failure')
    log=(run/'sim.raw.log').read_text(errors='replace'); req(len(re.findall(r'^PASS M57 S0 ',log,re.M))==1,'missing unique PASS'); req('Fatal:' not in log and 'M54_ASSERTION_MODULE_ACTIVE=1' in log,'fatal or SVA inactive')
    phase=rep['launch_phase']; req(phase['prelaunch_artificial_bubbles']==0,'artificial prelaunch bubble remained'); req(phase['direct_groups']+phase['aligned_groups']==exp['fusion_groups'],'launch phase group conservation failure')
    req(r['identity']['ledger_gzip_sha256']==sha(run/'m57_s00_handshake_ledger.compact.log.gz'),'ledger SHA drift'); req(r['claim_boundary']['system_speedup_admitted'] is False,'system speedup boundary widened'); req((repo/c['paths']['fifo_deadlock_run']/'FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt').is_file(),'FIFO predecessor unsealed'); req((repo/c['paths']['old_tb_full_sim_run']/'FAILED_OR_INCOMPLETE_DO_NOT_CITE.txt').is_file(),'old TB predecessor unsealed')
    print('PASS M57 S00 exact-SHA full compact receipt validator')
if __name__=='__main__':
    try: main()
    except Exception as e: print('FAIL M57 validator: {}'.format(e)); raise SystemExit(1)
