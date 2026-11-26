#!/usr/bin/env python3
"""One bounded JSON-only observer for recovery_20261013; default is plan only.

Uses an existing SSH master, never starts one, and never reads weights or data.
It launches only the pinned local CPU summarizer. It does not
inspect, signal, resume or alter training/GPU processes. Review before --run.
"""
import argparse
import base64
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
RUN = '/data/flow_a/runs/recovery_20261013'
SUMMARIZER_SHA = 'a8810ee38768896281c2d36ffe993be15846cc18a243ab55bcdf1f32ad8dcf6f'
LOCAL = HERE/'results_20261013'
ARMS = ('dense', 'lowrank_param25', 'svs_hidden25', 'magnitude_hidden25')
FILES = ('queue_status.json', 'queue_launch.json', 'baseline_yield.json',
         'baseline_remainder_launch.json', 'smoke_audit.json', 'svs_stability.json') + tuple(
         a+'_'+stage+'_eval.json' for a in ARMS for stage in ('initial', 'endpoint')) + tuple(
         'arms/'+a+'/'+name for a in ARMS for name in ('wall_budget.json', 'segment_result.json'))
TERMINAL = ('COMPLETE_BASELINE_REMAINDER_LAUNCHED',
            'EXPERIMENT_INCOMPLETE_BASELINE_REMAINDER_LAUNCHED',
            'NEEDS_OPERATOR_BASELINE_RESUME_NOT_CONFIRMED')
MAX_FILE = 16*1024*1024
MAX_RAW = 64*1024*1024
MAX_WIRE = 96*1024*1024
INTERVAL = 60
SSH_TIMEOUT = 30
MAX_SECONDS = 6*3600
FAIL_LIMIT = 3

# Fixed code, sent over an already-established SSH transport. The only request
# field is the local SHA map; neither remote paths nor commands come from it.
REMOTE_CODE = r'''
import base64,datetime,hashlib,json,pathlib,sys
RUN='/data/flow_a/runs/recovery_20261013'
ARMS=('dense','lowrank_param25','svs_hidden25','magnitude_hidden25')
FILES=('queue_status.json','queue_launch.json','baseline_yield.json','baseline_remainder_launch.json','smoke_audit.json','svs_stability.json')+tuple(a+'_'+s+'_eval.json' for a in ARMS for s in ('initial','endpoint'))+tuple('arms/'+a+'/'+n for a in ARMS for n in ('wall_budget.json','segment_result.json'))
root=pathlib.Path(RUN)
assert root.is_dir() and not root.is_symlink(),'Fixed run missing or symlink'
request=json.load(sys.stdin)
assert set(request)=={'known_sha256'} and isinstance(request['known_sha256'],dict),'Unexpected request'
known=request['known_sha256']
assert set(known)<=set(FILES) and all(isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdef' for c in v) for v in known.values()),'Invalid local SHA map'
files={};missing=[];total=0;queue=None
for rel in FILES:
    path=root/rel
    current=path
    while current!=root:
        assert not current.is_symlink(),'Symlink refused: '+rel
        current=current.parent
    if not path.exists():
        missing.append(rel);continue
    assert path.is_file() and path.stat().st_size<=16*1024*1024,'Oversize/not-file JSON: '+rel
    raw=path.read_bytes()
    assert len(raw)<=16*1024*1024,'Growing oversize JSON: '+rel
    value=json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    assert isinstance(value,dict),'Non-object JSON: '+rel
    sha=hashlib.sha256(raw).hexdigest()
    record={'sha256':sha,'bytes':len(raw)}
    if rel=='queue_status.json':
        assert value.get('schema')=='flow_recovery_queue_v1','Foreign queue schema'
        queue={'status':value.get('status'),'experiment_status':value.get('experiment_status'),'active_job':value.get('active_job')}
    if known.get(rel)!=sha:
        total+=len(raw)
        assert total<=64*1024*1024,'Snapshot exceeds bounded transfer'
        record['base64']=base64.b64encode(raw).decode('ascii')
    files[rel]=record
print(json.dumps({'schema':'flow_recovery_json_transport_v1','run_directory':RUN,'remote_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'files':files,'missing':missing,'queue':queue},separators=(',',':')))
'''


def require(ok, message):
    if not ok: raise ValueError(message)


def now(): return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(raw): return hashlib.sha256(raw).hexdigest()


def json_object(raw):
    value=json.loads(raw,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    require(isinstance(value,dict),'Expected JSON object')
    return value


def local_path(relative):
    require(relative in FILES,'Path outside fixed JSON allowlist')
    path=LOCAL/relative; current=path
    while current!=LOCAL:
        require(not current.is_symlink(),'Local symlink refused: '+relative)
        current=current.parent
    return path


def local_hashes():
    require(not LOCAL.is_symlink(),'Local results directory may not be a symlink')
    result={}
    for relative in FILES:
        path=local_path(relative)
        if path.exists():
            require(path.is_file() and path.stat().st_size<=MAX_FILE,'Invalid local JSON artifact')
            raw=path.read_bytes();json_object(raw);result[relative]=sha(raw)
    return result


def atomic_bytes(path, raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    require(not path.is_symlink(),'Refuse symlink output')
    temp=path.with_name(path.name+'.watch_tmp_'+str(os.getpid()))
    with temp.open('xb') as f:
        f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def save_status(value):
    atomic_bytes(LOCAL/'watch_status.json',(json.dumps(value,indent=2,allow_nan=False)+'\n').encode())


def validate_packet(raw, known):
    require(len(raw)<=MAX_WIRE,'Oversize SSH response')
    value=json_object(raw)
    require(value.get('schema')=='flow_recovery_json_transport_v1' and value.get('run_directory')==RUN,
            'Remote snapshot identity differs')
    files=value['files'];missing=value['missing']
    require(isinstance(files,dict) and isinstance(missing,list) and len(missing)==len(set(missing)) and
            not set(files)&set(missing) and set(files)|set(missing)==set(FILES),'Incomplete/foreign path inventory')
    require(not set(missing)&set(known),'Previously synced evidence missing remotely; preserve previous local snapshot')
    changes={};total=0
    for relative,record in files.items():
        require(relative in FILES and set(record)<= {'sha256','bytes','base64'} and
                type(record.get('bytes')) is int and 0<=record['bytes']<=MAX_FILE,'Invalid artifact record')
        if 'base64' in record:
            data=base64.b64decode(record['base64'],validate=True)
            require(len(data)==record['bytes'] and sha(data)==record['sha256'],'Transport hash/length mismatch')
            json_object(data);changes[relative]=data;total+=len(data)
        else:
            require(known.get(relative)==record.get('sha256'),'Omitted changed artifact bytes')
    require(total<=MAX_RAW,'Too much changed JSON evidence')
    queue=value.get('queue')
    if queue is not None:
        require(isinstance(queue,dict) and 'queue_status.json' in files,'Queue metadata lacks source')
        qraw=changes.get('queue_status.json')
        if qraw is None:qraw=local_path('queue_status.json').read_bytes()
        q=json_object(qraw)
        require(q.get('schema')=='flow_recovery_queue_v1' and
                all(queue.get(k)==q.get(k) for k in ('status','experiment_status','active_job')),'Queue metadata/source mismatch')
    return value,changes


def summarize_local():
    path=HERE/'summarize_recovery.py'
    require(sha(path.read_bytes())==SUMMARIZER_SHA,'Local summarizer pin differs')
    spec=importlib.util.spec_from_file_location('flow_json_watch_summary',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    report=module.summarize(LOCAL)
    out=LOCAL/'summary'
    module.atomic(out/'recovery_summary.json',json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    module.atomic(out/'RECOVERY_SUMMARY.md',module.markdown(report))
    return {'status':report['status'],'completion':report['completion']}


def poll(control_path):
    known=local_hashes()
    # ProxyCommand=false prevents a new direct connection if the existing master
    # disappears. No password, private key or reusable master is stored/created.
    argv=['ssh','-F','/dev/null','-S',control_path,'-o','ControlMaster=no','-o','BatchMode=yes',
          '-o','ProxyCommand=false','-o','ConnectTimeout=10','-o','ServerAliveInterval=10',
          '-o','ServerAliveCountMax=1','-p','30449','root@36.213.75.115',
          'python3 -c '+shlex.quote(REMOTE_CODE)]
    result=subprocess.run(argv,input=json.dumps({'known_sha256':known}).encode(),
        stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=SSH_TIMEOUT)
    require(result.returncode==0,'SSH/remote read failed: '+result.stderr[-1000:].decode(errors='replace'))
    packet,changes=validate_packet(result.stdout,known)
    # Validate the whole packet before replacing any local evidence.
    for relative,raw in changes.items():atomic_bytes(local_path(relative),raw)
    summary=summarize_local()
    return {'remote_utc':packet['remote_utc'],'queue':packet['queue'],
            'changed_files':list(changes),'changed_bytes':sum(map(len,changes.values())),
            'remote_missing_files':packet['missing'],'snapshot_sha256':{k:v['sha256'] for k,v in packet['files'].items()},
            'local_summary':summary}


def watch(control_path):
    LOCAL.mkdir(parents=True,exist_ok=True)
    require(not LOCAL.is_symlink(),'Local results directory may not be a symlink')
    require(sha((HERE/'summarize_recovery.py').read_bytes())==SUMMARIZER_SHA,'Local summarizer pin differs')
    with (LOCAL/'.watch.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        require(not (LOCAL/'watch_status.json').exists(),'One-shot watcher already has status; inspect it before a new run')
        start=time.monotonic();deadline=start+MAX_SECONDS
        status={'schema':'flow_recovery_watch_v1','status':'WATCHING','started_utc':now(),'pid':os.getpid(),
            'fixed_remote_run':RUN,'fixed_local_results':str(LOCAL),'interval_seconds':INTERVAL,
            'max_seconds':MAX_SECONDS,'ssh_timeout_seconds':SSH_TIMEOUT,'max_consecutive_failures':FAIL_LIMIT,
            'polls':0,'consecutive_failures':0,'successful_syncs':0,'last_sync_utc':None,'last_remote_utc':None,
            'remote_terminal_observed':False,'watcher_sha256':sha(Path(__file__).read_bytes()),
            'synced_file_allowlist':list(FILES),'touches_training_or_GPU':False}
        save_status(status)
        try:
            while True:
                if deadline-time.monotonic()<SSH_TIMEOUT:
                    status['status']='TIMED_OUT';break
                tick=time.monotonic();status['polls']+=1
                try:
                    update=poll(control_path)
                    status.update(last_sync_utc=now(),last_remote_utc=update['remote_utc'],last_snapshot=update,
                        successful_syncs=status['successful_syncs']+1,consecutive_failures=0)
                    queue=update['queue'] or {}
                    if queue.get('status') in TERMINAL:
                        status.update(status='TERMINAL_SYNCED',remote_terminal_observed=True,terminal_queue_status=queue['status'])
                        save_status(status);break
                except (OSError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired) as exc:
                    status.update(last_error=type(exc).__name__+': '+str(exc),last_error_utc=now(),
                                  consecutive_failures=status['consecutive_failures']+1)
                    if status['consecutive_failures']>=FAIL_LIMIT:
                        status['status']='FAILED_CONSECUTIVE_POLLS';save_status(status);break
                save_status(status)
                print(json.dumps({'watch_status':status['status'],'polls':status['polls'],
                    'successful_syncs':status['successful_syncs'],'last_sync_utc':status['last_sync_utc'],
                    'consecutive_failures':status['consecutive_failures']}),flush=True)
                time.sleep(max(0,min(INTERVAL-(time.monotonic()-tick),deadline-time.monotonic())))
        except BaseException as exc:
            status.update(status='INTERRUPTED' if isinstance(exc,KeyboardInterrupt) else 'FAILED_OBSERVER',
                          last_error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            status.update(finished_utc=now(),elapsed_seconds=time.monotonic()-start)
            save_status(status)
        return status


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--control-path',required=True,help='Existing SSH ControlPath; no new master is created')
    p.add_argument('--run',action='store_true')
    a=p.parse_args(argv)
    require(a.control_path.startswith('/tmp/') and '\x00' not in a.control_path,'Expected existing /tmp SSH ControlPath')
    if not a.run:
        print(json.dumps({'status':'WATCHER_PLAN_ONLY_NOT_STARTED','remote_run':RUN,'local_results':str(LOCAL),
            'files':FILES,'max_seconds':MAX_SECONDS,'ssh_timeout_seconds':SSH_TIMEOUT,
            'interval_seconds':INTERVAL,'local_summarizer_sha256':SUMMARIZER_SHA,
            'no_weights_GT_events_GPU_or_training_control':True},indent=2));return 0
    def interrupted(*_):raise KeyboardInterrupt('Watcher stopped by operator')
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    result=watch(a.control_path)
    print(json.dumps({'status':result['status'],'last_sync_utc':result['last_sync_utc'],
        'terminal_queue_status':result.get('terminal_queue_status')}))
    return 0 if result['status']=='TERMINAL_SYNCED' else 1


if __name__=='__main__':raise SystemExit(main())
