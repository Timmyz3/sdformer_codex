#!/usr/bin/env python3
"""Read-only archive -> dedicated A100 DSEC transfer, stdlib only.

Default: write a deterministic plan, NO SSH. Add --run for transfer, --smoke
for one ~60 MiB prefix batch. --self-test uses temporary local directories only.
SSH endpoints must have existing multiplex sockets; no password is used.
With --target-local, run on the destination host and receive locally, avoiding
an intermediate SSH relay. --source-csv relocates the audited CSV only; its
original SHA256 in --audit remains mandatory.
With --source-local, run on the archive host and push to the SSH destination;
no source SSH socket is used.
Source files are read once while SHA256 is accumulated; gzip level 1 tar streams
through bounded local buffers. Receiver stages individual files, verifies the
trailer and gzip CRC, then installs without replacing an existing pathname.
Verified receipts are rehashed remotely before they permit skipping a batch.
No GPU, packages, source writes, extractall(), rsync, or delete/mirror operation.
"""
import argparse
import concurrent.futures
import csv
import datetime
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import threading
import time

HERE = Path(__file__).resolve().parent
DEFAULT_AUDIT = HERE.parent / 'preflight/data_protocol/archive_data_audit.json'
DEFAULT_MANIFEST = HERE / 'phase1_deployment_manifest.json'
CHUNK = 1024 * 1024

# Identical validation/openat helpers run on both endpoints; no local code or
# dependencies need to be installed on the archive server. Python >= 3.6/Linux.
REMOTE_COMMON = r'''
import datetime, gzip, hashlib, io, json, os, re, stat, sys, tarfile, time, uuid
CHUNK = 1024 * 1024
TRAILER = '__FLOW_SOURCE_HASHES__.json'
def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')
def digest(x):
    return hashlib.sha256(canonical(x)).hexdigest()
def require(ok, message):
    if not ok: raise ValueError(message)
def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()
def safe_rel(path):
    require(isinstance(path, str) and path and len(path) < 4096, 'invalid path')
    require(not path.startswith('/') and '\\' not in path and '\x00' not in path, 'unsafe path')
    parts = path.split('/')
    require(all(p not in ('', '.', '..') and not any(ord(c) < 32 for c in p) for p in parts), 'unsafe path components')
    require(parts[0] in ('saved_flow_data', 'train_events', 'train_optical_flow'), 'not a DSEC whitelist path')
    return parts
def verify_manifest(m):
    require(m.get('schema') == 'flow-dsec-batch-v1', 'manifest schema')
    core = {k:v for k,v in m.items() if k not in ('batch_id', 'manifest_sha256')}
    sha = digest(core)
    require(m.get('manifest_sha256') == sha and m.get('batch_id') == sha[:24], 'manifest digest')
    for key in ('source_root', 'target_root'):
        p = m[key]
        require(isinstance(p,str) and p.startswith('/') and p != '/' and os.path.normpath(p) == p, 'invalid root')
    require(m['source_root'] != m['target_root'], 'source and target roots equal')
    require(isinstance(m['files'],list) and 0 < len(m['files']) < 100000, 'manifest file count')
    seen = set()
    for f in m['files']:
        p=f['relative_path']; safe_rel(p)
        require(p not in seen, 'duplicate manifest path'); seen.add(p)
        require(type(f['bytes']) is int and f['bytes'] >= 0, 'invalid size')
        require(not f.get('sha256') or re.fullmatch('[0-9a-f]{64}',f['sha256']), 'invalid known SHA')
    return m
def root_fd(path, create=False):
    # Walk absolute components using O_NOFOLLOW, including the root path itself.
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in path.strip('/').split('/'):
            if create:
                try: os.mkdir(part,0o755,dir_fd=fd)
                except FileExistsError: pass
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd); fd=nxt
        return fd
    except BaseException:
        os.close(fd); raise
def parent_fd(root, rel, create=False):
    parts=rel.split('/'); fd=os.dup(root)
    try:
        for part in parts[:-1]:
            if create:
                try: os.mkdir(part,0o755,dir_fd=fd)
                except FileExistsError: pass
            nxt=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd); fd=nxt
        return fd,parts[-1]
    except BaseException:
        os.close(fd); raise
def open_regular(root, rel):
    fd,name=parent_fd(root,rel)
    try: f=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
    finally: os.close(fd)
    try:
        s=os.fstat(f); require(stat.S_ISREG(s.st_mode),'not regular: '+rel)
        return os.fdopen(f,'rb'),s
    except BaseException:
        os.close(f); raise
def stable(a,b):
    return (a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)==(b.st_dev,b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns)
def hash_file(root, f):
    h=hashlib.sha256(); stream,before=open_regular(root,f['relative_path'])
    with stream:
        require(before.st_size==f['bytes'],'existing size mismatch: '+f['relative_path'])
        for block in iter(lambda: stream.read(CHUNK),b''): h.update(block)
        require(stable(before,os.fstat(stream.fileno())),'changed while hashing')
    return h.hexdigest()
def check_records(m, records):
    require(isinstance(records,list) and len(records)==len(m['files']),'hash inventory count')
    for expected,record in zip(m['files'],records):
        require(set(record)=={'relative_path','bytes','sha256'},'hash inventory fields')
        require(record['relative_path']==expected['relative_path'] and record['bytes']==expected['bytes'],'hash inventory membership/size/order')
        require(re.fullmatch('[0-9a-f]{64}',record['sha256']) is not None,'invalid hash')
        require(not expected.get('sha256') or expected['sha256']==record['sha256'],'audited known hash mismatch')
def check_receipt(m,r):
    require(r.get('schema')=='flow-dsec-receipt-v1' and r.get('status')=='verified','unverified receipt')
    for k in ('batch_id','manifest_sha256','source_root','target_root'):
        require(r.get(k)==m[k],'receipt identity mismatch: '+k)
    check_records(m,r['files'])
    require(r['file_count']==len(m['files']) and r['total_bytes']==sum(f['bytes'] for f in m['files']),'receipt totals')
    source=r['source_hashes']
    require(source['schema']=='flow-dsec-source-hashes-v1' and source['manifest_sha256']==m['manifest_sha256'] and source['files']==r['files'],'source hashes differ')
    require(r['source_hashes_sha256']==digest(source),'source hash inventory digest')
def receipt_rel(m):
    return '.flow_transfer/receipts/'+m['batch_id']+'.json'
def read_receipt(root,m):
    try: f,s=open_regular(root,receipt_rel(m))
    except FileNotFoundError: return None
    with f:
        require(s.st_size<=16*1024*1024,'oversized receipt')
        r=json.load(f)
    check_receipt(m,r); return r
def publish_receipt(root,m,r):
    directory,name=parent_fd(root,receipt_rel(m),True)
    temp='.'+name+'.'+uuid.uuid4().hex+'.partial'
    try:
        fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=directory)
        with os.fdopen(fd,'wb') as out:
            out.write(canonical(r)+b'\n'); out.flush(); os.fsync(out.fileno())
        try: os.link(temp,name,src_dir_fd=directory,dst_dir_fd=directory,follow_symlinks=False)
        except FileExistsError:
            prior=read_receipt(root,m)
            require(prior and prior['files']==r['files'],'conflicting receipt already exists')
        os.unlink(temp,dir_fd=directory); os.fsync(directory)
    finally:
        try: os.unlink(temp,dir_fd=directory)
        except FileNotFoundError: pass
        os.close(directory)
'''

SOURCE_CODE = REMOTE_COMMON + r'''
m=verify_manifest(json.load(sys.stdin)); root=root_fd(m['source_root'])
class HashReader:
    def __init__(self,stream): self.stream=stream; self.hash=hashlib.sha256(); self.n=0
    def read(self,size=-1):
        b=self.stream.read(size); self.hash.update(b); self.n+=len(b); return b
records=[]
try:
    with gzip.GzipFile(fileobj=sys.stdout.buffer,mode='wb',compresslevel=1,mtime=0) as zipped:
        with tarfile.open(fileobj=zipped,mode='w|',format=tarfile.USTAR_FORMAT) as tar:
            for f in m['files']:
                stream,before=open_regular(root,f['relative_path'])
                with stream:
                    require(before.st_size==f['bytes'],'source size changed: '+f['relative_path'])
                    reader=HashReader(stream)
                    info=tarfile.TarInfo(f['relative_path']); info.size=f['bytes']; info.mode=0o644; info.mtime=0
                    tar.addfile(info,reader)
                    require(reader.n==f['bytes'] and stable(before,os.fstat(stream.fileno())),'source changed while streaming')
                    value=reader.hash.hexdigest()
                    require(not f.get('sha256') or value==f['sha256'],'source audited hash mismatch')
                    records.append({'relative_path':f['relative_path'],'bytes':f['bytes'],'sha256':value})
            trailer={'schema':'flow-dsec-source-hashes-v1','manifest_sha256':m['manifest_sha256'],'files':records}
            blob=canonical(trailer); info=tarfile.TarInfo(TRAILER); info.size=len(blob); info.mode=0o644
            tar.addfile(info,io.BytesIO(blob))
finally: os.close(root)
'''

CHECK_CODE = REMOTE_COMMON + r'''
m=verify_manifest(json.load(sys.stdin))
try: root=root_fd(m['target_root'])
except FileNotFoundError:
    print(json.dumps({'status':'missing'})); sys.exit(0)
try:
    r=read_receipt(root,m)
    if r is None: print(json.dumps({'status':'missing'}))
    else:
        for f in r['files']: require(hash_file(root,f)==f['sha256'],'verified receipt file hash changed: '+f['relative_path'])
        print(json.dumps({'status':'verified','receipt':r}))
finally: os.close(root)
'''

TARGET_CODE = REMOTE_COMMON + r'''
header=sys.stdin.buffer.readline(16*1024*1024+1)
require(header.endswith(b'\n') and len(header)<=16*1024*1024,'missing/oversized manifest header')
m=verify_manifest(json.loads(header)); root=root_fd(m['target_root'],True)
staged=[]; records=[]; start=time.monotonic(); installed=0; reused=0
try:
    # Duplicate invocations of one batch are rejected rather than interleaved.
    import fcntl
    lockdir,lockname=parent_fd(root,'.flow_transfer/locks/'+m['batch_id']+'.lock',True)
    try: lock=os.open(lockname,os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600,dir_fd=lockdir)
    finally: os.close(lockdir)
    require(stat.S_ISREG(os.fstat(lock).st_mode),'lock is not regular')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    with gzip.GzipFile(fileobj=sys.stdin.buffer,mode='rb') as zipped:
        with tarfile.open(fileobj=zipped,mode='r|') as tar:
            for expected in m['files']:
                member=tar.next()
                require(member is not None and member.name==expected['relative_path'],'unexpected tar member/order')
                require(member.isreg() and not member.issparse() and member.size==expected['bytes'] and not member.pax_headers,'unsafe member or wrong size')
                directory,name=parent_fd(root,member.name,True)
                temp='.'+name+'.flow-'+m['batch_id']+'-'+uuid.uuid4().hex+'.partial'
                staged.append((directory,temp,name,expected))
                fd=os.open(temp,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o644,dir_fd=directory)
                h=hashlib.sha256(); count=0
                with os.fdopen(fd,'wb') as out:
                    inp=tar.extractfile(member)
                    for block in iter(lambda: inp.read(CHUNK),b''):
                        out.write(block); h.update(block); count+=len(block)
                    require(count==expected['bytes'],'short tar member')
                    out.flush(); os.fsync(out.fileno())
                records.append({'relative_path':member.name,'bytes':count,'sha256':h.hexdigest()})
            member=tar.next()
            require(member is not None and member.name==TRAILER and member.isreg() and not member.pax_headers and 0<member.size<16*1024*1024,'missing/unsafe source hash trailer')
            trailer=json.load(tar.extractfile(member))
            require(trailer.get('schema')=='flow-dsec-source-hashes-v1' and trailer.get('manifest_sha256')==m['manifest_sha256'],'source trailer identity')
            check_records(m,trailer.get('files'))
            require(records==trailer['files'],'source vs received SHA256 mismatch')
            require(tar.next() is None,'extra tar member')
        # Force gzip footer/CRC validation before any destination file is installed.
        while zipped.read(CHUNK): pass
    # Precheck ALL conflicts before committing any new file. Existing same bytes
    # are reused; mismatched pre-existing files are never replaced or deleted.
    for f in records:
        try: existing_hash=hash_file(root,f)
        except FileNotFoundError: continue
        require(existing_hash==f['sha256'],'refuse differing existing file: '+f['relative_path'])
    for (directory,temp,name,expected),record in zip(staged,records):
        try:
            os.link(temp,name,src_dir_fd=directory,dst_dir_fd=directory,follow_symlinks=False)
            installed+=1
        except FileExistsError:
            require(hash_file(root,record)==record['sha256'],'destination raced or differs')
            reused+=1
        os.unlink(temp,dir_fd=directory); os.fsync(directory)
    # New final files are the fsynced, locally hashed staging inodes. They were
    # installed atomically. Reused files were rehashed; later resume rehashes all.
    r={'schema':'flow-dsec-receipt-v1','status':'verified','batch_id':m['batch_id'],
       'manifest_sha256':m['manifest_sha256'],'source_root':m['source_root'],'target_root':m['target_root'],
       'file_count':len(records),'total_bytes':sum(f['bytes'] for f in records),'files':records,
       'source_hashes':trailer,'source_hashes_sha256':digest(trailer),'verified_utc':utc(),
       'installed_files':installed,'reused_files':reused,'receiver_seconds':time.monotonic()-start,
       'verification':'streamed SHA256 equals source trailer; gzip CRC; atomic non-overwriting links; fsync'}
    check_receipt(m,r); publish_receipt(root,m,r)
    print(json.dumps(r))
finally:
    for directory,temp,name,expected in staged:
        try: os.unlink(temp,dir_fd=directory)
        except FileNotFoundError: pass
        os.close(directory)
    os.close(root)
'''


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('ascii')


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(CHUNK), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.' + path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        os.replace(temp, str(path))
    finally:
        if os.path.exists(temp): os.unlink(temp)


def batch_manifest(files, source_root, target_root):
    core = {'schema': 'flow-dsec-batch-v1', 'source_root': source_root,
            'target_root': target_root, 'files': files}
    sha = hashlib.sha256(canonical(core)).hexdigest()
    core.update(batch_id=sha[:24], manifest_sha256=sha)
    # Validate with precisely the same definitions used remotely.
    remote_functions()['verify_manifest'](core)
    return core


_REMOTE_FUNCTIONS = None


def remote_functions():
    global _REMOTE_FUNCTIONS
    if _REMOTE_FUNCTIONS is None:
        _REMOTE_FUNCTIONS = {}
        exec(REMOTE_COMMON, _REMOTE_FUNCTIONS)
    return _REMOTE_FUNCTIONS


def load_whitelist(audit_path, manifest_path, source_csv_override=None):
    audit = json.loads(audit_path.read_text())
    meta = json.loads(manifest_path.read_text())
    csv_name = meta['files_csv']
    if Path(csv_name).name != csv_name:
        raise ValueError('deployment CSV must be sibling of JSON manifest')
    csv_path = manifest_path.parent / csv_name
    if sha_file(csv_path) != meta['files_csv_sha256']:
        raise ValueError('sealed deployment CSV hash mismatch')
    rows = list(csv.DictReader(csv_path.open(newline='')))
    # Delivery order only: preserve the sealed membership, prioritize usable
    # train input/GT/mask triplets before validation and auxiliary metadata.
    rows.sort(key=lambda r: ({'train': 0, 'valid': 1, 'metadata': 2}.get(r['split'], 3),
                            r.get('target', ''),
                            {'input': 0, 'gt': 1, 'mask': 2}.get(r['category'], 3),
                            r['relative_path']))
    files = [{'relative_path': r['relative_path'], 'bytes': int(r['bytes']),
              'sha256': r.get('known_sha256') or None} for r in rows]
    source_csv = (Path(source_csv_override) if source_csv_override is not None
                  else Path(audit['transfer_manifest']['path']))
    if sha_file(source_csv) != audit['transfer_manifest']['sha256']:
        raise ValueError('archive audit CSV hash mismatch')
    archived = list(csv.DictReader(source_csv.open(newline='')))
    metadata = audit['phase1_metadata_transfer']['files']
    if len(archived) != 24456 or len(metadata) != 40 or len(files) != 24496:
        raise ValueError('scope must be exactly archive 24456 + metadata 40')
    expected = {r['relative_path']: int(r['bytes']) for r in archived + metadata}
    actual = {f['relative_path']: f['bytes'] for f in files}
    if len(expected) != 24496 or len(actual) != len(files) or actual != expected:
        raise ValueError('deployment membership differs from audited archive scope')
    if sum(f['bytes'] for f in files) != audit['phase1_total_bytes_with_metadata'] or meta['expected_files'] != len(files) or meta['expected_bytes'] != sum(actual.values()):
        raise ValueError('deployment/audit total mismatch')
    known = {r['relative_path']: r['sha256'] for r in metadata if r.get('sha256')}
    if any(known.get(f['relative_path'], f['sha256']) != f['sha256'] for f in files):
        raise ValueError('known metadata hashes disagree')
    return files, audit['remote_root'], {'audit_sha256': sha_file(audit_path),
        'deployment_json_sha256': sha_file(manifest_path),
        'deployment_csv_sha256': sha_file(csv_path)}


def split_batches(files, limit, smoke):
    result, batch, size = [], [], 0
    for f in files:
        # Receiver holds one parent FD per staged file; cap well below the
        # usual 1024-descriptor limit even for batches of small mask files.
        if batch and (size + f['bytes'] > limit or len(batch) >= 384):
            result.append(batch)
            if smoke: return result
            batch, size = [], 0
        batch.append(f); size += f['bytes']
    if batch: result.append(batch)
    return result[:1] if smoke else result


def ssh_args(endpoint, code=None):
    host, port, socket = endpoint
    base = ['ssh', '-T', '-S', socket, '-p', str(port), '-o', 'ControlMaster=no',
            '-o', 'BatchMode=yes', '-o', 'ProxyCommand=false',
            '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=30',
            '-o', 'ServerAliveCountMax=3', host]
    if code is not None:
        base += ['python3 -c ' + shlex.quote(code)]
    return base


def socket_check(endpoint):
    host, port, socket = endpoint
    if not os.path.exists(socket):
        raise ValueError('missing existing SSH control socket: ' + socket)
    subprocess.run(['ssh', '-S', socket, '-p', str(port), '-O', 'check', host],
                   check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=20)


def local_or_ssh(endpoint, code):
    return [sys.executable, '-c', code] if endpoint is None else ssh_args(endpoint, code)


def verified_receipt(endpoint, manifest):
    p = subprocess.run(local_or_ssh(endpoint, CHECK_CODE), input=canonical(manifest),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=600)
    if p.returncode:
        raise RuntimeError('receipt verification failed: ' + p.stderr.decode(errors='replace')[-3000:])
    result = json.loads(p.stdout)
    if result['status'] == 'missing': return None
    remote_functions()['check_receipt'](manifest, result['receipt'])
    return result['receipt']


PRINT_LOCK = threading.Lock()
STOP = threading.Event()


def progress(message):
    with PRINT_LOCK:
        print(datetime.datetime.now(datetime.timezone.utc).isoformat() + ' ' + message, flush=True)


def transfer_one(manifest, source, target, output, check_existing=True):
    batch_id = manifest['batch_id']
    if STOP.is_set(): raise RuntimeError('cancelled before starting batch')
    begin = time.monotonic()
    if check_existing:
        receipt = verified_receipt(target, manifest)
        if receipt is not None:
            persist_receipt(output, receipt)
            progress(batch_id + ' skipped: verified receipt and current destination hashes match')
            return {'batch_id': batch_id, 'status': 'verified', 'skipped': True,
                    'files': receipt['file_count'], 'bytes': receipt['total_bytes'],
                    'elapsed_seconds': time.monotonic()-begin}
    progress('{} start: {} files, {:.1f} MiB'.format(batch_id, len(manifest['files']), sum(f['bytes'] for f in manifest['files'])/2**20))
    log_dir = output / 'logs'; log_dir.mkdir(parents=True, exist_ok=True)
    src = dst = None
    compressed_bytes = 0
    try:
        with (log_dir/(batch_id+'.source.stderr')).open('wb') as src_log, (log_dir/(batch_id+'.target.stderr')).open('wb') as dst_log, (log_dir/(batch_id+'.target.stdout')).open('wb') as reply:
            dst = subprocess.Popen(local_or_ssh(target, TARGET_CODE), stdin=subprocess.PIPE,
                                   stdout=reply, stderr=dst_log, bufsize=0)
            dst.stdin.write(canonical(manifest)+b'\n'); dst.stdin.flush()
            src = subprocess.Popen(local_or_ssh(source, SOURCE_CODE), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=src_log, bufsize=0)
            src.stdin.write(canonical(manifest)); src.stdin.close()
            last = time.monotonic()
            while True:
                if STOP.is_set(): raise RuntimeError('cancelled')
                block = src.stdout.read(CHUNK)
                if not block: break
                # FileIO.write may return short writes; never silently drop bytes.
                view = memoryview(block)
                while view:
                    n = dst.stdin.write(view)
                    if not n: raise BrokenPipeError('receiver stopped accepting data')
                    view = view[n:]
                compressed_bytes += len(block)
                if time.monotonic()-last >= 30:
                    progress('{} relay {:.1f} MiB compressed; {:.1f}s'.format(batch_id,compressed_bytes/2**20,time.monotonic()-begin)); last=time.monotonic()
            src.stdout.close(); dst.stdin.close()
            source_rc=src.wait(timeout=120); target_rc=dst.wait(timeout=600)
            if source_rc or target_rc:
                raise RuntimeError('source rc={}, target rc={}; see logs/{}'.format(source_rc,target_rc,batch_id))
        receipt=json.loads((log_dir/(batch_id+'.target.stdout')).read_text())
        remote_functions()['check_receipt'](manifest,receipt)
        persist_receipt(output,receipt)
        progress('{} verified: {} files; {} new, {} reused'.format(batch_id,receipt['file_count'],receipt['installed_files'],receipt['reused_files']))
        return {'batch_id':batch_id,'status':'verified','skipped':False,'files':receipt['file_count'],
                'bytes':receipt['total_bytes'],'compressed_stream_bytes':compressed_bytes,
                'elapsed_seconds':time.monotonic()-begin}
    finally:
        for proc in (src,dst):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try: proc.wait(timeout=10)
                except subprocess.TimeoutExpired: proc.kill(); proc.wait(timeout=10)


def persist_receipt(output,receipt):
    batch_id=receipt['batch_id']
    write_json(output/'receipts'/(batch_id+'.json'),receipt)
    write_json(output/'source_hashes'/(batch_id+'.json'),receipt['source_hashes'])


def source_csv(output, manifests, smoke):
    files=[]; seen=set()
    for manifest in manifests:
        receipt=json.loads((output/'receipts'/(manifest['batch_id']+'.json')).read_text())
        remote_functions()['check_receipt'](manifest,receipt)
        for f in receipt['source_hashes']['files']:
            if f['relative_path'] in seen: raise ValueError('duplicate source inventory path')
            seen.add(f['relative_path']); files.append(f)
    wanted={f['relative_path'] for m in manifests for f in m['files']}
    if seen != wanted: raise ValueError('incomplete source inventory')
    path=output/('smoke_source_hashes.csv' if smoke else 'source_hashes.csv')
    fd,tmp=tempfile.mkstemp(prefix='.'+path.name+'.',dir=str(output))
    try:
        with os.fdopen(fd,'w',newline='') as out:
            writer=csv.DictWriter(out,fieldnames=['relative_path','bytes','sha256'])
            writer.writeheader(); writer.writerows(files); out.flush(); os.fsync(out.fileno())
        os.replace(tmp,str(path))
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return str(path)


def self_test():
    """Local only: real streaming code + adversarial corrupt/path/receipt cases."""
    results=[]
    with tempfile.TemporaryDirectory(prefix='flow_transfer_selftest_') as tmp:
        base=Path(tmp); src=base/'source'; dst=base/'target'; out=base/'evidence'
        rels=['saved_flow_data/test/one.npy','train_events/test/events/left/rectify_map.h5',
              'train_optical_flow/test/flow/forward_timestamps.txt']
        files=[]
        for i,rel in enumerate(rels):
            path=src/rel; path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes((bytes(range(256))*(1024+i)) + b'EOF')
            files.append({'relative_path':rel,'bytes':path.stat().st_size,'sha256':sha_file(path) if i==0 else None})
        m=batch_manifest(files,str(src),str(dst))
        first=transfer_one(m,None,None,out)
        assert first['status']=='verified' and not first['skipped']; results.append('normal_stream')
        for f in files: assert sha_file(src/f['relative_path'])==sha_file(dst/f['relative_path'])
        second=transfer_one(m,None,None,out)
        assert second['skipped']; results.append('resume_rehash')
        # Different batching overlaps smoke/full: same files reused without replacement.
        overlap=batch_manifest(files[:2],str(src),str(dst))
        transfer_one(overlap,None,None,out)
        r=json.loads((out/'receipts'/(overlap['batch_id']+'.json')).read_text())
        assert r['reused_files']==2 and r['installed_files']==0; results.append('overlap_reuse')
        unrelated=dst/'unrelated.txt'; unrelated.write_text('preserve me')
        conflict=dst/rels[2]; original=conflict.read_bytes(); conflict.write_bytes(b'Z'*len(original))
        try: verified_receipt(None,m)
        except RuntimeError: results.append('corrupt_receipt_file_rejected')
        else: raise AssertionError('bad receipt accepted')
        conflict_m=batch_manifest(files[2:],str(src),str(dst))
        try: transfer_one(conflict_m,None,None,out)
        except RuntimeError: results.append('existing_conflict_not_overwritten')
        else: raise AssertionError('conflict accepted')
        assert conflict.read_bytes()==b'Z'*len(original) and unrelated.read_text()=='preserve me'
        # Fresh target + source hash mismatch must publish neither files nor receipt.
        bad_dst=base/'bad_known'; badfiles=[dict(files[0],sha256='0'*64)]
        bad=batch_manifest(badfiles,str(src),str(bad_dst))
        try: transfer_one(bad,None,None,out)
        except (RuntimeError,BrokenPipeError): results.append('source_known_hash_failure')
        else: raise AssertionError('bad known hash accepted')
        assert not (bad_dst/rels[0]).exists()
        for path in ['../outside','/absolute','saved_flow_data/../x','saved_flow_data//x','saved_flow_data/./x','unrelated/x']:
            try: batch_manifest([{'relative_path':path,'bytes':1,'sha256':None}],str(src),str(dst))
            except ValueError: pass
            else: raise AssertionError('unsafe path accepted: '+path)
        results.append('unsafe_manifest_paths_rejected')
        symlink_dst=base/'symlink'; symlink_dst.mkdir(); outside=base/'outside'; outside.mkdir()
        (symlink_dst/'saved_flow_data').symlink_to(outside,target_is_directory=True)
        sm=batch_manifest(files[:1],str(src),str(symlink_dst))
        try: transfer_one(sm,None,None,out)
        except (RuntimeError,BrokenPipeError): results.append('symlink_parent_rejected')
        else: raise AssertionError('symlink destination accepted')
        assert not list(outside.iterdir())
        # Receiver must reject valid tar with bad trailer digest, unknown member,
        # truncated gzip/footer and bit corruption before creating final files.
        import gzip,tarfile
        def send_tar(kind):
            target=base/('attack_'+kind); mm=batch_manifest(files[:1],str(src),str(target))
            buf=io.BytesIO()
            with tarfile.open(fileobj=buf,mode='w',format=tarfile.USTAR_FORMAT) as tar:
                content=(src/rels[0]).read_bytes(); name=rels[0] if kind!='unknown' else 'saved_flow_data/unlisted.npy'
                item=tarfile.TarInfo(name); item.size=len(content); tar.addfile(item,io.BytesIO(content))
                record={'relative_path':rels[0],'bytes':len(content),'sha256':sha_file(src/rels[0])}
                if kind=='bad_trailer': record['sha256']='f'*64
                trailer=canonical({'schema':'flow-dsec-source-hashes-v1','manifest_sha256':mm['manifest_sha256'],'files':[record]})
                item=tarfile.TarInfo('__FLOW_SOURCE_HASHES__.json'); item.size=len(trailer); tar.addfile(item,io.BytesIO(trailer))
            zipped=gzip.compress(buf.getvalue(),compresslevel=1)
            if kind=='truncated': zipped=zipped[:-6]
            if kind=='crc': zipped=zipped[:-8]+bytes([zipped[-8]^1])+zipped[-7:]
            p=subprocess.run([sys.executable,'-c',TARGET_CODE],input=canonical(mm)+b'\n'+zipped,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
            assert p.returncode!=0 and not (target/rels[0]).exists(),kind
            assert not (target/'.flow_transfer/receipts'/ (mm['batch_id']+'.json')).exists(),kind
        for kind in ['unknown','bad_trailer','truncated','crc']:
            send_tar(kind); results.append(kind+'_rejected_before_publish')
        csv_path=source_csv(out,[m],True)
        assert len(list(csv.DictReader(open(csv_path))))==3; results.append('source_csv_export')
    result={'status':'passed','scope':'local synthetic files only; no SSH/GPU/archive transfer',
            'tests':results,'test_count':len(results),'script_sha256':sha_file(__file__)}
    write_json(HERE/'transfer_self_test_results.json',result)
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--audit',type=Path,default=DEFAULT_AUDIT)
    p.add_argument('--manifest',type=Path,default=DEFAULT_MANIFEST)
    p.add_argument('--source-csv',type=Path,help='relocated archive CSV; must match original SHA256 recorded in --audit')
    p.add_argument('--out-dir',type=Path,default=HERE/'transfer_run')
    p.add_argument('--target-root',default='/data/flow_a/data/DSEC')
    p.add_argument('--source-host',default='root@ssh.sd5ai.scnet.cn')
    p.add_argument('--source-port',type=int,default=10037)
    p.add_argument('--source-socket',default='/tmp/flow_archive_setup_20261011.sock')
    p.add_argument('--source-local',action='store_true',help='read on this host with local Python; no source SSH/socket')
    p.add_argument('--target-host',default='root@36.213.75.115')
    p.add_argument('--target-port',type=int,default=30449)
    p.add_argument('--target-socket',default='/tmp/flow_a100_new_20261011.sock')
    p.add_argument('--target-local',action='store_true',help='receive on this host with local Python; no target SSH/socket')
    p.add_argument('--jobs',type=int,choices=[1,2],default=2)
    p.add_argument('--batch-mib',type=int,default=768)
    p.add_argument('--smoke',action='store_true',help='one prefix batch of at most ~60 MiB')
    p.add_argument('--run',action='store_true',help='perform SSH streaming; default plans only')
    p.add_argument('--self-test',action='store_true',help='synthetic local protocol tests, no SSH')
    args=p.parse_args()
    if args.self_test:
        if args.run: p.error('--self-test cannot be combined with --run')
        self_test(); return
    if not 512<=args.batch_mib<=1024: p.error('--batch-mib must be 512..1024')
    files,source_root,provenance=load_whitelist(args.audit,args.manifest,args.source_csv)
    groups=split_batches(files,(60 if args.smoke else args.batch_mib)*2**20,args.smoke)
    manifests=[batch_manifest(g,source_root,args.target_root) for g in groups]
    args.out_dir.mkdir(parents=True,exist_ok=True)
    for m in manifests: write_json(args.out_dir/'manifests'/(m['batch_id']+'.json'),m)
    tag='smoke' if args.smoke else 'full'
    plan={'schema':'flow-dsec-transfer-plan-v1','mode':tag,'source_root':source_root,'target_root':args.target_root,
          'files':sum(len(m['files']) for m in manifests),'uncompressed_bytes':sum(f['bytes'] for m in manifests for f in m['files']),
          'batches':[{'batch_id':m['batch_id'],'manifest_sha256':m['manifest_sha256'],'files':len(m['files']),
                      'bytes':sum(f['bytes'] for f in m['files'])} for m in manifests],
          'jobs':args.jobs,'source_transport':'local' if args.source_local else 'ssh',
          'target_transport':'local' if args.target_local else 'ssh',
          'provenance':provenance,'script_sha256':sha_file(__file__)}
    write_json(args.out_dir/(tag+'_plan.json'),plan)
    progress('{} plan: {} batches, {} files, {:.3f} GiB; run={}'.format(tag,len(manifests),plan['files'],plan['uncompressed_bytes']/2**30,args.run))
    if not args.run: return
    source=None if args.source_local else (args.source_host,args.source_port,args.source_socket)
    target=None if args.target_local else (args.target_host,args.target_port,args.target_socket)
    if source is not None: socket_check(source)
    if target is not None: socket_check(target)
    results=[]; failures=[]; started=time.time()
    summary_path=args.out_dir/(tag+'_summary.json')
    def summary(status):
        write_json(summary_path,{'schema':'flow-dsec-transfer-summary-v1','status':status,'mode':tag,
                   'plan':str(args.out_dir/(tag+'_plan.json')),'batches':results,'failures':failures,
                   'elapsed_seconds':time.time()-started,'verified_files':sum(r['files'] for r in results),
                   'verified_bytes':sum(r['bytes'] for r in results)})
    summary('running')
    # At most two source Python/gzip processes; each is single threaded.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        pending={executor.submit(transfer_one,m,source,target,args.out_dir):m['batch_id'] for m in manifests}
        try:
            for future in concurrent.futures.as_completed(pending):
                try: results.append(future.result())
                except Exception as exc:
                    failures.append({'batch_id':pending[future],'error':str(exc)})
                    STOP.set()
                    for other in pending: other.cancel()
                    progress(pending[future]+' FAILED: '+str(exc))
                summary('failed' if failures else 'running')
        except KeyboardInterrupt:
            STOP.set()
            for future in pending: future.cancel()
            failures.append({'error':'keyboard interrupt; rerun same plan to resume'})
            summary('interrupted')
            raise
    if failures:
        summary('failed'); raise SystemExit(1)
    inventory=source_csv(args.out_dir,manifests,args.smoke)
    summary('verified')
    progress('all selected batches verified; source inventory: '+inventory)


if __name__=='__main__':
    main()
