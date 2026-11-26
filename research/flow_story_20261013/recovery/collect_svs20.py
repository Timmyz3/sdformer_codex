#!/usr/bin/env python3
"""Twenty fixed train-gate-fit samples -> unchanged native sn2 counts / CPU FP64 SVS.

Independent fit20 adapter of pinned breadth/collect_svs.py. No old source edit.
The binary count and SVD formula helpers remain AST-identical to that source.
Only checkpoint135930/P_paper/full9 is accepted; no GT/mask array is read.

Default collect is CPU provenance preflight only. Explicit --run-inference
allows the requested forward device. SVD always runs in an isolated CPU worker
with a hard deadline. No partial score artifact is published as usable.
"""
import argparse
import contextlib
import copy
import fcntl
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
BREADTH_ROOT = HERE.parent/'breadth'
UPSTREAM_COLLECTOR_SHA = '49b0668626d4be21f861efc587a397e13b17a17bf4fe0d382478656560fac831'
DATA_CONTRACT_SHA = 'c4421befb3f2f9779b259ed0f0db114d70a06ff9d19723a98949e5ad281cfd03'
CHECKPOINT_SHA = 'd5cdd8bed7cd7eb1d189a78faf3f5864e9040d2a4b9e5f10f131b6932fbf0b30'
EVALUATOR_SHA = '1cacbaeea4d9eaf19a304c3faaa13f5278f0212b6cdc92210c4619447d571628'
STATIC_SHA = '5b1938b2815a8201066db01c9a1d3920aa523116a30b22a6f4883fc86c477aab'
FIT_SHA = '29ec7ba9c9faca74c62c3e170c9d3c70855a4605a7eab31f9331c519f56f8915'


def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()


def atomic(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)


def import_pinned(name, filename, expected):
    require(sha(BREADTH_ROOT/'collect_svs.py') == UPSTREAM_COLLECTOR_SHA, 'Upstream collector ancestry differs')
    path = BREADTH_ROOT/filename
    require(sha(path) == expected, 'Implementation pin differs: '+filename)
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m)
    return m


def static_module(prior):
    m = import_pinned('flow_collect_static', 'static_methods.py', STATIC_SHA)
    m.configure_prior_root(prior)
    return m


def fixed_fit_manifest():
    path = HERE/'manifests/svs_fit20.json'
    require(sha(path) == FIT_SHA, 'Fixed20 manifest changed')
    require(sha(HERE/'manifests/data_contract.json') == DATA_CONTRACT_SHA, 'Sealed recovery data contract changed')
    sample = json.loads(path.read_text())
    require(sample.get('schema') == 'flow_recovery_svs_fit20_v1' and sample.get('count') == 20 and
            len(sample['target_ids']) == len(set(sample['target_ids'])) == 20 and
            sample['target_ids_sha256'] == digest(sample['target_ids']) and
            sample.get('checkpoint_sha256') == CHECKPOINT_SHA and sample.get('checkpoint_optimizer_steps') == 135930 and
            sample.get('BN_policy') == 'P_paper' and sample.get('geometry') == 'full9', 'Fixed fit20 contract differs')
    return sample


def fixed_fit_rows(canonical, roles, metrics):
    sample = fixed_fit_manifest()
    records = metrics.canonical_rows(canonical); role_map = metrics.auxiliary_roles(roles, records)
    require(sample['canonical_sha256'] == metrics.CANONICAL_SHA and sample['roles_sha256'] == metrics.AUX_ROLES_SHA,
            'SVS manifest provenance differs')
    ordered = sorted((r for r in records if role_map.get(r['target_id']) == 'gate_fit'),
        key=lambda r: (hashlib.sha256((sample['seed']+r['target_id']).encode()).hexdigest(),r['target_id']))
    counts = {}; selected = []
    for row in ordered:
        seq = row['sequence']
        if counts.get(seq,0) < 2:
            selected.append(row); counts[seq] = counts.get(seq,0)+1
    require(len(counts) == 10 and set(counts.values()) == {2} and
            [r['target_id'] for r in selected] == sample['target_ids'] and
            set(sample['nested_fit2_target_ids']) <= set(sample['target_ids']) and
            all(r['protocol_split'] == 'train' for r in selected), 'SVS fit20 membership/role/nesting changed')
    return selected, sample


def preflight(a):
    require(a.checkpoint_sha256 == CHECKPOINT_SHA and a.bn == 'P_paper' and a.geometry == 'full9',
            'This sealed fit20 run requires checkpoint135930, P_paper and full9')
    evaluation = import_pinned('flow_collect_evaluation', 'evaluate_breadth.py', EVALUATOR_SHA)
    native, trainer, metrics, pins = evaluation.support_modules(a.prior_root)
    selected, sample = fixed_fit_rows(a.canonical, a.roles, metrics)
    train, hashes, binding, audit = native.validate_native_bundle(a.data_root, a.canonical, a.acceptance,
        a.acceptance_sha256, a.archive_audit, check_current_files=False)
    # Input-only scoring: validate sealed full-bundle metadata, then read/hash
    # only the fixed20 inputs below. No current valid/check file is opened.
    audit.update(current_file_stat_scope="fixed20 event inputs at actual read", GT_mask_arrays_read=False)
    binding.update(entrypoint_sha256=pins['baseline/train_native_baseline.py'], adapter_sha256=pins['baseline/native_data_acceptance.py'],
                   training_core_sha256=trainer.CORE_SHA, checkpoint_schema=evaluation.SCHEMA, checkpoint_creator=evaluation.CREATOR)
    core = trainer.load_core(); sources = native.pinned_json(a.source_manifest, core.SOURCE_MANIFEST_SHA)
    require(len(sources) == 63, 'Expected pinned63 native source files')
    for rel, value in sources.items(): require(sha(native.checked_path(a.source_root,rel)) == value, 'Native source differs:'+rel)
    require(sha(a.checkpoint) == a.checkpoint_sha256, 'Explicit checkpoint SHA differs')
    side = json.loads(a.checkpoint.with_suffix('.json').read_text())
    require(side.get('checkpoint_sha256') == a.checkpoint_sha256 and side.get('checkpoint_file') == a.checkpoint.name,
            'Checkpoint sidecar identity differs')
    run = json.loads((a.checkpoint.parent/'run_contract.json').read_text())
    config = evaluation.validate_contract(run, side, binding, sources, trainer, train, hashes)
    require(side.get('completed_optimizer_steps') == 135930, 'Checkpoint step is not the sealed135930')
    static_module(a.prior_root)
    plan = {'schema': 'flow_svs_collection_preflight_v1', 'status': 'PREFLIGHT_PASS_NO_FORWARD',
        'collector_sha256': sha(__file__), 'evaluator_sha256': EVALUATOR_SHA, 'static_methods_sha256': STATIC_SHA,
        'fit_manifest_sha256': FIT_SHA, 'data_manifest_contract_sha256': DATA_CONTRACT_SHA,
        'upstream_collector_sha256': UPSTREAM_COLLECTOR_SHA, 'fit_sample': sample, 'data_contract': binding,
        'data_preflight': audit, 'source_sha256': sources, 'checkpoint_sha256': a.checkpoint_sha256,
        'checkpoint_sidecar': side, 'bn_policy': a.bn, 'geometry': a.geometry,
        'forward_executed': False, 'training_executed': False, 'svs_scoring_seconds_cap': a.score_seconds,
        'scope': 'fixed20 input-only train-fit exploration; no AEE reproduction or method rejection claim'}
    return evaluation, native, metrics, selected, hashes, config, run, side, plan


def binary_time_counts(spikes, expected_channels=None):
    """After exact binary guard, losslessly reduce complete T10 to CPU uint8.

    Accepts CPU or caller's forward device; this function itself never selects
    a CUDA device. No float averaging before transfer or spatial downsampling.
    """
    import torch
    require(isinstance(spikes, torch.Tensor) and spikes.ndim == 5 and tuple(spikes.shape[:2]) == (10,1)
            and min(spikes.shape) > 0 and not spikes.is_complex(), 'Expected complete [10,1,H,W,F] sn2 tensor')
    require(expected_channels is None or spikes.shape[-1] == expected_channels, 'sn2 hidden width differs')
    require(bool(torch.isfinite(spikes).all()) and bool(((spikes == 0) | (spikes == 1)).all()),
            'SVS sn2 input is not finite binary spikes')
    return spikes.detach().sum(dim=0, dtype=torch.int16).to(dtype=torch.uint8, device='cpu').contiguous()


def ranks_from_counts(counts, channel_chunk=16):
    """Exact original SVS definition for losslessly counted binary T10 maps."""
    import torch
    require(isinstance(counts, torch.Tensor) and counts.device.type == 'cpu' and counts.dtype == torch.uint8
            and counts.ndim == 4 and counts.shape[0] == 1 and min(counts.shape) > 0,
            'Expected CPU uint8 [1,H,W,F] complete-T counts')
    require(bool((counts <= 10).all()), 'Counts exceed complete T10')
    require(type(channel_chunk) is int and 1 <= channel_chunk <= 64, 'Invalid SVD channel chunk')
    matrix = counts.permute(0,3,1,2)
    ranks = torch.empty((1, counts.shape[-1]), dtype=torch.int64)
    with torch.no_grad():
        for c in range(0, counts.shape[-1], channel_chunk):
            average = matrix[:,c:c+channel_chunk].to(torch.float64)/10
            ranks[:,c:c+channel_chunk] = (torch.linalg.svdvals(average) > 1e-6).sum(-1)
    return ranks


class CountHooks:
    def __init__(self, model, directory, paths, shape_by_path):
        self.model, self.directory, self.paths, self.shapes = model, Path(directory), list(paths), shape_by_path
        self.handles, self.records, self.seen, self.target = [], [], set(), None

    def __enter__(self):
        for path in self.paths:
            def hook(module, inputs, output, name=path):
                import numpy as np
                require(self.target is not None and name not in self.seen, 'Duplicate or unbound sn2 callback')
                counts = binary_time_counts(output, self.shapes[name]['hidden_channels'])
                filename = digest([self.target,name])+'.npy'
                destination = self.directory/filename
                require(not destination.exists(), 'Counts artifact is immutable')
                temp = destination.with_suffix('.npy.tmp')
                with temp.open('wb') as f:
                    np.save(f, counts.numpy(), allow_pickle=False); f.flush(); os.fsync(f.fileno())
                os.replace(temp, destination)
                self.records.append({'target_id': self.target, 'mlp_path': name, 'file': filename,
                    'bytes': destination.stat().st_size, 'sha256': sha(destination),
                    'shape_BHWF': list(counts.shape), 'dtype': 'uint8', 'complete_T': 10,
                    'binary_guard_passed': True, 'score_site': 'sn2 output before drop2/fc2'})
                self.seen.add(name)
            self.handles.append(self.model.get_submodule(path).sn2.register_forward_hook(hook))
        return self

    def begin(self, target):
        require(self.target is None, 'Previous sample not ended'); self.target, self.seen = target, set()

    def end(self):
        require(self.seen == set(self.paths), 'Not every native MLP sn2 was observed exactly once')
        self.target = None

    def __exit__(self, *args):
        for handle in self.handles: handle.remove()


def collect_counts(a, context):
    evaluation,native,metrics,rows,hashes,config,run,side,plan = context
    import numpy as np
    import torch
    torch.set_num_threads(a.cpu_threads); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    device=torch.device(a.device); require(device.type in ('cpu','cuda'), 'Unsupported device')
    require(device.type != 'cuda' or torch.cuda.is_available(), 'Requested CUDA unavailable')
    if device.type == 'cuda': torch.cuda.set_device(device)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    require(saved.get('schema_version') == evaluation.SCHEMA and saved.get('contract') == run['contract'] and
        saved.get('contract_sha256') == side['contract_sha256'] and saved.get('provenance') == run['provenance'] and
        saved.get('effective_config') == run['effective_config'] and
        saved['training']['completed_optimizer_steps'] == side['completed_optimizer_steps'], 'Checkpoint payload differs')
    state=saved['model_state_dict']; require(state and all(isinstance(v,torch.Tensor) and
        (not v.is_floating_point() or bool(torch.isfinite(v).all())) for v in state.values()), 'Nonfinite/non-tensor state')
    sys.path.insert(0,str(a.source_root)); os.environ['SDFORMER_SNN_BACKEND']='torch'
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    from models.STSwinNet_SNN.Spiking_submodules import PSN
    from utils.runtime_backend import configure_snn_backend
    from spikingjelly.activation_based import functional
    for obj in (MS_SpikingformerFlowNet_en4,PSN,configure_snn_backend):
        require(Path(inspect.getfile(obj)).resolve().is_relative_to(a.source_root), 'Unpinned native model import')
    config=copy.deepcopy(config); geometry={'name':a.geometry,'input_hw':[288,384] if a.geometry=='crop_center9' else [480,640],
        'window':[2,9,9], 'T':10, 'B':1, 'precision':'FP32 TF32off noautocast', 'input_preprocess':'pinned evaluate_breadth.prepare_input'}
    config['swin_transformer']['input_size']=geometry['input_hw']
    model=MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']),copy.deepcopy(config['swin_transformer']))
    model.load_state_dict(state,strict=True); del saved,state
    functional.set_step_mode(model,'m'); model.to(device); configure_snn_backend(model,device,config,PSN)
    evaluation.apply_bn(model,a.bn,torch)
    before=evaluation.bn_signature(model,torch)
    methods=static_module(a.prior_root); svs=methods._svs()
    paths=sorted(n for n,m in model.named_modules() if type(m).__name__ == 'MS_Spiking_Mlp')
    require(len(paths) == 12, 'Expected twelve native MLPs')
    shapes={p:svs.validate_native_mlp(model.get_submodule(p)) for p in paths}
    bn_contract={'policy':a.bn,'eval_dropout_off':True,'original_dense_bn_signature':before,
                 'recalibration_performed':False}
    provenance={'checkpoint_sha256':a.checkpoint_sha256,'data_contract_sha256':digest(plan['data_contract']),
        'source_sha256':digest(plan['source_sha256']),'bn_policy_sha256':digest(bn_contract),
        'geometry_sha256':digest(geometry),'fit_ids_sha256':digest([r['target_id'] for r in rows]),'split':'train'}
    counts_dir=a.output_dir/'counts'; counts_dir.mkdir()
    ys,xs=metrics.crop_slices(a.geometry); input_evidence={}; start=time.monotonic()
    with torch.inference_mode(), CountHooks(model,counts_dir,paths,shapes) as hooks:
        for row in rows:
            rel=native.target_files(row)['input']; path=native.checked_path(a.data_root,rel)
            raw=path.read_bytes(); require(len(raw)==hashes[rel]['bytes'] and hashlib.sha256(raw).hexdigest()==hashes[rel]['sha256'],
                                         'Accepted event input changed')
            import io
            events=np.load(io.BytesIO(raw),allow_pickle=False)
            require(events.shape==(10,480,640) and str(events.dtype)=='float32' and bool(np.isfinite(events).all()), 'Invalid native event array')
            input_evidence[row['target_id']]={'relative_path':rel, **hashes[rel]}
            hooks.begin(row['target_id']); functional.reset_net(model)
            output=model(evaluation.prepare_input(events[:,ys,xs],device,torch))
            require(all(bool(torch.isfinite(v).all()) for v in output['flow']), 'Nonfinite forward flow')
            if device.type=='cuda': torch.cuda.synchronize(device)
            hooks.end(); del output
            print(json.dumps({'collected_target':row['target_id'],'native_MLPs':len(paths)}),flush=True)
        records=hooks.records
    require(before==evaluation.bn_signature(model,torch), 'BN buffers mutated while collecting')
    functional.reset_net(model); del model; gc.collect()
    if device.type=='cuda': torch.cuda.empty_cache()
    manifest={'schema':'flow_svs_counts_v1','status':'ALL_NATIVE_MLP_COUNTS_COMPLETE','collector_sha256':sha(__file__),
        'evaluator_sha256':EVALUATOR_SHA,'static_methods_sha256':STATIC_SHA,'fit_manifest_sha256':FIT_SHA,
        'upstream_collector_sha256':UPSTREAM_COLLECTOR_SHA,'data_manifest_contract_sha256':DATA_CONTRACT_SHA,
        'breadth_root':str(BREADTH_ROOT),
        'prior_root':str(a.prior_root),'counts_root':str(counts_dir),'fit_ids':[r['target_id'] for r in rows],
        'mlp_paths':paths,'shapes':shapes,'files':records,'provenance':provenance,
        'binding_contracts':{'data':plan['data_contract'],'source':plan['source_sha256'],'BN':bn_contract,'geometry':geometry},
        'input_sha256':input_evidence,'BN_buffers_unchanged':True,'training_performed':False,
        'forward_device':str(device),'forward_seconds_including_count_IO':time.monotonic()-start,
        'score_semantics':'complete binary sn2 time counts; CPU float64(counts)/10 then original spatial SVD',
        'scope':'twenty train-gate-fit samples; no GT/check/valid arrays read; no accuracy/method-rejection claim'}
    path=a.output_dir/'counts_manifest.json'; atomic(path,manifest)
    return path,sha(path)


def load_counts_manifest(path, expected):
    require(sha(path)==expected,'Counts manifest SHA differs')
    m=json.loads(Path(path).read_text())
    require(m.get('schema')=='flow_svs_counts_v1' and m.get('status')=='ALL_NATIVE_MLP_COUNTS_COMPLETE' and
            m.get('collector_sha256')==sha(__file__) and m.get('static_methods_sha256')==STATIC_SHA and
            m.get('evaluator_sha256')==EVALUATOR_SHA and m.get('fit_manifest_sha256')==FIT_SHA,'Counts provenance differs')
    ids=m['fit_ids']; paths=m['mlp_paths']
    fixed=fixed_fit_manifest()
    require(m.get('upstream_collector_sha256')==UPSTREAM_COLLECTOR_SHA and
            m.get('data_manifest_contract_sha256')==DATA_CONTRACT_SHA and
            m['provenance'].get('checkpoint_sha256')==CHECKPOINT_SHA and
            m['binding_contracts']['BN'].get('policy')=='P_paper' and
            m['binding_contracts']['geometry'].get('name')=='full9', 'Fit20 provenance changed')
    require(ids==fixed['target_ids'] and len(paths)==len(set(paths))==12,
            'Incomplete fixed fit IDs or MLP paths')
    identities=[(r['target_id'],r['mlp_path']) for r in m['files']]
    require(len(identities)==len(set(identities)) and set(identities)=={(i,p) for i in ids for p in paths},'Counts do not cover every sample/layer')
    bindings=m['binding_contracts']; prov=m['provenance']
    require(prov['split']=='train' and prov['fit_ids_sha256']==digest(ids) and
        all(prov[k]==digest(bindings[v]) for k,v in (('data_contract_sha256','data'),('source_sha256','source'),
                                                    ('bn_policy_sha256','BN'),('geometry_sha256','geometry'))), 'Binding digest differs')
    require(m.get('BN_buffers_unchanged') is True and m.get('training_performed') is False, 'Collection changed model state')
    return m


def score_manifest(path, expected, output_dir, channel_chunk=16):
    import numpy as np
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    require(not torch.cuda.is_initialized(), 'CPU scorer unexpectedly initialized CUDA')
    m=load_counts_manifest(path,expected); methods=static_module(Path(m['prior_root'])); svs=methods._svs()
    root=Path(m['counts_root']); results={}; partial=[]; start=time.monotonic()
    files={(r['target_id'],r['mlp_path']):r for r in m['files']}
    for layer in m['mlp_paths']:
        accumulator=svs.SVSAccumulator(m['fit_ids'],len(m['fit_ids']))
        for target in m['fit_ids']:
            r=files[target,layer]; name=r['file']
            require(Path(name).name==name and name.endswith('.npy') and not (root/name).is_symlink(), 'Unsafe count filename')
            raw=(root/name).read_bytes()
            require(len(raw)==r['bytes'] and hashlib.sha256(raw).hexdigest()==r['sha256'] and
                    r['binary_guard_passed'] is True and r['complete_T']==10, 'Counts bytes/collection guard differ')
            import io
            array=np.load(io.BytesIO(raw),allow_pickle=False)
            require(list(array.shape)==r['shape_BHWF'] and str(array.dtype)==r['dtype']=='uint8' and
                    array.shape[-1]==m['shapes'][layer]['hidden_channels'],'Counts shape/dtype differs')
            ranks=ranks_from_counts(torch.from_numpy(array),channel_chunk)
            # Same sufficient statistic as SVSAccumulator.update; original full
            # T binary array is intentionally not reconstructed in memory.
            if accumulator.rank_sum is None: accumulator.rank_sum=torch.zeros(ranks.shape[1],dtype=torch.int64)
            accumulator.rank_sum+=ranks.sum(0); accumulator.seen_ids.append(target); accumulator.batch_sizes.append(1)
            partial.append({'layer':layer,'target_id':target,'rank_channels':ranks.numel()})
            atomic(Path(output_dir)/'score_progress.json',{'status':'SCORING_INCOMPLETE','completed':partial,
                'scoring_wall_seconds':time.monotonic()-start,'usable_replay_artifact':False})
        results[layer]=accumulator.result()
    artifact={'schema':'flow_breadth_svs_replay_v1','fit_ids':m['fit_ids'],'provenance':m['provenance'],'blocks':results,
        'evidence':{'counts_manifest_sha256':expected,'collector_sha256':sha(__file__),'evaluator_sha256':EVALUATOR_SHA,
            'static_methods_sha256':STATIC_SHA,'upstream_collector_sha256':UPSTREAM_COLLECTOR_SHA,
            'fit_manifest_sha256':FIT_SHA,'data_manifest_contract_sha256':DATA_CONTRACT_SHA,'formula':svs.FORMULA,'CPU_only_SVD':True,'cuda_initialized':torch.cuda.is_initialized(),
            'seconds':time.monotonic()-start,'source_site':'sn2 binary output before drop2/fc2',
            'score_input':'lossless uint8 counts; float64/10; no spatial approximation','sample_count':len(m['fit_ids']),
            'scope':'twenty-sample train-fit exploration, not adequate calibration or recovery proof'}}
    methods.validate_svs_replay(artifact,m['provenance'],m['mlp_paths'],m['shapes'],svs)
    candidate=Path(output_dir)/'svs_replay.worker_candidate.json'
    require(not candidate.exists(),'Do not overwrite a worker candidate')
    atomic(candidate,artifact)
    return artifact


def bounded_worker(argv, timeout_seconds, log_path):
    """CPU worker process-group deadline; SIGTERM/interrupt cannot orphan it."""
    require(0<timeout_seconds<=300,'CPU scoring deadline must be in (0,300]')
    process=None
    with Path(log_path).open('w') as log:
        try:
            process=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1',
                     'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'})
            try: return {'completed':True,'returncode':process.wait(timeout=timeout_seconds),'timed_out':False}
            except subprocess.TimeoutExpired: return {'completed':False,'returncode':None,'timed_out':True}
        finally:
            if process is not None:
                previous={s:signal.signal(s,signal.SIG_IGN) for s in (signal.SIGTERM,signal.SIGINT)}
                try:
                    try: os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()
                finally:
                    for s,handler in previous.items(): signal.signal(s,handler)


def run_scoring(path, expected, output_dir, timeout_seconds=300, channel_chunk=16):
    load_counts_manifest(path,expected)
    argv=[sys.executable,'-B',str(Path(__file__).resolve()),'_worker','--counts-manifest',str(path),
          '--manifest-sha256',expected,'--output-dir',str(output_dir),'--channel-chunk',str(channel_chunk),
          '--breadth-root',str(BREADTH_ROOT)]
    outcome=bounded_worker(argv,timeout_seconds,Path(output_dir)/'cpu_svd.log')
    candidate=Path(output_dir)/'svs_replay.worker_candidate.json'
    if outcome['timed_out']:
        # A just-finished candidate is still not promoted after the deadline.
        status={'status':'PENDING_CPU_SVD_TIME_LIMIT','usable_replay_artifact':False,'score_seconds_cap':timeout_seconds,
                'counts_manifest':str(path),'counts_manifest_sha256':expected,'candidate_present_but_unpublished':candidate.exists()}
    elif outcome['returncode']!=0:
        status={'status':'CPU_SVD_ERROR_NOT_METHOD_FAILURE','usable_replay_artifact':False,'worker':outcome}
    else:
        m=load_counts_manifest(path,expected); methods=static_module(Path(m['prior_root'])); svs=methods._svs()
        artifact=json.loads(candidate.read_text())
        methods.validate_svs_replay(artifact,m['provenance'],m['mlp_paths'],m['shapes'],svs)
        destination=Path(output_dir)/'svs_replay.json'; require(not destination.exists(),'Published SVS artifact immutable')
        os.replace(candidate,destination)
        status={'status':'SVS_REPLAY_READY','usable_replay_artifact':True,'artifact':str(destination),
                'artifact_sha256':sha(destination),'sample_count':len(m['fit_ids']),'native_MLPs':len(m['mlp_paths'])}
    atomic(Path(output_dir)/'score_status.json',status); return status


def parse_args(argv=None):
    global BREADTH_ROOT
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='mode',required=True)
    c=sub.add_parser('collect')
    for key in ('prior-root','source-root','data-root','source-manifest','canonical','roles','archive-audit','acceptance','checkpoint','output-dir'):
        c.add_argument('--'+key,type=Path,required=True)
    for key in ('acceptance-sha256','checkpoint-sha256'): c.add_argument('--'+key,required=True)
    c.add_argument('--bn',choices=('P_paper',),default='P_paper')
    c.add_argument('--geometry',choices=('full9',),default='full9')
    c.add_argument('--breadth-root',type=Path,default=BREADTH_ROOT)
    c.add_argument('--device',default='cuda:0'); c.add_argument('--gpu-lock',type=Path)
    c.add_argument('--cpu-threads',type=int,choices=(1,2),default=2); c.add_argument('--run-inference',action='store_true')
    c.add_argument('--score-seconds',type=float,default=300); c.add_argument('--channel-chunk',type=int,default=16)
    for mode in ('score','_worker'):
        q=sub.add_parser(mode)
        q.add_argument('--breadth-root',type=Path,default=BREADTH_ROOT)
        q.add_argument('--counts-manifest',type=Path,required=True); q.add_argument('--manifest-sha256',required=True)
        q.add_argument('--output-dir',type=Path,required=True); q.add_argument('--channel-chunk',type=int,default=16)
        if mode=='score': q.add_argument('--score-seconds',type=float,default=300)
    a=p.parse_args(argv)
    for k,v in vars(a).items():
        if isinstance(v,Path): setattr(a,k,v.resolve())
    BREADTH_ROOT=a.breadth_root
    if not 1<=a.channel_chunk<=64 or hasattr(a,'score_seconds') and not 0<a.score_seconds<=300: p.error('Invalid bounded SVD limits')
    if a.mode!='_worker' and a.output_dir.exists(): p.error('Use a new output-dir; preserve previous evidence')
    if a.mode=='collect' and a.run_inference and a.device.startswith('cuda') and not a.gpu_lock: p.error('CUDA collection requires the explicit shared --gpu-lock path')
    return a


def main(argv=None):
    a=parse_args(argv)
    if a.mode=='_worker': score_manifest(a.counts_manifest,a.manifest_sha256,a.output_dir,a.channel_chunk); return 0
    a.output_dir.mkdir(parents=True)
    def interrupted(signum,frame): raise InterruptedError('Operator signal '+str(signum))
    signal.signal(signal.SIGTERM,interrupted); signal.signal(signal.SIGINT,interrupted)
    try:
        if a.mode=='score':
            status=run_scoring(a.counts_manifest,a.manifest_sha256,a.output_dir,a.score_seconds,a.channel_chunk)
        else:
            context=preflight(a); atomic(a.output_dir/'preflight.json',context[-1])
            if not a.run_inference: print(json.dumps(context[-1],indent=2)); return 0
            lock=a.gpu_lock.open('a') if a.gpu_lock else contextlib.nullcontext()
            with lock as handle:
                if a.gpu_lock: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
                path,expected=collect_counts(a,context)
                # Keep resource ownership until the forward process exits. SVD
                # child is CPU-only; the parent performs no later GPU work.
                status=run_scoring(path,expected,a.output_dir,a.score_seconds,a.channel_chunk)
        print(json.dumps(status,indent=2)); return 0 if status['usable_replay_artifact'] else 3
    except BaseException as exc:
        atomic(a.output_dir/'failure.json',{'status':'COLLECTION_ERROR_NOT_METHOD_FAILURE','error':str(exc),
               'exception':type(exc).__name__,'traceback':traceback.format_exc(),'usable_replay_artifact':False})
        raise


if __name__=='__main__': raise SystemExit(main())
