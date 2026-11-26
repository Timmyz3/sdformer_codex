#!/usr/bin/env python3
"""One separately counted final-M forward exports 32 small actual PE vectors.

No optimizer, teacher, GT-value load, or modification of the formal1289 run.
The completed formal audit and final checkpoint are required before CUDA.
"""
import argparse, contextlib, copy, fcntl, hashlib, importlib.util, json
import signal, sys, time, traceback
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
RUNNER_SHA = '7c12c39b5178b0fa7c5b54d65911e9ab32a24fc6309fda08dead152d583df5e2'
AUDITOR_SHA = 'c31a3837c9408126a92bc418d507ba6372c53e07bb892578eb0db2f1628fd975'
CONTRACT_SHA = '5a5018e92bfb632b56788dc20f7a3d7a41ce1b732e73f7e78245040fe2236d16'
TARGETS = (
    'sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp.fc1',
    'sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv1.0',
)
GRAPH = 'flow_r30_joint_Q4_Q8_M_all12mlp_patch4_currentBN_v1'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def payload(a):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def child(folder, name):
    p = folder / name
    require(p.resolve().is_relative_to(folder.resolve()), 'Actual run child artifact')
    return p


def validate_completed(result, audit, result_sha, weights_sha, contract_sha):
    require(result['schema'] == 'flow_r30_joint_recovery_v1' and
            result['graph_id'] == GRAPH and result['arm'] == 'joint',
            'Only actual R30 joint graph')
    require(result['status'] == 'JOINT512_AND256_DEVELOPMENT_PREDICTIONS_COMPLETE' and
            result['engineering_smoke'] is False and
            result['model_forwards'] == result['expected_model_forwards'] == 1289 and
            result['completed_optimizer_steps'] == 512,
            'Complete formal512, never engineering smoke')
    require(audit['schema'] == 'flow_r30_independent_actual_execution_audit_v1' and
            audit['status'] == 'PASS' and audit['engineering_smoke'] is False and
            audit['quality_analysis_allowed'] is True and audit['graph_id'] == GRAPH,
            'Completed independent actual execution audit')
    require(audit['auditor_sha256'] == AUDITOR_SHA and
            audit['source_result_sha256'] == result_sha and
            audit['checkpoint_sha256'] == result['weights_sha256'] == weights_sha and
            audit['source_mask_contract_sha256'] == contract_sha == CONTRACT_SHA,
            'Actual audit binds exact result, final weights and fixed masks')
    require(audit['counts'] == dict(optimizer_updates=512, quality_predictions=256,
                                   formal_model_forwards=1289),
            'Exact audited formal scope')
    require(audit['final_state_sha256'] == result['final_state_sha256'] ==
            result['training_state_sha256'] and result['final_state_sha256'] !=
            result['initial_state_sha256'], 'One changed final joint W')
    require(result['final_parity']['M']['training_deployment']['pass'] and
            result['final_parity']['M']['dense_integer']['pass'],
            'Accepted actual final mixed parity')


def checkpoint_state(cp, result, digest):
    state = cp['model_state_dict']
    require(cp['schema'] == 'flow_r30_joint_checkpoint_v1' and cp['arm'] == 'joint' and
            cp['completed_optimizer_steps'] == 512 and len(state) == 477,
            'Only formal R30 final477 checkpoint')
    require(cp['sources'] == result['sources'] and cp['recipe'] == result['recipe'] and
            cp['joint_contract'] == result['joint_contract'] and
            cp['initial_state_sha256'] == result['initial_state_sha256'],
            'Checkpoint belongs to this actual formal recipe and source')
    require(digest(state) == cp['final_state_sha256'] == result['final_state_sha256'],
            'Actual final tensor digest')
    return state


def coordinates(selected):
    """Predeclared four classes; no GT/activity/quality-driven row choice."""
    import torch
    require(selected.device.type == 'cpu' and selected.dtype == torch.bool and
            selected.ndim == 2, 'Actual CPU spatial mask')
    h, w = selected.shape
    hh, ww = h // 9 * 9, w // 9 * 9
    require(hh < h and ww < w, 'Both actual right and bottom tails required')
    high, low = [], []
    for y in range(h // 9):
        for x in range(w // 9):
            block = selected[y*9:y*9+9, x*9:x*9+9]
            require(bool(block.all()) or not bool(block.any()), 'Whole complete tile')
            (high if bool(block.all()) else low).append((y*9+4, x*9+4))
    require(high and low and not bool(selected[hh:].any()) and
            not bool(selected[:, ww:].any()), 'High/low complete groups; tails implicit')
    return [
        ('restored_complete_Q8', *high[0], True),
        ('low_complete_Q4', *low[0], False),
        ('right_tail_Q8', 0, ww, True),
        ('bottom_tail_Q8', hh, 0, True),
    ]


def binary_row(x, kind, t, b, y, col):
    """Small actual prehook slice; conv order exactly channel,dy,dx."""
    import torch
    if kind == 'linear':
        value = x[t, b, y, col].detach().cpu().contiguous().clone()
    else:
        _, _, c, h, w = x.shape
        value = torch.zeros(c, 3, 3, dtype=x.dtype)
        y0, y1, x0, x1 = max(0, y-1), min(h, y+2), max(0, col-1), min(w, col+2)
        value[:, y0-y+1:y1-y+1, x0-col+1:x1-col+1] = (
            x[t, b, :, y0:y1, x0:x1].detach().cpu())
        value = value.reshape(-1)
    require(value.dtype == torch.float32 and bool(torch.isfinite(value).all()) and
            bool(((value == 0) | (value == 1)).all()), 'Actual binary input slice')
    return value.to(torch.int8)


@contextlib.contextmanager
def capture_small(wrappers, selected, core, target_names=TARGETS):
    """Only two full integer outputs retained transiently, immediately sliced."""
    handles, pending, captures = [], {}, {}
    try:
        for name in target_names:
            module = wrappers[name]
            module.capture_integer = True
            group = core.group_name(name)
            locs = coordinates(selected[group])

            def before(mod, ins, name=name, locs=locs):
                require(name not in pending and name not in captures, 'One actual target call')
                x = ins[0]
                t, b = x.shape[:2]
                h, w = x.shape[2:4] if mod.kind == 'linear' else x.shape[-2:]
                require(t == 10 and b == 1 and tuple(selected[core.group_name(name)].shape) ==
                        (h, w), 'Original full T/B and output geometry')
                q, scale = core.quantize(mod.weight.detach().cpu())
                rows = []
                for time_index in (0, 9):
                    for category, y, col, high in locs:
                        rows.append(dict(category=category, coordinate=[time_index, 0, y, col],
                                         high8=high, input=binary_row(
                                             x, mod.kind, time_index, 0, y, col)))
                pending[name] = dict(rows=rows, q=q, scale=scale, H=h, W=w,
                                     input_shape=list(x.shape),
                                     source_weight_sha256=core.digest(mod.weight))

            def after(mod, ins, out, name=name):
                import torch
                item = pending.pop(name)
                try:
                    integer = mod.last_integer
                    require(integer is not None and integer.dtype == torch.int32 and
                            tuple(integer.shape) == (10*item['H']*item['W'], len(item['q'])),
                            'Actual full row-major int32 source')
                    cases = []
                    for row in item['rows']:
                        t, b, y, col = row['coordinate']
                        index = (t + b) * item['H'] * item['W'] + y * item['W'] + col
                        for start in (0, len(item['q']) - 16):
                            require(start >= 0, 'At least16 actual outputs')
                            cases.append(dict(
                                category=row['category'], coordinate=row['coordinate'],
                                row_index=index, output_start=start, high8=row['high8'],
                                x=row['input'], q8=item['q'][start:start+16].clone(),
                                scale=item['scale'][start:start+16].clone(),
                                expected=integer[index, start:start+16].detach().cpu().clone()))
                    captures[name] = dict(
                        cases=cases, input_shape=item['input_shape'], kind=mod.kind,
                        source_weight_sha256=item['source_weight_sha256'],
                        full_q8_sha256=core.digest(item['q']),
                        full_scale_sha256=core.digest(item['scale']),
                        q8_shape=list(item['q'].shape), diagnostic_quantization_calls=1,
                        actual_work=copy.deepcopy(mod.last_audit))
                finally:
                    mod.last_integer = None
                    mod.capture_integer = False
            handles.append(module.register_forward_pre_hook(before))
            handles.append(module.register_forward_hook(after))
        yield captures
        require(not pending and set(captures) == set(target_names), 'Every fixed capture completed')
    finally:
        for handle in handles:
            handle.remove()
        for name in target_names:
            wrappers[name].last_integer = None
            wrappers[name].capture_integer = False


def export_cases(folder, captures):
    import numpy as np
    entries, arrays, rows, expected_rows = [], {}, [], []
    for name in TARGETS:
        for case in captures[name]['cases']:
            index = len(entries)
            x, q = case['x'].numpy(), case['q8'].numpy()
            actual, scale = case['expected'].numpy(), case['scale'].numpy()
            require(x.dtype == q.dtype == np.int8 and actual.dtype == np.int32 and
                    q.shape == (16, len(x)), 'Fixed actual vector encodings')
            effective = q.astype(np.int64)
            if not case['high8']:
                effective = (effective >> 4) << 4
            golden = effective @ x.astype(np.int64)
            require(np.array_equal(golden, actual.astype(np.int64)), 'Independent int64 actual dot')
            key = f'case{index:02d}'
            for suffix, value in (('x', x), ('q8', q), ('scale', scale), ('actual_int32', actual)):
                arrays[key+'__'+suffix] = value
            entries.append(dict(
                case=index, target=name, category=case['category'],
                coordinate=case['coordinate'], row_index=case['row_index'],
                output_start=case['output_start'], outputs=16, K=len(x),
                mode='Q8' if case['high8'] else 'Q4',
                payloads={suffix: dict(key=key+'__'+suffix, sha256=payload(value),
                                       shape=list(value.shape), dtype=str(value.dtype))
                          for suffix, value in (('x', x), ('q8', q), ('scale', scale),
                                                ('actual_int32', actual))},
                independent_int64_equal=True))
            rows.append(f"{len(x)} {int(case['high8'])}\n" +
                        ' '.join(map(str, x.tolist()))+'\n' +
                        ' '.join(map(str, q.reshape(-1).tolist()))+'\n')
            expected_rows.append(' '.join(map(str, actual.tolist()))+'\n')
    require(len(entries) == 32, 'Exactly32 fixed vectors')
    vector_file = folder/'final_vectors_v1.txt'
    expected_file = folder/'actual_expected_int32.txt'
    archive = folder/'final_vector_payloads_v1.npz'
    vector_file.write_text('32\n'+''.join(rows))
    expected_file.write_text('32\n'+''.join(expected_rows))
    np.savez_compressed(archive, **arrays)
    return dict(cases=entries, vectors_file=vector_file.name, vectors_sha256=sha(vector_file),
                expected_file=expected_file.name, expected_sha256=sha(expected_file),
                payload_file=archive.name, payload_file_sha256=sha(archive))


def prepare(a):
    import torch
    require(not torch.cuda.is_initialized() and not a.output.exists(), 'Fresh CPU preflight')
    require(sha(__file__) == a.source_sha256, 'Explicit new exporter source')
    implementation = a.r30_root/'implementation'
    require(sha(implementation/'run_joint_recovery.py') == RUNNER_SHA and
            sha(a.r30_root/'audit_actual_v1.py') == AUDITOR_SHA, 'Frozen production and auditor')
    sys.path.insert(0, str(implementation))
    spec = importlib.util.spec_from_file_location('_r30_final_vector_helpers',
                                                 implementation/'run_joint_recovery.py')
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    production = copy.copy(a)
    production.runner_sha256, production.smoke = RUNNER_SHA, False
    smoke, helper, p, members, first, contract, refs = runner.prepare(production)
    require(sha(a.audit) == a.audit_sha256, 'Explicit actual audit bytes')
    result, audit = read(a.formal_result), read(a.audit)
    weights = child(a.formal_result.parent, result['weights_file'])
    validate_completed(result, audit, sha(a.formal_result), sha(weights), a.mask_contract_sha256)
    require(result['sources'] == dict(
        runner=RUNNER_SHA, core=runner.CORE_SHA, mask_helper=runner.MASK_SHA,
        smoke_helpers=runner.SMOKE_SHA, formal_helpers=runner.HELPER_SHA,
        members=runner.MEMBERS_SHA, mask_contract=CONTRACT_SHA),
        'Executed final model uses these exact production helpers')
    require(result['final_parity']['M']['input_target_id'] == first['target_id'] and
            result['final_parity']['M']['input_evidence'] == helper.source_evidence(
                members, first['target_id']), 'Same formal full parity training identity')
    selected, layout = runner.masks.materialize(contract, 'full9')
    require(layout == result['final_parity']['M']['mask_layout'], 'Exact formal full M layout')
    for target in TARGETS:
        coordinates(selected[runner.core.group_name(target)])
    return runner, smoke, helper, p, members, first, selected, layout, result, weights


def execute(a, context, receipt):
    import numpy as np
    import torch
    runner, smoke, helper, p, members, first, selected, layout, formal, weights = context
    require(not torch.cuda.is_initialized(), 'GPU lock before CUDA')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20261019)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    torch.cuda.manual_seed_all(20261019)
    torch.cuda.reset_peak_memory_stats()
    start = time.monotonic()
    model, functional, bn_scope = smoke.make_model(p, 'full9', device, torch)
    cp = torch.load(weights, map_location='cpu', weights_only=True)
    # r13.state_digest expects a model. Hash state tensors without model reconstruction.
    def state_digest(state):
        h = hashlib.sha256()
        for name in sorted(state):
            v = state[name].detach().cpu().contiguous()
            h.update(name.encode()); h.update(str(v.dtype).encode())
            h.update(str(tuple(v.shape)).encode()); h.update(v.numpy().tobytes())
        return h.hexdigest()
    state = checkpoint_state(cp, formal, state_digest)
    model.load_state_dict(state, strict=True)
    del cp, state
    model.eval().requires_grad_(False)
    r13 = p['modules'][0]
    before = r13.state_digest(model)
    require(before == formal['final_state_sha256'], 'Final W loaded before only inference')
    signature = p['base'].bn_signature(model, torch)
    evidence = helper.source_evidence(members, first['target_id'])
    events = helper.event_only(p, first, evidence, np)
    chunk = p['base'].prepare_input(events, device, torch)
    del events
    functional.reset_net(model)
    psn_calls, psn_handles = {}, []
    with runner.core.installed(model, 'M', masks=selected, ste=False) as wrappers:
        with helper.observed(model, runner.core, torch) as (bn_calls, affine_calls):
            try:
                for name, module in model.named_modules():
                    if type(module).__name__ == 'PSN':
                        psn_calls[name] = 0
                        def count(mod, ins, out, name=name):
                            psn_calls[name] += 1
                        psn_handles.append(module.register_forward_hook(count))
                with capture_small(wrappers, selected, runner.core) as captured:
                    with torch.no_grad():
                        flows = model(chunk)['flow']
                    receipt['model_forwards'] += 1
            finally:
                for handle in psn_handles:
                    handle.remove()
        work = {name: copy.deepcopy(module.last_audit) for name, module in wrappers.items()}
    require(len(psn_calls) == 105 and set(psn_calls.values()) == {1}, 'All actual105 PSN calls')
    head_checks = {}
    for i, flow in enumerate(flows):
        key = 'head'+str(i)
        reference = formal['final_parity']['M']['training_deployment']['checks'][key]
        require(flow.shape == (1, 2, 480, 640) and flow.dtype == torch.float32 and
                bool(torch.isfinite(flow).all()), 'Actual full finite head')
        actual_sha = runner.core.digest(flow)
        require(actual_sha == reference['actual_sha256'] == reference['reference_sha256'],
                'Exact formal finalM head '+key)
        head_checks[key] = dict(shape=list(flow.shape), dtype=str(flow.dtype),
                                actual_sha256=actual_sha, formal_sha256=reference['actual_sha256'],
                                bitwise_equal=True)
    require(len(flows) == 4 and receipt['model_forwards'] == 1 and
            r13.state_digest(model) == before and p['base'].bn_signature(model, torch) == signature,
            'Single separately counted forward; all final state/BN unchanged')
    vectors = export_cases(a.output, captured)
    receipt.update(status='FINAL_M_VECTOR_EXPORT_COMPLETE', **vectors,
        source_state_sha256=before, final_state_sha256=r13.state_digest(model),
        input_target_id=first['target_id'], input_evidence=evidence, mask_layout=layout,
        BN_scope=bn_scope, actual_BN_calls=bn_calls, actual_affine_calls=affine_calls,
        actual_PSN_calls=psn_calls, work=work, heads=head_checks,
        capture_sources={name: {k: v for k, v in record.items() if k != 'cases'}
                         for name, record in captured.items()},
        diagnostic_quantization_calls=2, optimizer_updates=0, GT_values_read=False,
        quality_computed=False, wall_seconds=time.monotonic()-start,
        max_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_reserved_bytes=torch.cuda.max_memory_reserved())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for key in ('r30-root', 'r28-root', 'r26-root', 'members', 'mask-contract',
                'reference-contract', 'formal-result', 'audit', 'output'):
        ap.add_argument('--'+key, type=Path, required=True)
    for key in ('source-sha256', 'mask-contract-sha256', 'audit-sha256'):
        ap.add_argument('--'+key, required=True)
    ap.add_argument('--preflight-output', type=Path)
    ap.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    context = prepare(a)
    receipt = dict(schema='flow_r30_final_PE_vectors_v1', status='PREFLIGHT_PASS',
                   graph_id=GRAPH, exporter_sha256=sha(__file__), formal_model_forwards=1289,
                   expected_model_forwards=1, model_forwards=0, wall_cap_minutes=10,
                   formal_result_sha256=sha(a.formal_result), actual_audit_sha256=sha(a.audit),
                   checkpoint_sha256=sha(context[-1]), mask_contract_sha256=CONTRACT_SHA,
                   sources=context[-2]['sources'], GT_values_read=False, quality_computed=False,
                   scope='Separate post-audit final-M vector capture; no optimizer or quality evaluation')
    if not a.execute:
        require(a.preflight_output is not None and not a.preflight_output.exists(),
                'Fresh CPU preflight receipt')
        write(a.preflight_output, receipt)
        return
    lock = context[3]['modules'][3].GPU_LOCK.open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output.mkdir(parents=True, exist_ok=False)
        def alarm(signum, frame):
            raise TimeoutError('Fixed10minute one-forward exporter cap')
        signal.signal(signal.SIGALRM, alarm)
        signal.alarm(600)
        try:
            execute(a, context, receipt)
        except BaseException as error:
            receipt.update(status='FINAL_M_VECTOR_EXPORT_FAILED', error=str(error),
                           traceback=traceback.format_exc())
            write(a.output/'FINAL_VECTOR_MANIFEST_V1.json', receipt)
            raise
        finally:
            signal.alarm(0)
        write(a.output/'FINAL_VECTOR_MANIFEST_V1.json', receipt)
    finally:
        lock.close()


if __name__ == '__main__':
    main()
