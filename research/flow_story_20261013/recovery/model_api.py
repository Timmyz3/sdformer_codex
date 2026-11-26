"""Pinned recovery structures and safe endpoint loading; never re-SVD an endpoint."""
import copy
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import sys

SCHEMA = 'flow_native_static_recovery_checkpoint_v1'
ARMS = ('dense', 'lowrank_param25', 'svs_hidden25', 'magnitude_hidden25')
STATIC_SHA = '5b1938b2815a8201066db01c9a1d3920aa523116a30b22a6f4883fc86c477aab'
SVS_SHA = '001cdf3e49e63a2eee04671c1752ad43a9b7b08c43e67ef02f04351f595a8273'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def import_pinned(name, path, expected):
    require(sha(path) == expected, 'Implementation SHA differs: '+str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def dependencies(prior_root, breadth_root):
    static = import_pinned('recovery_pinned_static', Path(breadth_root)/'static_methods.py', STATIC_SHA)
    static.configure_prior_root(prior_root)
    svs = import_pinned('recovery_pinned_svs', Path(prior_root)/'transfer_experiments/svs/svs_mlp.py', SVS_SHA)
    return static, svs


def instantiate_native(config, source_root, sources):
    root = Path(source_root).resolve()
    require(len(sources) == 63, 'Expected full pinned63 native source mapping')
    for rel, expected in sources.items():
        path = (root/rel).resolve()
        require(path.is_relative_to(root) and sha(path) == expected, 'Native source mismatch: '+rel)
    sys.path.insert(0, str(root))
    from models.STSwinNet_SNN.Spiking_STSwinNet import MS_SpikingformerFlowNet_en4
    require(Path(inspect.getfile(MS_SpikingformerFlowNet_en4)).resolve().is_relative_to(root), 'Native import root differs')
    return MS_SpikingformerFlowNet_en4(copy.deepcopy(config['model']), copy.deepcopy(config['swin_transformer']))


def state_layout(model):
    return {name: {'shape': list(tensor.shape), 'dtype': str(tensor.dtype)} for name, tensor in model.state_dict().items()}


def semantic_config(config):
    result = copy.deepcopy(config)
    if 'data' in result:
        result['data']['path'] = '<verified_DSEC>/saved_flow_data'
    if 'loader' in result:
        result['loader']['gpu'] = '<logical-device>'
        result['loader']['n_workers'] = '<IO-only-worker-count>'
    return result


def architecture_from_model(model, arm, sources, report):
    require(arm in ARMS, 'Unknown recovery arm')
    blocks = {}
    for path, module in model.named_modules():
        if module.__class__.__name__ != 'MS_Spiking_Mlp':
            continue
        if arm == 'lowrank_param25':
            blocks[path] = {'kind': 'factors', 'layers': {
                name: {'in_features': getattr(module, name).in_features, 'out_features': getattr(module, name).out_features,
                       'rank': getattr(module, name).rank, 'bias': getattr(module, name).up.bias is not None}
                for name in ('fc1', 'fc2')}}
        elif arm in ('svs_hidden25', 'magnitude_hidden25'):
            blocks[path] = {'kind': 'hidden', 'keep_indices': report['blocks'][path]['selection']['keep_indices'],
                            'pe_group_width': 16}
        else:
            blocks[path] = {'kind': 'native'}
    require(len(blocks) == 12, 'Expected12 native MLP blocks')
    return {'schema': 'flow_recovery_architecture_v1', 'arm': arm, 'source_sha256': sources,
            'static_source_sha256': STATIC_SHA, 'svs_source_sha256': SVS_SHA,
            'blocks': blocks, 'state_layout': state_layout(model),
            'parameter_count': sum(p.numel() for p in model.parameters()),
            'reconstruction': 'Fixed dimensions and channel indices; no score recomputation or SVD on endpoint weights'}


def build_initial_model(effective_config, dense_state, arm, *, source_root, prior_root, breadth_root,
                        source_sha256, svs_artifact=None, svs_expected=None):
    """One-time projection from the common dense snapshot, retaining tracked training BN."""
    require(arm in ARMS, 'Unknown arm')
    model = instantiate_native(effective_config, source_root, source_sha256)
    model.load_state_dict(dense_state, strict=True)
    static, _ = dependencies(prior_root, breadth_root)
    report = {'method': 'dense', 'parameters_after': sum(p.numel() for p in model.parameters())}
    if arm != 'dense':
        method = {'lowrank_param25': 'lowrank', 'svs_hidden25': 'svs', 'magnitude_hidden25': 'magnitude'}[arm]
        kwargs = {'keep_ratio': .25}
        if method == 'lowrank':
            kwargs['rank_basis'] = 'parameter_ratio'
        else:
            kwargs['pe_group_width'] = 16
        if method == 'svs':
            kwargs.update(svs_scores=svs_artifact, svs_expected_provenance=svs_expected)
        model, report = static.build_static_candidate(model, method, **kwargs)
    import torch
    require(all(p.requires_grad for p in model.parameters()), 'Recovery contains frozen parameters')
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            require(module.track_running_stats and module.running_mean is not None and module.running_var is not None,
                    'Do not carry P-paper scoring BN changes into training')
    model.train()
    return model, architecture_from_model(model, arm, source_sha256, report), report


def construct_model(effective_config, architecture, *, source_root, prior_root, breadth_root):
    """Build CPU endpoint structure only. Caller strict-loads the saved model_state_dict."""
    require(architecture.get('schema') == 'flow_recovery_architecture_v1' and architecture.get('arm') in ARMS,
            'Unknown recovery architecture')
    require(architecture.get('static_source_sha256') == STATIC_SHA and architecture.get('svs_source_sha256') == SVS_SHA,
            'Architecture implementation source differs')
    model = instantiate_native(effective_config, source_root, architecture['source_sha256'])
    static, svs = dependencies(prior_root, breadth_root)
    paths = {n for n, m in model.named_modules() if m.__class__.__name__ == 'MS_Spiking_Mlp'}
    require(len(paths) == 12 and paths == set(architecture['blocks']), 'Architecture MLP coverage differs')
    arm = architecture['arm']
    if arm in ('svs_hidden25', 'magnitude_hidden25'):
        plan = {}
        for path, block in architecture['blocks'].items():
            require(block['kind'] == 'hidden' and block['pe_group_width'] == 16, 'Hidden architecture differs')
            width = model.get_submodule(path).fc1.out_features
            require(len(block['keep_indices']) == int(width*.25/16)*16, 'Hidden width differs from fixed25 contract')
            plan[path] = block['keep_indices']
        model, _ = svs.narrow_model_deepcopy(model, plan, pe_group_width=16)
    elif arm == 'lowrank_param25':
        for path, block in architecture['blocks'].items():
            require(block['kind'] == 'factors' and set(block['layers']) == {'fc1', 'fc2'}, 'Factor architecture differs')
            for name, spec in block['layers'].items():
                module = model.get_submodule(path)
                old = getattr(module, name)
                require((old.in_features, old.out_features, old.bias is not None) ==
                        (spec['in_features'], spec['out_features'], spec['bias']) and
                        spec['rank'] == static.choose_rank(old, keep_ratio=.25, rank_basis='parameter_ratio'), 'Factor dimensions/rank differ')
                setattr(module, name, static.FactoredLinear(**spec, dtype=old.weight.dtype, device='cpu'))
    else:
        require(all(v == {'kind': 'native'} for v in architecture['blocks'].values()), 'Dense architecture contains changes')
    require(state_layout(model) == architecture['state_layout'] and
            sum(p.numel() for p in model.parameters()) == architecture['parameter_count'], 'Reconstructed state layout differs')
    return model


def read_checkpoint(path, expected_sha256, *, expected_contract=None):
    """Safe CPU deserialization. Evaluation must additionally bind its current data/source contract."""
    import torch
    path = Path(path)
    side = json.loads(path.with_suffix('.json').read_text())
    require(sha(path) == expected_sha256 == side.get('checkpoint_sha256') and side.get('checkpoint_file') == path.name
            and side.get('schema_version') == SCHEMA, 'Recovery checkpoint SHA/schema sidecar differs')
    value = torch.load(path, map_location='cpu', weights_only=True)
    require(value.get('schema_version') == SCHEMA and value.get('contract_sha256') == digest(value['contract'])
            == side.get('contract_sha256') and value.get('architecture_sha256') == digest(value['architecture'])
            == value['contract']['architecture_sha256'], 'Internal recovery contract differs')
    require(expected_contract is None or value['contract'] == expected_contract, 'Resume contract differs')
    require(value['architecture']['arm'] == value['contract']['arm'] and
            value['architecture']['source_sha256'] == value['contract']['source_sha256'], 'Architecture arm/source differs from contract')
    require(semantic_config(value['effective_config']) == value['contract']['effective_config'],
            'Payload configuration differs from semantic recovery contract')
    require(side.get('arm') == value['contract']['arm'] and side.get('architecture_sha256') == value['architecture_sha256'],
            'Checkpoint sidecar architecture/arm differs')
    require(value['training']['visited_targets'] == side.get('visited_targets') and
            value['training']['completed_optimizer_steps'] == side.get('completed_optimizer_steps'), 'Checkpoint cursor sidecar differs')
    require(all(isinstance(v, torch.Tensor) and (not v.is_floating_point() or torch.isfinite(v).all())
                for v in value['model_state_dict'].values()), 'Nonfinite recovery state')
    require({n: {'shape': list(v.shape), 'dtype': str(v.dtype)} for n, v in value['model_state_dict'].items()} ==
            value['architecture']['state_layout'], 'Checkpoint tensors do not match fixed architecture')
    return value, side


def load_recovery_checkpoint(path, expected_sha256, *, source_root, prior_root, breadth_root, expected_contract=None):
    """Return (CPU model, payload, sidecar); strict reconstruction, no endpoint SVD/rescoring."""
    value, side = read_checkpoint(path, expected_sha256, expected_contract=expected_contract)
    model = construct_model(value['effective_config'], value['architecture'], source_root=source_root,
                            prior_root=prior_root, breadth_root=breadth_root)
    model.load_state_dict(value['model_state_dict'], strict=True)
    return model, value, side
