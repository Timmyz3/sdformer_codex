"""Execute the handed-off matcher against names from its own probe receipts.

No model weights, GPU, or production files are changed. The second arm only
normalizes names in this tiny harness; it does not validate a bypass function.
"""
from pathlib import Path
import ast
import json
import runpy
import sys

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / 'remote_assets'

class Attention:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        return fn

class Block:
    def __init__(self):
        self.attn = Attention()

class Model:
    def __init__(self, names):
        self.blocks = [(name, Block()) for name in names]

    def named_modules(self):
        return iter(self.blocks)

mask = json.loads((ASSETS / 'binary_deadmask.json').read_text())
result = {}
for tag, mask_key in [('h67ep35', 'h67_ep35'), ('local5ep44', 'local5_ep44')]:
    sys.argv = ['verify_deadskip.py', 'unused-config', 'unused-checkpoint', tag]
    module = runpy.run_path(str(ASSETS / 'verify_deadskip.py'), run_name='_audit_only_')
    tree = ast.parse(module['VERIFY_BLOCK'])
    installer = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_install_skip')
    code = compile(ast.Module(body=[installer], type_ignores=[]), '<original installer>', 'exec')
    data = json.loads((ASSETS / f'deadprobe_A_theta_loaded_{tag}.json').read_text())
    names = sorted({name.split('.attn.')[0] for name in data if '.attn.' in name})
    counts = {}
    for arm in ['original', 'normalized_names_control']:
        supplied = names if arm == 'original' else [n.replace('encoders.swin3d.', '') for n in names]
        ns = {'model': Model(supplied), '_SKIP': module['SKIP_BLOCKS'], '_skip_handles': [],
              'print': lambda *args, **kwargs: None}
        exec(code, ns)
        ns['_install_skip']()
        counts[arm] = len(ns['_skip_handles'])
    claimed = mask[mask_key]['dead_modules']
    rows = {}
    for short in claimed:
        matched = [(name, row) for name, row in data.items() if short + '.' in name or name.endswith(short)]
        rows[short] = [dict(name=name, r=row['r']) for name, row in matched]
    result[tag] = dict(block_names_from_receipt=names, requested_blocks=module['SKIP_BLOCKS'],
                       installed_hook_counts=counts, claimed_dead_rates=rows)
sets = [set(mask[key]['dead_modules']) for key in ['h67_ep35', 'local5_ep44']]
result['inventory'] = dict(checkpoint_module_observations=sum(map(len, sets)),
                           unique_module_locations=len(sets[0] | sets[1]),
                           shared_locations=len(sets[0] & sets[1]))
result['scope'] = 'Matcher-only reproduction; no new network equivalence or timing experiment.'
(ROOT / 'DEADSKIP_MATCH_REVIEW.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({tag: result[tag]['installed_hook_counts'] for tag in ['h67ep35', 'local5ep44']}))
print(json.dumps(result['inventory']))
