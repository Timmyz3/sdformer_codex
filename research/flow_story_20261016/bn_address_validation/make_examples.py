#!/usr/bin/env python3
"""Write bounded, synthetic resource sensitivity and an inspectable odd trace."""
import json
from pathlib import Path
import address_model as m


def write_new(path, value):
    if path.exists(): raise RuntimeError('Preserve existing artifact: ' + str(path))
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    here = Path(__file__).resolve().parent
    base = m.default_config()
    configs = dict(base=base, shared_ports=dict(base, shared_read_write=True),
                   one_operation=dict(base, max_active_operations=1, register_words=3),
                   four_banks=dict(base, banks=4))
    examples = []
    for name, cfg in configs.items():
        write_new(here / ('example_' + name + '.json'), cfg)
        for samples, channels in ((5, 3), (5, 768)):
            result = m.reduce_group(samples, channels, cfg)
            replay = m.validate_trace(result)
            examples.append(dict(name=name, samples=samples, channels=channels,
                synthetic_component_cycles=result['synthetic_component_cycles'],
                transaction_counts=result['transaction_counts'],
                peak_active_operations=result['peak_active_operations'],
                bank_conflict_requests=result['bank_conflict_requests'], replay=replay))
            if name == 'base' and channels == 3:
                write_new(here / 'odd_5x3_full_trace.json', result)
    write_new(here / 'synthetic_sensitivity.json', dict(schema='flow_reduction_address_sensitivity_v1',
        status='COMPLETE_SYNTHETIC_COMPONENT_ONLY', source_sha256={str(Path(__file__).resolve()):m.sha(__file__),
        str(Path(m.__file__).resolve()):m.sha(m.__file__)}, cases=examples,
        calibrated_hardware_cycles=None, full_BN_cycles=None, full_network_cycles=None,
        policy_speedup=None, RTL_validated=False, native_FP32_equivalence_validated=False))
    print(json.dumps(dict(cases=len(examples), status='COMPLETE_SYNTHETIC_COMPONENT_ONLY')))


if __name__ == '__main__': main()
