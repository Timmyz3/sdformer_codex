"""A legal raw-input counterexample to interpreting local score as full service cost."""
import json
import numpy as np
from audit import HERE, emit_temporal, run

words = np.zeros((96, 4, 4), np.int64)
words[0] = 5  # t0 == t2, t1 and all later time slots are zero.
q1 = np.zeros((8, 96, 3, 3), np.int64)
q1[:2, 0, 0, 0] = 1
q2 = np.zeros((96, 8), np.int64)
q2[:, 0] = 1
q2[:8, 1] = 1  # rank costs are [12, 1, 0, ...].
fixture, tile = emit_temporal('anchor_save_tax', words, q1, q2, [79, 119])
binary = HERE / 'build_temporal_direction/obj/Vconsumer_stream'
rows = []
for mode in [0, 1, 2]:
    rows += run(binary, fixture, mode, 0, tile)
base = rows[0]
delta = rows[2]
assert delta['core_choice_anchor'] == 4
assert delta['core_reference_writes'] == 48
assert delta['core_base_reads'] == 48
assert base['core_mac_issues'] - delta['core_mac_issues'] == 52
assert delta['core_encoder_cycles'] == 88
assert delta['total_cycles'] - base['total_cycles'] == 132
(HERE / 'anchor_tax_results.json').write_text(json.dumps(rows, indent=2) + '\n')
print(json.dumps({k: [base[k], delta[k]] for k in ['total_cycles', 'core_mac_issues', 'core_encoder_cycles', 'core_base_reads', 'core_reference_writes', 'core_choice_anchor']}, indent=2))
