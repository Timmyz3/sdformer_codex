"""Root's independent arithmetic/source-scope audit; no data, GT or GPU."""
from pathlib import Path
from hashlib import sha256
import json

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
RESEARCH = ROUND.parents[1]
OLD = RESEARCH / 'flow_story_20261021/round34_deployability/hardware'
read = lambda p: json.loads(p.read_text())
digest = lambda p: sha256(p.read_bytes()).hexdigest()

cost_path = ROUND / 'hardware/SOURCE_COST_BOUNDS_V1.json'
cost = read(cost_path)
base = read(OLD / 'actual_v2/rowband128.json')

# Count independently by one cell's loop/reduction lengths.  Three raw scalar
# movements occur in either pass; only one was needed in the fused source.
n = 19200
raw = 10*n
delta = 9*n
cheap_add = 2*(raw-1) + 10*(n-1) + 3*9
cheap_arithmetic = (3*cheap_add + raw + 2*raw+1 + raw + 2*raw
                    + 2 + 3*30 + 16*6)
cheap_scalar_mem = 3*raw+6
late_add = delta + (delta-1) + (n-1)
late_arithmetic = 3*late_add + 3*delta + raw + 16*3
late_scalar_mem = 3*raw+3*delta+3*delta+3*raw+raw+n+2
band_bytes = 4*raw*4
band_gather = 10*32 + 3*(band_bytes//64)
cheap = 16*(cheap_arithmetic+cheap_scalar_mem)+4*band_gather
late = late_arithmetic+late_scalar_mem
assert (cheap,late,band_gather)==(55877120,4204844,144320)
assert cheap+16*late-16*3*raw == base['feature_ready_tick']
assert 2*band_bytes > 4194304 > band_bytes+2*n*4

heads = {'folded128':8599,'ordinary128':26646}
checks = []
for row in cost['schedule_rows']:
    k = row['completed_late_cells']
    bands = (k+3)//4
    new = cheap + k*late + bands*band_gather
    assert row['second_pass_rowbands']==bands
    assert row['second_pass_DMA_bytes']==bands*band_bytes
    assert row['feature_source_floor_ticks_no_zero_cells']==new
    for name,head in heads.items():
        for zeros in (0,16):
            item = row['free_certificate_prefix_saving_by_head'][name][f'zero_mass_cells_{zeros}']
            old = base['feature_ready_tick']+head-32*zeros
            paid = new-32*zeros
            assert item['old_full_prefix_ticks']==old
            assert item['optimistic_new_ticks']==paid
            assert item['saved_ticks']==old-paid
            assert item['passes_10pct_necessary_gate']==(10*(old-paid)>=old)
            assert abs(item['saved_fraction']-(old-paid)/old)<1e-15
    checks.append({'k':k,'source_floor_ticks':new})

out = {
    'status':'PASS_PRESCRIBED_SOURCE_COST_RELAXATION',
    'auditor':'root, independent of hardware cost implementer',
    'auditor_sha256':digest(Path(__file__)),
    'source_cost_sha256':digest(cost_path),
    'old_feature_report_sha256':digest(OLD/'actual_v2/rowband128.json'),
    'stage_checks':checks,
    'stage_head_zero_combinations':17*2*2,
    'old_fused_feature_total_reconstructed':base['feature_ready_tick'],
    'full_miss_extra_raw_copy_ticks':16*3*raw,
    'full_miss_extra_gather_ticks':4*band_gather,
    'all_actual_control_bounds_parameter_costs_optimistically_zero':True,
    'scope':'One prescribed source-like two-pass rowband schedule; not a universal hardware lower bound, cycle replay or exact NumPy implementation.',
    'raw_arrays_read':0,'GT_arrays_read':0,'GPU_calls':0,'RTL_or_DC_runs':0,
}
target = HERE/'SOURCE_COST_INDEPENDENT_V1.json'
assert not target.exists()
target.write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'status':out['status'],'combinations':out['stage_head_zero_combinations'],'sha256':digest(target)}))
