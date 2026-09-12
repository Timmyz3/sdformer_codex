"""Recompute this stage's paired AEE tables from the actual completed frame rows."""
from pathlib import Path
import csv
import json
import statistics

HERE = Path(__file__).resolve().parent
OPEN = HERE.parents[1]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    combo = read(HERE / 'combinations/run.json')
    representations = read(HERE / 'representations/aee/run.json')
    weights = read(HERE / 'weight_controls/aee/run.json')
    source34 = read(HERE / 'source34/aee/run.json')
    assert combo['complete'] and representations['complete'] and weights['complete'] and source34['complete']
    with (HERE / 'source_nb0_valid825.csv').open() as stream:
        nb0 = {row['file']: row for row in csv.DictReader(stream)}
    original825 = {row['axis']: row for row in
                   read(OPEN / 'accuracy_baseline/valid825_summary.json')['arms']}
    all_rows = []
    results = []

    def evaluate(axis, mode, population, measured, activity_path, extra):
        names = [row['file'] for row in measured]
        assert len(set(names)) == len(names)
        assert len(names) == (825 if population == 'valid825' else 10)
        assert set(names) == (set(nb0) if population == 'valid825' else set(combo['files']))
        assert all(int(row['valid_pixels']) == int(float(nb0[row['file']]['valid_pixels']))
                   for row in measured)
        pixels = sum(int(row['valid_pixels']) for row in measured)
        mean = statistics.mean(row['AEE'] for row in measured)
        reference = statistics.mean(float(nb0[name]['AEE']) for name in names)
        deltas = [row['AEE'] - float(nb0[row['file']]['AEE']) for row in measured]
        item = dict(axis=axis, mode=mode, population=population, frames=len(measured),
                    valid_pixels=pixels, AEE_frame_mean=mean,
                    AEE_pixel_mean=sum(row['AEE'] * int(row['valid_pixels']) for row in measured) / pixels,
                    NB0_AEE_frame_mean=reference, delta_NB0=mean-reference,
                    relative_AEE_reduction_NB0=1-mean/reference,
                    better_than_NB0=mean < reference,
                    paired_frames=dict(better=sum(delta < 0 for delta in deltas),
                                       equal=sum(delta == 0 for delta in deltas),
                                       worse=sum(delta > 0 for delta in deltas)),
                    same_frame_set=True, same_per_frame_valid_pixels=True, **extra)
        if population == 'diverse10':
            hold = [row for row in measured if row['file'] != combo['files'][0]]
            item['holdout9'] = dict(frames=len(hold),
                valid_pixels=sum(int(row['valid_pixels']) for row in hold),
                AEE_frame_mean=statistics.mean(row['AEE'] for row in hold),
                NB0_AEE_frame_mean=statistics.mean(float(nb0[row['file']]['AEE']) for row in hold))
            item['holdout9']['better_than_NB0'] = (item['holdout9']['AEE_frame_mean'] <
                                                  item['holdout9']['NB0_AEE_frame_mean'])
        activity = read(activity_path)
        observed = activity['frames']
        elements = sum(row['elements'] for row in observed)
        item['activity'] = dict(actual_frames=len(observed),
            ATLIF_called_module_entries=len(activity['ATLIF_layers']),
            ATLIF_output_elements=elements,
            ATLIF_nonzero_fraction=sum(row['nonzero'] for row in observed)/elements,
            scope=activity['ATLIF_hook_scope'],
            fixed_helper_gate_totals=activity['fixed_helper_gate_totals'],
            clip_counts=activity['clip_counts'], state_ranges=activity['state_ranges'],
            onepass_calls=activity['onepass_calls'], onepass_min_variance=activity['onepass_min_variance'])
        results.append(item)
        for row in measured:
            all_rows.append(dict(axis=axis, mode=mode, population=population, file=row['file'],
                valid_pixels=row['valid_pixels'], AEE=row['AEE'],
                NB0_AEE=float(nb0[row['file']]['AEE']), delta_NB0=row['AEE']-float(nb0[row['file']]['AEE'])))

    for axis, record in combo['axes'].items():
        assert record['complete']
        for population in ('diverse10', 'valid825'):
            directory = HERE / 'combinations' / axis / population
            extra = dict(identity=record['identity'], PED_mode=record['PED_mode'],
                         BN_function='onepass paired256 tree, existing seed+3Newton, separate MUL/ADD',
                         parameter_directory=str(directory.parent.relative_to(HERE)))
            if population == 'diverse10':
                extra['single_item_controls'] = record['stages'][population]['single_item_controls']
            else:
                extra['historical_original32_CUDA_AEE'] = original825[axis]['metrics']['AEE_frame_mean']
            measured = read(directory / 'combo_frames.json')
            evaluate(axis, 'R24_onepass', population, measured, directory/'activity_summary.json', extra)
    for axis, record in representations['axes'].items():
        assert record['complete']
        for mode, item in record['modes'].items():
            directory = HERE / 'representations/aee' / axis / mode
            quant = read(directory / 'representation_activity.json')
            count = sum(row['elements'] for row in quant)
            words = sum(row['P2_T10_source_words'] for row in quant)
            extra = dict(parent='R24_onepass', parent_diverse10_AEE=record['baseline_diverse10'],
                delta_parent=item['delta_to_fixed_R24_onepass'], full825=False,
                same_actual_proj_gate=all(row['same_actual_proj_gate'] for row in quant),
                representation_activity=dict(elements=count,
                    quant_zero_fraction=sum(row['quant_zero'] for row in quant)/count,
                    quant_clipped=sum(row['quant_clipped'] for row in quant),
                    empty_P2_T10_source_fraction=sum(row['empty_P2_T10_source_words'] for row in quant)/words,
                    input_physical_MAE=statistics.mean(row['input_physical_MAE'] for row in quant)))
            evaluate(axis, mode, 'diverse10', read(directory/(mode+'_frames.json')),
                     directory/'activity_summary.json', extra)
    for axis, record in weights['axes'].items():
        assert record['complete']
        for mode, item in record['modes'].items():
            directory = HERE / 'weight_controls/aee' / axis / mode
            evaluate(axis, 'U_only_'+mode, 'diverse10', read(directory/(mode+'_frames.json')),
                     directory/'activity_summary.json', dict(parent='original32_CUDA',
                         parent_diverse10_AEE=record['baseline_diverse10'],
                         delta_parent=item['delta_parent'], full825=False,
                         source_package='stage_20260912/weight_compensation',
                         hardware_compressed_execution=False))
    source_directory = HERE / 'source34/aee/diverse10'
    evaluate('ordinary', 'source34_contiguous334', 'diverse10',
             read(source_directory/'source34_frames.json'), source_directory/'activity_summary.json',
             dict(parent='R24_onepass', parent_diverse10_AEE=source34['baseline_diverse10'],
                  delta_parent=source34['delta_parent'], full825=False,
                  source_package='stage_20260912/hardware/structured_source_control',
                  source_function='As_q16 contiguous3/3/4, unchanged exponent15/RNE15/sat24 and predicate'))
    save(HERE/'stage_summary.json', dict(complete=True, fresh_inference=True,
        training=False, EDA=False, hardware_speedup=False, rows=results,
        baseline='Local upstream SDformerFlow NB0; not the authors checkpoint. Same DSEC population and valid-pixel counts.',
        metric='Frame mean is the admission metric. Pixel mean is reported separately.',
        caveat='Derived coarse-head candidates vs original final-head NB0 is a task-quality comparison; no numerical-equivalence claim.',
        novelty='R24 blocking and onepass BN are common baselines. Dg prediction/quantization alone is not X; new execution and strongest-control advantage remain unproved.'))
    with (HERE/'paired_frames.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    print(json.dumps([{key: row[key] for key in ('axis','mode','population','AEE_frame_mean','better_than_NB0')}
                      for row in results], indent=2))


if __name__ == '__main__':
    main()
