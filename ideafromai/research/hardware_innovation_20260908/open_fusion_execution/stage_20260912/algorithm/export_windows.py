"""Export existing combination captures without their large full-frame tensors."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--axes', nargs='+', default=['ordinary', 'lifting_raw'])
    args = parser.parse_args()
    index = {}
    for axis in args.axes:
        source = HERE / 'captures' / axis / '000_zurich_city_09_a_0001.npz'
        target = HERE / 'hardware_exports' / axis
        target.mkdir(parents=True, exist_ok=True)
        with np.load(source) as captured:
            omitted = [key for key in captured.files if key.startswith('full_') or key in
                       ('proj_bn_full_input_fp32', 'proj_bn_full_output_fp32')]
            arrays = {key: captured[key] for key in captured.files if key not in omitted}
        np.savez_compressed(target / source.name, **arrays)
        for name in ('deployed_constants.npz', 'student_parameters.npz', 'BN_parameters.npz'):
            shutil.copyfile(HERE / 'combinations' / axis / name, target / name)
        shutil.copyfile(source.parent / 'live_parameters.npz', target / 'live_parameters.npz')
        index[axis] = dict(
            frame='zurich_city_09_a_0001.npy', source=str(source),
            retained={key: dict(shape=list(value.shape), dtype=str(value.dtype))
                      for key, value in arrays.items()}, omitted=omitted,
            scope='Real R24+onepass combination: original corner/interior windows, full-domain BN statistics, actual source and both consumers.',
            constants='deployed_constants.npz contains the actually replaced PED R24; student_parameters.npz retains the original temporal student archive.',
            full_frame_archives='Remain in ignored captures/ on the A800, not in this small export.',
            hardware_execution=False)
    (HERE / 'hardware_exports' / 'index.json').write_text(
        json.dumps(index, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({axis: len(record['retained']) for axis, record in index.items()}))


if __name__ == '__main__':
    main()
