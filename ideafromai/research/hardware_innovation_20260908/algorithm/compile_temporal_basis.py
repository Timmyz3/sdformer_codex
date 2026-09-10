"""Train-weighted signed bases for the existing S2 temporal dictionaries.

This is ordinary exact linear factorization, a stronger implementation control,
not a new algebra or measured speedup. No GPU or validation fitting is involved.
"""
from itertools import combinations
from pathlib import Path
import json
import numpy as np


def signed_width(a):
    low, high = int(a.min()), int(a.max())
    bits = 1
    while low < -(1 << (bits - 1)) or high >= 1 << (bits - 1):
        bits += 1
    return bits


def compile_one(dictionary, temporal, counts, mapping, words):
    rank = int(np.linalg.matrix_rank(dictionary))
    projected_counts = np.bincount(mapping, weights=counts, minlength=1024).astype(np.int64)
    frequencies = projected_counts[words]
    candidates = []
    for selected in combinations(range(1, len(dictionary)), rank):
        basis = dictionary[list(selected)]
        if np.linalg.matrix_rank(basis) != rank:
            continue
        coordinates_float = np.linalg.lstsq(basis.T, dictionary.T, rcond=None)[0].T
        coordinates = np.rint(coordinates_float).astype(np.int64)
        if np.abs(coordinates).max() > 1 or not np.array_equal(coordinates @ basis, dictionary):
            continue
        compiled = temporal @ basis.T
        candidates.append((int(frequencies @ np.abs(coordinates).sum(1)),
                           signed_width(compiled), selected, coordinates, compiled))
    candidates.sort(key=lambda item: item[:3])
    best_updates, bits, selected, coordinates, compiled = candidates[0]
    # Exact equality for every code, including the zero code.
    assert np.array_equal(compiled @ coordinates.T, temporal @ dictionary.T)
    unique_nonzero_rows = np.unique(dictionary.T, axis=0)
    unique_nonzero_rows = unique_nonzero_rows[unique_nonzero_rows.any(1)]
    folded_updates = int(frequencies @ unique_nonzero_rows.sum(0))
    class_updates = int(frequencies[1:].sum())
    record = {
        'rank': rank,
        'nonzero_code_classes': len(words)-1,
        'ordinary_folded_time_rows': len(unique_nonzero_rows),
        'candidate_scope': 'independent subsets of the existing nonzero codes; exact ternary coordinates only',
        'feasible_bases': len(candidates),
        'selection': 'minimize train-projected source updates, then B signed width, then code indices',
        'selected_code_indices': list(selected),
        'selected_time_words': words[list(selected)].tolist(),
        'coordinates': coordinates.tolist(),
        'projected_train_code_counts': frequencies.tolist(),
        'train_scalar_contributions_per_output_h': {
            'ordinary_folded_time_rows': folded_updates,
            'onehot_seven_classes': class_updates,
            'selected_signed_basis': best_updates},
        'signed_basis_contribution_reduction_vs_folded': 1-best_updates/folded_updates,
        'B_min': int(compiled.min()), 'B_max': int(compiled.max()), 'B_signed_bits': bits,
        'B_nonzero_terms': int(np.count_nonzero(compiled)),
        'exact_code_consumer_identity': True,
        'limits': 'Contribution counts exclude encoding, memory, ports, PSN service and source production. No speed, energy or novelty claim.'}
    return record, coordinates, compiled


def compile_encoder(dictionary, mapping, words):
    active = np.flatnonzero(np.any(dictionary != dictionary[0], axis=0))
    raw = np.arange(1024, dtype=np.int64)
    packed = ((raw[:, None] >> active) & 1) @ (1 << np.arange(len(active)))
    minimal_raw = ((np.arange(1 << len(active))[:, None] >> np.arange(len(active))) & 1) @ (1 << active)
    word_to_code = np.full(1024, -1, dtype=np.int64)
    word_to_code[words] = np.arange(len(words))
    rom = word_to_code[mapping[minimal_raw]]
    assert np.array_equal(rom[packed], word_to_code[mapping])
    return {
        'necessary_source_PSN_rows': active.tolist(),
        'removed_constant_dictionary_rows': np.setdiff1d(np.arange(10), active).tolist(),
        'exact_for_all_1024_raw_gate_words': True,
        'ROM_entries': len(rom), 'ROM_output_bits': 3,
        'ROM_single_copy_bytes': len(rom)*3//8,
        'PSN_source_scalar_MACs_per_channel_before_after': [100, len(active)*10],
        'limits': 'Ordinary exact encoder simplification shared by every route. Full continuous T10 source input is still needed; duplicate dictionary columns alone do not permit removing their separate source gates.'}, rom


def main():
    folder = Path(__file__).resolve().parent/'stage2_temporal_codes'
    data = np.load(folder/'codebooks.npz')
    records, arrays = {}, {}
    for block in range(6):
        name = f's2b{block}'
        record, coordinates, compiled = compile_one(
            data[name+'_dictionary'].astype(np.int64),
            data[name+'_A_int16'].astype(np.int64),
            data[name+'_train_histogram'], data[name+'_word_map'], data[name+'_words'])
        records[name] = record
        record['ordinary_exact_encoder'], encoder = compile_encoder(
            data[name+'_dictionary'], data[name+'_word_map'], data[name+'_words'])
        arrays[name+'_coordinates_int8'] = coordinates.astype(np.int8)
        arrays[name+'_B_int32'] = compiled.astype(np.int32)
        arrays[name+'_selected_code_indices'] = np.array(record['selected_code_indices'])
        arrays[name+'_encoder_rom_uint8'] = encoder.astype(np.uint8)
        arrays[name+'_source_PSN_rows'] = np.array(record['ordinary_exact_encoder']['necessary_source_PSN_rows'])
        print(name, 'rank', record['rank'], 'folded', record['ordinary_folded_time_rows'],
              'selected', record['selected_time_words'],
              'contribution_reduction', record['signed_basis_contribution_reduction_vs_folded'])
    (folder/'signed_basis.json').write_text(json.dumps(records, indent=2)+'\n')
    np.savez_compressed(folder/'signed_basis.npz', **arrays)


if __name__ == '__main__':
    main()
