"""One frozen signature per bank, with full-T direct FTP as the denominator."""
import sys
sys.dont_write_bytecode = True
assert sys.version_info[:2] == (3, 12)
from pathlib import Path
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
PARSER = ROOT.parent / 'mechanism_rebuild_gh_20260906/scripts'
sys.path.insert(0, str(PARSER))
from screen_threshold_packets import sources, EXPECTED

T, BANKS = 10, 8
POPC = np.array([v.bit_count() for v in range(1 << T)], dtype=np.int64)
COEF = np.maximum(POPC - 1, 0)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def histogram(sig):
    h = np.zeros((len(sig), 1 << T), dtype=np.int32)
    np.add.at(h, (np.arange(len(sig))[:, None], sig), 1)
    return h


def execute(sig, selected):
    """Value/order diagnostic only; sequential host loop is not a hardware timeline."""
    C = len(sig)
    c = np.arange(C, dtype=np.int64)
    theta_num = 9 + c % 5
    phi = np.stack([((c * 7 + 3) % 15 - 7) * theta_num,
                    ((c * 11 + 2) % 13 - 6) * theta_num], axis=1)
    remaining = np.array([np.count_nonzero(sig[b::BANKS] == selected[b])
                          for b in range(BANKS)], dtype=np.int64)
    slots = np.zeros((BANKS, 2), dtype=np.int64)
    live = np.zeros(BANKS, dtype=bool)
    Y = np.zeros((T, 2), dtype=np.int64)
    y_live = np.zeros(T, dtype=bool)
    adds, reads = 0, 0

    def emit(value, mask):
        nonlocal adds
        for t in range(T):
            if (mask >> t) & 1:
                if y_live[t]:
                    Y[t] += value
                    adds += 1
                else:
                    Y[t] = value
                    y_live[t] = True

    for channel, mask in enumerate(sig):
        mask = int(mask)
        if not mask:
            continue
        reads += 1
        bank = channel % BANKS
        if mask == int(selected[bank]):
            if live[bank]:
                slots[bank] += phi[channel]
                adds += 1
            else:
                slots[bank] = phi[channel]
                live[bank] = True
            remaining[bank] -= 1
            assert remaining[bank] >= 0
            if remaining[bank] == 0:
                emit(slots[bank], mask)
                live[bank] = False
        else:
            emit(phi[channel], mask)
    assert not live.any() and not remaining.any()
    g = ((sig[None, :] >> np.arange(T, dtype=np.int64)[:, None]) & 1).astype(np.int64)
    ref = g @ phi
    assert np.array_equal(Y, ref)
    direct = int(np.maximum(g.sum(1) - 1, 0).sum())
    saved = sum(max(int(np.count_nonzero(sig[b::BANKS] == selected[b])) - 1, 0)
                * int(COEF[selected[b]]) for b in range(BANKS))
    assert adds == direct - saved and reads == int(np.count_nonzero(sig))
    return {'sources': C, 'coefficient_reads': reads, 'direct_adds': direct,
            'candidate_adds': adds, 'verified_lane_values': T * 2, 'errors': 0}


def main():
    out = ROOT / 'c2_bank_mode_r1.json'
    assert not out.exists()
    plan_path = ROOT / 'c2_bank_mode_plan.json'
    plan_sha = sha(plan_path)
    plan = json.loads(plan_path.read_text())
    selected_toy = np.full(BANKS, 3, dtype=np.int64)
    toy_inputs = [np.zeros(16, dtype=np.int64), np.array([3] + [0] * 15),
                  np.full(16, 3), np.full(16, 5), np.arange(16),
                  np.array([3, 5, 3, 7, 0, 1023, 1, 0] * 3)]
    toys = [execute(s, selected_toy) for s in toy_inputs]
    arrays = sources()
    rows = []
    for stage in [0, 3]:
        name = f'sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.0.mlp.fc2'
        spec, S = arrays[name]
        M, C = S.shape
        P, H = M // T, spec['output_channels']
        assert M % T == 0 and C % BANKS == 0
        sig = np.sum(S.reshape(T, P, C).astype(np.uint16)
                     * (1 << np.arange(T, dtype=np.uint16))[:, None, None],
                     axis=0, dtype=np.uint16)
        n_cal = min(256, P // 4)
        direct = np.maximum(S.reshape(T, P, C).sum(2, dtype=np.int64) - 1, 0).sum(0)
        saved = np.zeros(P, dtype=np.int64)
        selected, per_bank, selected_counts = [], [], []
        for bank in range(BANKS):
            h = histogram(sig[:, bank::BANKS])
            score = (np.maximum(h[:n_cal] - 1, 0) * COEF[None, :]).sum(0)
            score[POPC < 2] = -1
            pattern = int(np.argmax(score))
            counts = h[:, pattern].astype(np.int64)
            saved_bank = np.maximum(counts - 1, 0) * COEF[pattern]
            saved += saved_bank
            selected.append(pattern)
            selected_counts.append(counts)
            per_bank.append({'bank': bank, 'pattern_hex': f'{pattern:03x}',
                             'pattern_popcount': int(POPC[pattern]),
                             'calibration_saved_adds': int(saved_bank[:n_cal].sum()),
                             'evaluation_saved_adds': int(saved_bank[n_cal:].sum()),
                             'evaluation_positions_with_multiple_members': int((counts[n_cal:] > 1).sum())})
        all_h = histogram(sig)
        unrestricted_saved = (np.maximum(all_h - 1, 0) * COEF[None, :]).sum(1)
        assert np.all(saved <= unrestricted_saved) and np.all(unrestricted_saved <= direct)
        coeff_reads = (sig != 0).sum(1, dtype=np.int64)
        diag_indices = sorted(set([0, n_cal - 1, n_cal, n_cal + 1, P // 2, P - 1]))
        diagnostics = []
        for p in diag_indices:
            d = execute(sig[p], np.array(selected))
            assert d['direct_adds'] == int(direct[p])
            assert d['candidate_adds'] == int(direct[p] - saved[p])
            diagnostics.append({'spatial_position': p, **d})
        subsets = []
        for label, lo, hi in [('calibration', 0, n_cal), ('remaining_same_sample', n_cal, P)]:
            d = int(direct[lo:hi].sum())
            s = int(saved[lo:hi].sum())
            subsets.append({'set': label, 'position_start': lo, 'position_end_exclusive': hi,
                            'positions': hi - lo, 'direct_FTP_adds_per_output_channel': d,
                            'one_mode_per_bank_adds_per_output_channel': d - s,
                            'saved_adds_per_output_channel': s,
                            'arithmetic_reduction_fraction': s / max(d, 1),
                            'unrestricted_identical_temporal_group_adds_per_output_channel': int((direct[lo:hi] - unrestricted_saved[lo:hi]).sum()),
                            'coefficient_reads_per_output_channel_both_modes': int(coeff_reads[lo:hi].sum())})
        count_bits = (C // BANKS).bit_length()
        row = {'module': name, 'M': M, 'T': T, 'P': P, 'C': C, 'H': H,
               'calibration_positions': n_cal, 'selected_per_bank': per_bank, 'sets': subsets,
               'explicit_metadata': {'count_bits_per_bank_per_position': count_bits,
                    'count_bits_per_position': BANKS * count_bits,
                    'all_position_count_bits': P * BANKS * count_bits,
                    'selected_pattern_bits_per_layer': BANKS * T,
                    'live_partial_vector_count': BANKS,
                    'live_partial_payload_formula_bits': '8 * active_output_tile_width * accumulator_width',
                    'additional_unmodeled': ['packing count construction and read ports', 'post-join zero retirement',
                         'wider operand path and local adder timing', 'BN2 persistent Y and real residual',
                         'multi-bank same-time output arbitration']},
               'last_member_value_diagnostics': diagnostics}
        rows.append(row)
        print(stage, json.dumps(subsets[-1], ensure_ascii=False), flush=True)
    assert sha(plan_path) == plan_sha
    report = {'status': 'BANK_LOCAL_SINGLE_MODE_ARITHMETIC_OPPORTUNITY_ONLY', 'date': '2026-09-07',
              'plan': plan, 'plan_sha256': plan_sha, 'script_sha256': sha(Path(__file__)),
              'parser_sha256': sha(PARSER / 'screen_threshold_packets.py'), 'capture_sha256': EXPECTED,
              'synthetic_last_member_diagnostics': toys, 'layers': rows,
              'limits': ['This is a new bounded layout of a reviewed 6/10 concept, not a demonstrated strong contribution.',
                 'Remaining positions are within an already-inspected sample, not independent-frame generalization.',
                 'Direct FTP denominator shares coefficients across full T; unrestricted temporal grouping is an additional stronger arithmetic reference.',
                 'A sequential diagnostic host loop proves values under a rational integer contract, not eight-bank throughput or frozen FP32 equality.',
                 'No cycles, PPA, real-weight evaluation, BN2+shortcut closure, VCS or production RTL.']}
    with out.open('x') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
