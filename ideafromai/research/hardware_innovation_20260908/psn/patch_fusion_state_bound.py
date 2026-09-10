"""Geometry-only patch fusion comparison; no latency, SRAM, or integer-width claim.

Both routes produce the identical (B+2)^2 sn2 spatial domain for a B^2 Conv2
output tile. Each producer supplies all C=96 / T=10 gate bits. The gather
route drains every newly ready Conv2 output before advancing the producer.
The scatter route likewise finalizes old output accumulators before opening
new ones. Dense potential dependencies bound storage; no captured firing
rate is assumed. Weight/input/residual retention and ports are outside this
small graph check, and are explicitly listed in the result.
"""
from pathlib import Path
import json


def frontier(b):
    side = b + 2
    deps = {}
    users = {q: [] for q in range(side * side)}
    for y in range(1, b + 1):
        for x in range(1, b + 1):
            o = (y - 1) * b + x - 1
            ds = [(y + dy) * side + x + dx
                  for dy in (-1, 0, 1) for dx in (-1, 0, 1)]
            deps[o] = ds
            for q in ds:
                users[q].append(o)

    last = {o: max(ds) for o, ds in deps.items()}
    remaining = {q: len(os) for q, os in users.items()}
    gate_live, acc_live = set(), set()
    gate_peak = acc_peak = 0
    gate_peak_at = acc_peak_at = None
    gather_done = scatter_done = 0
    for q in range(side * side):
        if remaining[q]:
            gate_live.add(q)
        if len(gate_live) > gate_peak:
            gate_peak, gate_peak_at = len(gate_live), q
        for o in users[q]:
            if last[o] == q:
                assert set(deps[o]).issubset(gate_live)
                for p in deps[o]:
                    remaining[p] -= 1
                    if remaining[p] == 0:
                        gate_live.remove(p)
                gather_done += 1

        # A final update needs its live accumulator until that update commits.
        # Finish such updates first; only then allocate first-touch outputs.
        ordered = sorted(users[q], key=lambda o: (last[o] != q,
                                                  o not in acc_live, o))
        for o in ordered:
            acc_live.add(o)
            if len(acc_live) > acc_peak:
                acc_peak, acc_peak_at = len(acc_live), q
            if last[o] == q:
                acc_live.remove(o)
                scatter_done += 1
    assert not gate_live and not acc_live
    assert gather_done == scatter_done == b * b
    return dict(producer_positions=side * side,
                output_positions=b * b,
                dependency_edges=9 * b * b,
                gather_gate_words_peak_including_current_input=gate_peak,
                gather_peak_producer_yx=list(divmod(gate_peak_at, side)),
                scatter_acc_vectors_peak_including_final_update=acc_peak,
                scatter_peak_producer_yx=list(divmod(acc_peak_at, side)))


def tiled_required_positions(height, width, b, radius):
    total = 0
    for y in range(0, height, b):
        for x in range(0, width, b):
            rows = min(height, y + b + radius) - max(0, y - radius)
            cols = min(width, x + b + radius) - max(0, x - radius)
            total += rows * cols
    return total


def main():
    rows = []
    for b in (8, 16):
        r = frontier(b)
        r['output_tile_side'] = b
        r['gate_bitmap_whole_halo_bytes'] = (b + 2)**2 * 10 * 96 // 8
        r['gate_bitmap_frontier_bytes'] = r['gather_gate_words_peak_including_current_input'] * 10 * 96 // 8
        r['conv2_gather_one_time_one_H8_acc_bytes_32bit'] = 8 * 4
        r['conv2_scatter_all_T10_all_H96_frontier_bytes_32bit'] = r['scatter_acc_vectors_peak_including_final_update'] * 10 * 96 * 4
        r['conv2_scatter_all_T10_one_H8_frontier_bytes_32bit'] = r['scatter_acc_vectors_peak_including_final_update'] * 10 * 8 * 4
        r['H8_scatter_requires_gate_replay_or_12_H_groups_live'] = True
        r['uncropped_output_halo_extra_bytes_T10_H96_32bit'] = ((b + 2)**2 - b*b) * 10 * 96 * 4
        r['disjoint_output_tiles_without_halo_reuse'] = {
            'sn2_positions_produced': tiled_required_positions(240, 320, b, 1),
            'sn1_support_positions_needed': tiled_required_positions(240, 320, b, 2),
            'unique_positions': 240 * 320,
        }
        rows.append(r)

    result = {
        'kind': 'exact finite dependency-graph count; illustrative logical bit storage',
        'network': 'patch r1: C96/H96, Conv3x3 -> fixed BN -> noncausal T10 PSN -> theta*g -> Conv3x3 -> fixed BN + continuous identity',
        'identity_boundary': 'same-function scheduling only; no reordering of floating-point arithmetic is validated here',
        'width': '32-bit slots are the current software container size, not an admitted integer range; no 24-bit patch assumption',
        'schedule': 'raster producer; each producer word contains all T10/C96 bits; drain every ready consumer before next producer; no stalls',
        'gather_acc_policy': 'one time/output H8 group at a time; extra replay and weight traffic must be scheduled, not assumed free',
        'scatter_acc_policy': 'all H96/T10 consumers retained; process final-use destinations before first-touch destinations',
        'single_full_image_gate_bitmap_bytes': 240 * 320 * 96 * 10 // 8,
        'full_image_gate_bitmap_one_write_one_read_bytes': 2 * 240 * 320 * 96 * 10 // 8,
        'two_full_dense_conv_weight_bytes': {'FP32': 2 * 96 * 96 * 9 * 4, 'hypothetical_INT8_not_admitted': 2 * 96 * 96 * 9},
        'two_H8_dense_weight_stripes_FP32_bytes': 2 * 8 * 96 * 9 * 4,
        'rows': rows,
        'omitted': [
            'Conv1 source/sn1 production, wide Y/U workspace, and its real precision',
            'coefficient residency/refetch; tiny-frontier schedule need not minimize weight traffic',
            'continuous identity retention or reread and final fixed-BN affine/output traffic',
            'actual spike supports/first-nonzero allocation; current patch capture does not contain spatial sn2 bitmaps',
            'physical bank/read/write ports, metadata, backpressure and halo multicasts',
            'cycle, energy, SRAM macro, area, and overall speed comparisons',
        ],
    }
    out = Path(__file__).with_suffix('.json')
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({'rows': rows, 'result': str(out)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
