#!/usr/bin/env python3
"""Exact, stable top-k selection for bounded integer spike counts.

This is a CPU functional reference only. It neither runs SDformer nor models RTL
cycles, power, SRAM ports, PSN dynamics, normalization, or a training gradient.
The histogram implementation is an established selection technique, not a claim
of a novel algorithm. Scores must already be computed and supplied as integers.
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from typing import Sequence


def stable_bucket_topk(scores: Sequence[int], k: int, max_score: int) -> list[bool]:
    """Select exactly k indices, breaking ties by increasing original index."""
    if not isinstance(max_score, int) or isinstance(max_score, bool) or max_score < 0:
        raise ValueError('max_score must be a nonnegative integer')
    if not isinstance(k, int) or isinstance(k, bool) or not 0 <= k <= len(scores):
        raise ValueError('k must be an integer in [0, len(scores)]')
    if any(not isinstance(v, int) or isinstance(v, bool) or not 0 <= v <= max_score
           for v in scores):
        raise ValueError('scores must be integers in [0, max_score]')
    if k == 0:
        return [False] * len(scores)
    histogram = [0] * (max_score + 1)
    for value in scores:
        histogram[value] += 1
    remaining = k
    cutoff = max_score
    while histogram[cutoff] < remaining:
        remaining -= histogram[cutoff]
        cutoff -= 1
    selected = []
    for value in scores:
        choose = value > cutoff or (value == cutoff and remaining > 0)
        selected.append(choose)
        if choose and value == cutoff:
            remaining -= 1
    assert sum(selected) == k
    return selected


def spatial_halo_union(selected: Sequence[tuple[int, int]], height: int,
                       width: int, radius: int = 1) -> set[tuple[int, int]]:
    """Return in-image source coordinates needed by selected DWConv outputs.

    Assumes stride 1, dilation 1, square (2*radius+1) kernel, zero padding, and
    pointwise projection followed by the depthwise convolution. Not a general
    sparse convolution mapper and not a memory-transaction count.
    """
    if min(height, width) <= 0 or radius < 0:
        raise ValueError('height/width must be positive and radius nonnegative')
    result = set()
    for y, x in selected:
        if not (0 <= y < height and 0 <= x < width):
            raise ValueError('selected coordinate outside image')
        for iy in range(max(0, y-radius), min(height, y+radius+1)):
            for ix in range(max(0, x-radius), min(width, x+radius+1)):
                result.add((iy, ix))
    return result


def self_test() -> dict:
    checked = 0
    for n in range(7):
        for scores in itertools.product(range(4), repeat=n):
            for k in range(n+1):
                actual = stable_bucket_topk(scores, k, 3)
                expected = set(sorted(range(n), key=lambda i: (-scores[i], i))[:k])
                assert actual == [i in expected for i in range(n)]
                checked += 1
    rng = random.Random(20260922)
    for _ in range(500):
        n = rng.randrange(1, 451)
        scores = [rng.randrange(33) for _ in range(n)]
        k = rng.randrange(n+1)
        expected = set(sorted(range(n), key=lambda i: (-scores[i], i))[:k])
        assert stable_bucket_topk(scores, k, 32) == [i in expected for i in range(n)]
        checked += 1
    assert len(spatial_halo_union([(1, 1)], 4, 4)) == 9
    assert len(spatial_halo_union([(0, 0)], 4, 4)) == 4
    assert len(spatial_halo_union([], 4, 4)) == 0
    return {'selection_cases': checked, 'halo_checks': 3, 'status': 'PASS',
            'scope': 'CPU functional toy/reference, not model accuracy or hardware speed'}


def demo() -> dict:
    scores = [1, 6, 2, 0, 5, 3, 1, 4]
    mask = stable_bucket_topk(scores, 3, 8)
    # Same number of selected outputs, different actual spatial source demand.
    scattered = [(y, x) for y in range(1, 16, 4) for x in range(1, 16, 4)]
    clustered = [(y, x) for y in range(4, 8) for x in range(4, 8)]
    return {
        'scores': scores, 'keep_k': 3,
        'selected_positions_one_based': [i+1 for i,v in enumerate(mask) if v],
        'conditional_pointwise_K_rows': {'all': 8, 'selected': 3,
            'conditions': 'selection known; pointwise K; independent fixed affine normalization; hard output mask; full required temporal group'},
        'halo_demo': {'image': [16,16], 'selected_output_count': len(scattered),
            'scattered_source_count': len(spatial_halo_union(scattered,16,16)),
            'clustered_source_count': len(spatial_halo_union(clustered,16,16)),
            'scope': 'coordinate count, not SRAM transaction count or an accuracy claim'},
        'independent_random_retention_25_percent_3x3_interior_coverage': 1-(1-.25)**9,
        'histogram_bits_for_450_scores_in_0_to_32': 33*9,
        'histogram_scope': 'histogram counters only, excludes all other state/logic/ports',
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', help='Optional JSON output file; prints results either way')
    args = parser.parse_args()
    result = {'verification': self_test(), 'demo': demo()}
    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
