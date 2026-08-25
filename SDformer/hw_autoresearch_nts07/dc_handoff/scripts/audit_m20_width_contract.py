#!/usr/bin/env python3
"""Prove the integer-width bounds used by the M20 moment tile."""

from __future__ import annotations

import argparse
import json


def audit(in_width: int, maximum_population: int, lanes: int) -> dict[str, int | str]:
    if in_width < 2 or maximum_population < 1 or lanes != 16:
        raise ValueError("M20 requires IN_W>=2, positive population, and exactly 16 lanes")
    growth = (maximum_population - 1).bit_length()
    sum_width = in_width + growth
    square_width = 2 * in_width - 1
    sumsq_width = square_width + growth
    input_min = -(1 << (in_width - 1))
    input_max = (1 << (in_width - 1)) - 1
    sum_min = maximum_population * input_min
    sum_max = maximum_population * input_max
    sumsq_max = maximum_population * input_min * input_min
    if not (-(1 << (sum_width - 1)) <= sum_min):
        raise AssertionError("signed sum minimum is not representable")
    if not (sum_max <= (1 << (sum_width - 1)) - 1):
        raise AssertionError("signed sum maximum is not representable")
    if not (sumsq_max <= (1 << sumsq_width) - 1):
        raise AssertionError("unsigned sumsq maximum is not representable")
    return {
        "schema": "m20_dynamic_bn_moment_width_audit_v1",
        "status": "PASS_INTEGER_BOUNDS_NO_ACCUMULATOR_TRUNCATION",
        "lanes": lanes,
        "input_width": in_width,
        "maximum_reduction_population": maximum_population,
        "population_growth_width": growth,
        "count_width": maximum_population.bit_length(),
        "sum_width": sum_width,
        "square_width": square_width,
        "sumsq_width": sumsq_width,
        "sum_minimum": sum_min,
        "sum_maximum": sum_max,
        "sumsq_maximum": sumsq_max,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-width", type=int, default=32)
    parser.add_argument("--maximum-population", type=int, default=4194304)
    parser.add_argument("--lanes", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(audit(args.in_width, args.maximum_population, args.lanes), indent=2))
    for in_width in (2, 8, 16, 32):
        for population in (1, 2, 3, 257, 4194304):
            audit(in_width, population, 16)
    print("PASS M20 width audit 20/20 parameter corners")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
