#!/usr/bin/env python3
"""Small integer counterexamples only; no Claude writes, trace replay, or RTL run."""
import itertools
import json
from fractions import Fraction


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def cert_cycles(y, a, threshold, e, head_check=False):
    # Mirrors T10's prefix recurrence for these e>0 legal examples.
    assert e > 0 and all(v >> e == -(v < 0) for v in y)
    v = [dot(row, [-int(x < 0) for x in y]) for row in a]
    p = [sum(max(w, 0) for w in row) for row in a]
    n = [sum(min(w, 0) for w in row) for row in a]
    locked = [False] * len(a)

    def check(m):
        for t in range(len(a)):
            lo = (v[t] << m) + n[t] * ((1 << m) - 1)
            hi = (v[t] << m) + p[t] * ((1 << m) - 1)
            locked[t] |= lo >= threshold[t] or hi < threshold[t]
        return all(locked)

    if head_check and check(e):
        return 1
    for fed, m in enumerate(range(e - 1, -1, -1), 1):
        plane = [(x >> m) & 1 for x in y]
        v = [2 * x + dot(row, plane) for x, row in zip(v, a)]
        if check(m):
            return 1 + fed
    raise AssertionError("exact final prefix did not lock")


def reconstruct(y, e):
    v = -int(y < 0)
    for m in range(e - 1, -1, -1):
        v = 2 * v + ((y >> m) & 1)
    return v


def main():
    a = [[4096 * int(t == s) for s in range(10)] for t in range(10)]
    head_old = cert_cycles([1] * 10, a, [0] * 10, 1)
    head_new = cert_cycles([1] * 10, a, [0] * 10, 1, head_check=True)
    assert (head_old, head_new) == (2, 1)

    width_old = cert_cycles([-4] * 10, a, [-3 * 4096] * 10, 3)
    width_new = cert_cycles([-4] * 10, a, [-3 * 4096] * 10, 2)
    assert (width_old, width_new) == (4, 3)
    assert reconstruct(-4, 2) == reconstruct(-4, 3) == reconstruct(-4, 4) == -4

    # Exact e=bit_length(max(abs(y)))=1, all sign=0: at least one bit is 1.
    admissible_sums = [sum(y) for y in itertools.product((0, 1), repeat=10) if any(y)]
    assert len(admissible_sums) == 1023
    assert (min(admissible_sums), max(admissible_sums)) == (1, 10)

    sampled = {}
    for k in map(Fraction, (0, 0.5, 1, 2, 4)):
        threshold = -10 * 4096 + k * 40 * 4096
        sampled[str(k)] = cert_cycles([5] * 10, a, [threshold] * 10, 3)
    k = Fraction(3, 8)
    inside = cert_cycles([5] * 10, a, [-10 * 4096 + k * 40 * 4096] * 10, 3)
    assert set(sampled.values()) == {2} and inside == 4

    print(json.dumps({"head_check_cycles": [head_old, head_new],
                      "negative_power_width_cycles": [width_old, width_new],
                      "widening_reconstruction": [-4, -4, -4],
                      "exact_e_domain": {"valid_vectors": 1023, "sum_bound": [1, 10],
                                         "relaxed_box_bound": [0, 10]},
                      "T6_sampled_cycles": sampled, "T6_inside_3_over_8_cycles": inside},
                     separators=(",", ":")))


if __name__ == "__main__":
    main()
