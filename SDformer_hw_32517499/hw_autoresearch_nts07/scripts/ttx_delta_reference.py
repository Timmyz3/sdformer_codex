"""Bit-exact reference for two-slice incremental dyadic TTX scoring."""

from __future__ import annotations

import itertools


def lane_score64(q: int, k: int) -> int:
    """Return 64 times the H60 lane score for alpha0=1/64."""

    if q not in (0, 1) or k not in (0, 1):
        raise ValueError("q and k must be binary")
    if q == 1 and k == 1:
        return 64
    if q == 0 and k == 0:
        return 1
    return 0


def full_score64(q: list[int], k: list[int]) -> int:
    if len(q) != len(k):
        raise ValueError("q and k widths differ")
    return sum(lane_score64(qi, ki) for qi, ki in zip(q, k))


def delta_score64(
    q_prev: list[int],
    k_prev: list[int],
    q_curr: list[int],
    k_curr: list[int],
) -> tuple[int, int]:
    """Update the previous score only on lanes where Q or K toggles."""

    widths = {len(q_prev), len(k_prev), len(q_curr), len(k_curr)}
    if len(widths) != 1:
        raise ValueError("all vectors must have the same width")
    score = full_score64(q_prev, k_prev)
    updates = 0
    for qp, kp, qc, kc in zip(q_prev, k_prev, q_curr, k_curr):
        if qp != qc or kp != kc:
            score += lane_score64(qc, kc) - lane_score64(qp, kp)
            updates += 1
    return score, updates


def exhaustive_one_lane_check() -> None:
    for qp, kp, qc, kc in itertools.product((0, 1), repeat=4):
        delta, _ = delta_score64([qp], [kp], [qc], [kc])
        full = full_score64([qc], [kc])
        if delta != full:
            raise AssertionError((qp, kp, qc, kc, delta, full))


if __name__ == "__main__":
    exhaustive_one_lane_check()
    print("Delta-TTX exhaustive one-lane equivalence: PASS")

