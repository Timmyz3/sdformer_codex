"""Exhaustive algebra check; no workload/performance/RTL admission."""
import itertools
import json
from pathlib import Path


def scores(grids, shape, q, velocities):
    height, width = shape
    out = []
    for vx, vy in velocities:
        r = ub = denominator = 0
        for age, cells in enumerate(grids, 1):
            x, y = q[0] - age * vx, q[1] - age * vy
            if 0 <= x < width and 0 <= y < height:
                denominator += 1
                r += (x, y) in cells
                ub += any(a == x for a, _ in cells) and any(b == y for _, b in cells)
        if denominator:
            out.append((vx, vy, r, ub, denominator))
    return out


def tie_key(row):
    return abs(row[0]) + abs(row[1]), row[0], row[1]


def beats(a, b, upper=False):
    ai = 3 if upper else 2
    left, right = a[ai] * b[4], b[2] * a[4]
    return left > right or (left == right and tie_key(a) < tie_key(b))


def winner_exact(rows):
    best = rows[0]
    for row in rows[1:]:
        if beats(row, best):
            best = row
    return best[:2]


def winner_pruned(rows):
    best = rows[0]
    verified, pruned = 1, 0
    for row in rows[1:]:
        if not beats(row, best, upper=True):
            pruned += 1
            continue
        verified += 1
        if beats(row, best):
            best = row
    return best[:2], verified, pruned


def main():
    velocities = list(itertools.product(range(-1, 2), repeat=2))
    points = list(itertools.product(range(2), repeat=2))
    bound_checks = winner_checks = 0
    for mask in range(1 << 8):
        grids = [set(p for j, p in enumerate(points) if (mask >> (4 * t + j)) & 1) for t in range(2)]
        for query in points:
            rows = scores(grids, (2, 2), query, velocities)
            for row in rows:
                assert 0 <= row[2] <= row[3] <= row[4]
                bound_checks += 1
            assert winner_exact(rows) == winner_pruned(rows)[0]
            winner_checks += 1
    ghost = scores([{(2, 0), (1, 2)}, {(2, 1), (0, 2)}], (3, 3), (2, 2), velocities)
    assert next(r for r in ghost if r[:2] == (0, 0))[2:] == (0, 2, 2)
    assert winner_exact(ghost) == (1, 0)
    return {
        "status": "PASS_ALGEBRA_ONLY",
        "domain": "All 256 two-bin 2x2 binary histories; all four query points; nine integer velocities; zero-length paths excluded; L1 then lexicographic ties.",
        "bound_checks": bound_checks,
        "winner_checks": winner_checks,
        "ghost_example_rows": ghost,
        "ghost_exact_winner": winner_exact(ghost),
        "not_evidence_for": ["DSEC opportunity", "flow accuracy", "hardware speedup", "memory benefit", "RTL correctness"],
        "PPA_ADMISSION": 0,
        "RTL_SPEEDUP_ADMISSION": 0,
    }


if __name__ == "__main__":
    result = main()
    output = Path(__file__).with_name("event_flow_bound_toy_check.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
