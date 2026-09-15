"""Small exact counterexample to finite-threshold-sweep => envelope guarantee.

This is not a counterexample measured on the four captured traces.
Use ten independent identical lanes (A = 4096*I, Y = 5), so group and
per-lane retirement coincide. The protocol is the original T5 e=3 BF format.
"""
import json
from fractions import Fraction
from pathlib import Path


def cycles(threshold):
    a, y, e = 4096, 5, 3
    for j in range(e - 1, -1, -1):
        low = a * (y >> j) * (1 << j)
        high = low + a * ((1 << j) - 1)
        if low >= threshold or high < threshold:
            return 1 + e - j
    raise AssertionError("exact final plane did not lock")


def run():
    final, prefix = -10 * 4096, 30 * 4096
    ks = [Fraction(0), Fraction(1, 2), Fraction(1), Fraction(2), Fraction(4)]
    tested = [dict(k=str(k), threshold=int(final + k * (prefix - final)),
                   cycles=cycles(int(final + k * (prefix - final)))) for k in ks]
    inside = final + Fraction(3, 8) * (prefix - final)
    assert inside == 5 * 4096
    assert all(row["cycles"] == 2 for row in tested)
    assert cycles(int(inside)) == 4
    result = {
        "scope": "logical implication counterexample, not the four-trace distribution",
        "Y": [5] * 10, "A": "4096 times identity", "e": 3,
        "sampled_points": tested,
        "unsampled_point": {"k": "3/8", "threshold": int(inside), "cycles": 4},
        "conclusion": "all five sampled points retire after one data plane; an interior point needs three",
    }
    Path(__file__).with_name("t6_envelope_example.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    run()
