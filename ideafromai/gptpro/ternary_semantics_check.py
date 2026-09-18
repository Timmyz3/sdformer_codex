#!/usr/bin/env python3
"""Forward-semantics check only; not a PyTorch training implementation.

Run: python3 ternary_semantics_check.py
This demonstrates why sign(h)*amplitude has no silence interval.
"""
from __future__ import annotations
import json
import math
from typing import Iterable


def checked(values: Iterable[float]) -> list[float]:
    result = [float(x) for x in values]
    if not all(math.isfinite(x) for x in result):
        raise ValueError("Membrane values must be finite.")
    return result


def sign_amplitude(values: Iterable[float], amplitude: float = 1.0) -> list[float]:
    if not math.isfinite(amplitude) or amplitude <= 0:
        raise ValueError("amplitude must be finite and positive")
    return [amplitude * ((x > 0) - (x < 0)) for x in checked(values)]


def ternary_deadzone(values: Iterable[float], positive_threshold: float = 1.0,
                     negative_threshold: float = 1.0,
                     amplitude: float = 1.0) -> list[float]:
    for name, value in (("positive_threshold", positive_threshold),
                        ("negative_threshold", negative_threshold),
                        ("amplitude", amplitude)):
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    return [amplitude * ((x >= positive_threshold) - (x <= -negative_threshold))
            for x in checked(values)]


def main() -> None:
    x = [-2.0, -0.2, 0.0, 0.2, 2.0]
    old = sign_amplitude(x)
    new = ternary_deadzone(x)
    assert old == [-1.0, -1.0, 0.0, 1.0, 1.0]
    assert new == [-1.0, 0.0, 0.0, 0.0, 1.0]
    assert ternary_deadzone([-1.0, 1.0]) == [-1.0, 1.0]
    assert ternary_deadzone([-2.0, -1.0, 0.5, 1.0],
                            positive_threshold=1.0,
                            negative_threshold=2.0) == [-1.0, 0.0, 0.0, 1.0]
    print(json.dumps({"input": x, "sign_amplitude": old, "deadzone_ternary": new,
                      "status": "PASS", "scope": "forward semantics only; no training or server run"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
