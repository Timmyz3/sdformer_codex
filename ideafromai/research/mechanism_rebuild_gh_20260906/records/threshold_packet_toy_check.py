"""Small exact arithmetic checks for a research proposal, not an RTL test.

No checkpoint, dataset, EDA flow, or production file is read or modified.
All performance/accuracy/PPA admissions remain zero.
"""
from fractions import Fraction as F
from itertools import product
from math import ceil, log2
from pathlib import Path
import json


def encode(values, predicted_threshold, capacity):
    bits = [int(v >= predicted_threshold) for v in values]
    kept = []
    lower = upper = None

    def discard(index, value):
        nonlocal lower, upper
        if bits[index] == 0:
            lower = value if lower is None else max(lower, value)
        else:
            upper = value if upper is None else min(upper, value)

    for index, value in enumerate(values):
        key = (abs(value - predicted_threshold), index)
        kept.append((key, index, value))
        kept.sort(key=lambda item: item[0])
        if len(kept) > capacity:
            _, old_index, old_value = kept.pop()
            discard(old_index, old_value)
    return bits, lower, upper, kept


def decode(packet, true_threshold):
    bits, lower, upper, kept = packet
    if lower is not None and not lower < true_threshold:
        return None
    if upper is not None and not true_threshold <= upper:
        return None
    final = bits[:]
    for _, index, value in kept:
        final[index] = int(value >= true_threshold)
    return final


def check_packets():
    encodes = decisions = certified = corrected = 0
    thresholds = [F(x, 2) for x in range(-5, 6)]
    for values in product(range(-2, 3), repeat=4):
        for predicted in thresholds:
            previous_accepts = set()
            for capacity in range(5):
                packet = encode(values, predicted, capacity)
                encodes += 1
                keep_indices = {item[1] for item in packet[3]}
                discarded_off = [v for i, v in enumerate(values) if i not in keep_indices and v < predicted]
                discarded_on = [v for i, v in enumerate(values) if i not in keep_indices and v >= predicted]
                assert packet[1] == (max(discarded_off) if discarded_off else None)
                assert packet[2] == (min(discarded_on) if discarded_on else None)
                accepted_here = set()
                for actual in thresholds:
                    decisions += 1
                    result = decode(packet, actual)
                    if result is not None:
                        certified += 1
                        accepted_here.add(actual)
                        assert result == [int(v >= actual) for v in values]
                        corrected += (result != packet[0])
                    elif capacity == len(values):
                        raise AssertionError("Full retained payload must decode every threshold")
                assert previous_accepts <= accepted_here
                previous_accepts = accepted_here
    return dict(encodes=encodes, threshold_decisions=decisions,
                certified_decisions=certified, certified_with_bit_patches=corrected)


def check_bn_psn_algebra():
    # d denotes an already supplied positive sqrt(var+eps), not a new BN reducer.
    matrix = [[F(1), F(-2)], [F(3), F(1)]]
    bias = [F(1, 3), F(-2, 3)]
    theta, center = F(4, 3), F(1, 5)
    checks = 0
    for y in product(range(-2, 3), repeat=2):
        for gamma, beta, mu, d in product([F(-3), F(0), F(2)], [F(-1), F(2)], [F(-2), F(1)], [F(1, 2), F(2)]):
            for t, row in enumerate(matrix):
                u = sum(a * v for a, v in zip(row, y))
                r = sum(row)
                original = sum(a * (gamma * (v - mu) / d + beta) for a, v in zip(row, y)) + bias[t] - center
                expected = int(original >= theta)
                if gamma == 0:
                    actual = int(beta * r + bias[t] - center >= theta)
                else:
                    tau = mu * r + d / gamma * (theta + center - bias[t] - beta * r)
                    direction = 1 if gamma > 0 else -1
                    actual = int(direction * u >= direction * tau)
                assert actual == expected
                checks += 1
    return checks


def state_examples():
    # Shape is from archived ep34 capture metadata, not the handoff topology sketch.
    n, channels, hidden, timesteps = 192000, 96, 384, 10
    width, block = 32, 64
    packets = ceil((n // timesteps) / block) * timesteps * hidden
    rows = []
    for capacity in [0, 2, 4, 8]:
        bits = block + 2 * width + capacity * (width + ceil(log2(block)))
        rows.append(dict(B=block, K=capacity, payload_bits_per_packet=bits,
                         payload_bytes_per_layer=packets * bits // 8))
    return dict(N_includes_T=n, C=channels, H=hidden, T=timesteps,
                assumed_U_width=width, full_U_bytes=n * hidden * width // 8,
                binary_source_bytes=n * channels // 8, rows=rows,
                excluded="Sentinel flags, count, epoch/layer/weight identity, queues, retained source, scratch Y/U, SRAM padding/ports, traffic and replay; these are payload examples, not PPA or savings measurements.")


def main():
    result = {"status": "PASS_SMALL_DOMAIN_ALGEBRA_ONLY",
              "packet": check_packets(),
              "bn_psn_rational_checks": check_bn_psn_algebra(),
              "state_examples": state_examples(),
              "example": {},
              "PPA_ADMISSION": 0, "RTL_SPEEDUP_ADMISSION": 0,
              "FROZEN_FP_EQUIVALENCE": 0,
              "not_proven": ["DSEC opportunity", "actual replay rate", "finite-width hardware", "original floating point equality", "circuit energy or latency", "paper acceptance"]}
    values = [F(-9), F(-2), F(4), F(8)]
    packet = encode(values, F(1), 1)
    actual = F(-3)
    decoded = decode(packet, actual)
    assert decoded == [0, 1, 1, 1]
    result["example"] = dict(U=[-9, -2, 4, 8], predicted_tau=1, true_tau=-3,
                             K=1, retained_position=packet[3][0][1],
                             lower=int(packet[1]), upper=int(packet[2]),
                             predicted_bits=packet[0], certified_patched_bits=decoded)
    output = Path(__file__).with_name("threshold_packet_toy_check.json")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
