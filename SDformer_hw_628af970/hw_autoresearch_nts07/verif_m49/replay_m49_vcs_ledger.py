#!/usr/bin/env python3
"""Independently replay an M49 accepted-handshake ledger.

The replay reconstructs the lowest-row-per-bank K2 union schedule without
importing RTL/TB code.  It proves unique weight-row issue, per-destination
signed updates, atomic launch output order, and exact signed-19x96 results.
"""

from __future__ import print_function

import argparse
import hashlib
import json
from pathlib import Path


BANKS = 8
LANES = 96
ACC_W = 19
ACC_MASK = (1 << ACC_W) - 1


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signed(value, bits):
    value &= (1 << bits) - 1
    return value - (1 << bits) if value & (1 << (bits - 1)) else value


def weight(source, lane):
    if source < BANKS:
        return -128
    return ((source * 37 + lane * 13 + 19) % 255) - 127


def bits(value):
    return set(index for index in range(256) if (value >> index) & 1)


def unpack_lanes(packed):
    return [signed(packed >> (lane * ACC_W), ACC_W) for lane in range(LANES)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replay(path):
    contexts = {}
    expected_by_tag = {}
    expected_order = []
    output_index = 0
    output_tags = set()
    active = None
    groups = []
    request_count = 0
    physical_reads = 0
    logical_updates = 0
    shared_updates = 0
    last_count = 0
    command_count = 0
    end_seen = False

    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        fields = raw.split()
        if not fields:
            continue
        kind = fields[0]
        if kind == "C":
            require(len(fields) == 7, "line {} malformed C".format(lineno))
            _, _cycle, ctx_s, tag_s, add_s, sub_s, seed_s = fields
            ctx = int(ctx_s)
            tag = int(tag_s, 16)
            add = bits(int(add_s, 16))
            sub = bits(int(sub_s, 16))
            require(not (add & sub), "line {} overlapping mask".format(lineno))
            seed = unpack_lanes(int(seed_s, 16))
            contexts[ctx] = {
                "tag": tag,
                "add": set(add),
                "sub": set(sub),
                "acc": list(seed),
                "count": 0,
                "launched": False,
            }
            expected_by_tag[tag] = contexts[ctx]
            command_count += 1
        elif kind == "L":
            require(len(fields) == 5, "line {} malformed L".format(lineno))
            _, _cycle, c0_s, use1_s, c1_s = fields
            c0, use1, c1 = int(c0_s), int(use1_s), int(c1_s)
            require(active is None, "line {} overlapping active group".format(lineno))
            require(c0 in contexts and not contexts[c0]["launched"],
                    "line {} illegal c0".format(lineno))
            require(not use1 or (c1 in contexts and c1 != c0
                                  and not contexts[c1]["launched"]),
                    "line {} illegal c1".format(lineno))
            contexts[c0]["launched"] = True
            if use1:
                contexts[c1]["launched"] = True
            expected_order.append(contexts[c0]["tag"])
            if use1:
                expected_order.append(contexts[c1]["tag"])
            union = contexts[c0]["add"] | contexts[c0]["sub"]
            if use1:
                union |= contexts[c1]["add"] | contexts[c1]["sub"]
            relation = "K1"
            if use1:
                s0 = contexts[c0]["add"] | contexts[c0]["sub"]
                s1 = contexts[c1]["add"] | contexts[c1]["sub"]
                if s0 == s1:
                    relation = "K2_FULL_SHARE"
                elif s0 & s1:
                    relation = "K2_PARTIAL_SHARE"
                else:
                    relation = "K2_NO_SHARE"
            group = {"c0": c0, "use1": bool(use1), "c1": c1,
                     "relation": relation, "requests": 0,
                     "physical_reads": 0, "logical_updates": 0}
            groups.append(group)
            if union:
                active = group
        elif kind == "R":
            require(len(fields) == 12, "line {} malformed R".format(lineno))
            (_, _cycle, c0_s, use1_s, c1_s, bank_s, addr_s,
             v0_s, sub0_s, v1_s, sub1_s, last_s) = fields
            require(active is not None, "line {} request outside group".format(lineno))
            c0, use1, c1 = int(c0_s), int(use1_s), int(c1_s)
            require((c0, bool(use1), c1) ==
                    (active["c0"], active["use1"], active["c1"]),
                    "line {} request pair mismatch".format(lineno))
            bank_valid = int(bank_s, 16)
            packed_addr = int(addr_s, 16)
            got_v0, got_sub0 = int(v0_s, 16), int(sub0_s, 16)
            got_v1, got_sub1 = int(v1_s, 16), int(sub1_s, 16)
            exp_bank = exp_v0 = exp_sub0 = exp_v1 = exp_sub1 = 0
            selected = []
            for bank in range(BANKS):
                sources0 = contexts[c0]["add"] | contexts[c0]["sub"]
                sources1 = ((contexts[c1]["add"] | contexts[c1]["sub"])
                            if use1 else set())
                candidates = sorted(source for source in sources0 | sources1
                                    if source % BANKS == bank)
                if not candidates:
                    continue
                source = candidates[0]
                row = source // BANKS
                exp_bank |= 1 << bank
                require(((packed_addr >> (bank * 5)) & 0x1f) == row,
                        "line {} bank {} is not lowest union row".format(
                            lineno, bank))
                in0 = source in sources0
                in1 = source in sources1
                if in0:
                    exp_v0 |= 1 << bank
                    if source in contexts[c0]["sub"]:
                        exp_sub0 |= 1 << bank
                if in1:
                    exp_v1 |= 1 << bank
                    if source in contexts[c1]["sub"]:
                        exp_sub1 |= 1 << bank
                selected.append((bank, source, in0, in1))
            require(bank_valid == exp_bank and got_v0 == exp_v0
                    and got_sub0 == exp_sub0 and got_v1 == exp_v1
                    and got_sub1 == exp_sub1,
                    "line {} bank/destination polarity mismatch".format(lineno))
            require(bank_valid != 0, "line {} empty request".format(lineno))
            for _bank, source, in0, in1 in selected:
                physical_reads += 1
                active["physical_reads"] += 1
                if in0:
                    sign = -1 if source in contexts[c0]["sub"] else 1
                    for lane in range(LANES):
                        contexts[c0]["acc"][lane] += sign * weight(source, lane)
                    contexts[c0]["add"].discard(source)
                    contexts[c0]["sub"].discard(source)
                    contexts[c0]["count"] += 1
                    logical_updates += 1
                    active["logical_updates"] += 1
                if in1:
                    sign = -1 if source in contexts[c1]["sub"] else 1
                    for lane in range(LANES):
                        contexts[c1]["acc"][lane] += sign * weight(source, lane)
                    contexts[c1]["add"].discard(source)
                    contexts[c1]["sub"].discard(source)
                    contexts[c1]["count"] += 1
                    logical_updates += 1
                    active["logical_updates"] += 1
                if in0 and in1:
                    shared_updates += 1
            request_count += 1
            active["requests"] += 1
            union_left = contexts[c0]["add"] | contexts[c0]["sub"]
            if use1:
                union_left |= contexts[c1]["add"] | contexts[c1]["sub"]
            exp_last = not union_left
            require(int(last_s) == int(exp_last),
                    "line {} last mismatch".format(lineno))
            if exp_last:
                last_count += 1
                active = None
        elif kind == "O":
            require(len(fields) == 5, "line {} malformed O".format(lineno))
            _, _cycle, tag_s, count_s, acc_s = fields
            tag, count = int(tag_s, 16), int(count_s)
            require(output_index < len(expected_order),
                    "line {} unexpected output".format(lineno))
            require(tag == expected_order[output_index],
                    "line {} output order mismatch".format(lineno))
            require(tag not in output_tags, "line {} duplicate output".format(lineno))
            require(tag in expected_by_tag,
                    "line {} missing tag model".format(lineno))
            model = expected_by_tag[tag]
            got_acc = unpack_lanes(int(acc_s, 16))
            require(count == model["count"],
                    "line {} source count mismatch".format(lineno))
            require(got_acc == model["acc"],
                    "line {} accumulator mismatch".format(lineno))
            output_tags.add(tag)
            output_index += 1
        elif kind == "END":
            require(len(fields) == 3, "line {} malformed END".format(lineno))
            legal_tags = int(fields[1].split("=", 1)[1])
            outputs = int(fields[2].split("=", 1)[1])
            require(legal_tags == command_count and outputs == output_index,
                    "END count mismatch")
            end_seen = True
        else:
            raise ValueError("line {} unknown record {}".format(lineno, kind))

    require(end_seen, "ledger has no END")
    require(active is None, "active group remains")
    require(output_index == len(expected_order) == command_count,
            "command/launch/output conservation mismatch")
    relations = dict((name, sum(group["relation"] == name for group in groups))
                     for name in ("K1", "K2_FULL_SHARE",
                                  "K2_PARTIAL_SHARE", "K2_NO_SHARE"))
    for name, count in relations.items():
        require(count > 0, "missing {} coverage".format(name))
    require(physical_reads < logical_updates,
            "K2 union did not save a physical weight read")
    return {
        "schema": "m49_vcs_handshake_ledger_replay_v1",
        "status": "PASS_STANDALONE_M49_K2_UNION_EXACT_LEDGER",
        "ledger": {"path": str(Path(path).resolve()),
                   "sha256": sha256(path)},
        "commands": command_count,
        "groups": len(groups),
        "group_relations": relations,
        "requests": request_count,
        "last_requests": last_count,
        "physical_unique_weight_row_issues": physical_reads,
        "logical_destination_updates": logical_updates,
        "shared_dual_destination_updates": shared_updates,
        "physical_read_reduction_vs_independent_updates": {
            "numerator": logical_updates - physical_reads,
            "denominator": logical_updates,
        },
        "outputs": output_index,
        "mismatch_count": 0,
        "claim_scope": [
            "standalone accepted-handshake exact K1/K2 union-source behavior",
            "lowest remaining union bank-row and unique weight-row issue",
            "signed19x96 output tag/order/value conservation",
        ],
        "not_admitted": [
            "M45 all10 cycles", "PPA/power/energy", "system speedup",
            "checkpoint or full-network equivalence",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.ledger)
    require(not args.output.exists(), "refusing to overwrite replay output")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS M49 LEDGER commands={} groups={} requests={} outputs={} reads={} updates={}".format(
        result["commands"], result["groups"], result["requests"],
        result["outputs"], result["physical_unique_weight_row_issues"],
        result["logical_destination_updates"]))


if __name__ == "__main__":
    main()
