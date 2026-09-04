#!/usr/bin/env python3
"""Independent accepted-handshake replay for the M54 K1..K4 ledger."""

from __future__ import print_function

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


BANKS = 8
LANES = 96
ACC_W = 19


def require(condition, message):
    if not condition:
        raise ValueError(message)


def signed(value, width):
    value &= (1 << width) - 1
    return value - (1 << width) if value & (1 << (width - 1)) else value


def model_weight(source, lane):
    if source < BANKS:
        return -128
    return ((source * 37 + lane * 13 + 19) % 255) - 127


def bit_set(value):
    return set(index for index in range(256) if (value >> index) & 1)


def unpack_lanes(value):
    return [signed(value >> (lane * ACC_W), ACC_W)
            for lane in range(LANES)]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def classify_group(group_contexts):
    count = len(group_contexts)
    if count == 1:
        return "K1"
    source_sets = [row["add"] | row["sub"] for row in group_contexts]
    if all(value == source_sets[0] for value in source_sets[1:]):
        return "K{}_FULL_SHARE".format(count)
    if all(not (source_sets[left] & source_sets[right])
               for left in range(count)
               for right in range(left + 1, count)):
        return "K{}_NO_SHARE".format(count)
    return "K{}_PARTIAL_SHARE".format(count)


def replay(path):
    contexts = {}
    expected_by_tag = {}
    expected_order = []
    output_index = 0
    output_tags = set()
    active = None
    groups = []
    request_count = 0
    request_tag_expected = 0
    last_requests = 0
    physical_reads = 0
    logical_updates = 0
    shared_reads = 0
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
            ctx, tag = int(ctx_s), int(tag_s, 16)
            add, sub = bit_set(int(add_s, 16)), bit_set(int(sub_s, 16))
            require(not (add & sub), "line {} overlapping masks".format(lineno))
            require(ctx not in contexts,
                    "line {} context reused before release".format(lineno))
            row = {
                "tag": tag, "add": set(add), "sub": set(sub),
                "acc": unpack_lanes(int(seed_s, 16)), "count": 0,
                "launched": False,
            }
            contexts[ctx] = row
            expected_by_tag[tag] = row
            command_count += 1
        elif kind == "L":
            require(len(fields) == 4, "line {} malformed L".format(lineno))
            _, _cycle, count_s, packed_s = fields
            count, packed = int(count_s), int(packed_s, 16)
            selected = [(packed >> (slot * 4)) & 0xf
                        for slot in range(count)]
            require(1 <= count <= 4 and len(set(selected)) == count,
                    "line {} illegal group count/distinctness".format(lineno))
            require(active is None, "line {} overlapping group".format(lineno))
            rows = []
            for ctx in selected:
                require(ctx in contexts and not contexts[ctx]["launched"],
                        "line {} unowned/relaunched context".format(lineno))
                contexts[ctx]["launched"] = True
                rows.append(contexts[ctx])
                expected_order.append(contexts[ctx]["tag"])
            group = {
                "contexts": selected,
                "relation": classify_group(rows),
                "requests": 0,
                "physical_reads": 0,
                "logical_updates": 0,
            }
            groups.append(group)
            union = set()
            for row in rows:
                union |= row["add"] | row["sub"]
            if union:
                active = group
            else:
                for ctx in selected:
                    del contexts[ctx]
        elif kind == "R":
            require(len(fields) == 10, "line {} malformed R".format(lineno))
            (_, _cycle, request_tag_s, count_s, contexts_s, bank_s,
             addr_s, valid_s, subtract_s, last_s) = fields
            require(active is not None,
                    "line {} request outside active group".format(lineno))
            request_tag = int(request_tag_s, 16)
            count, packed = int(count_s), int(contexts_s, 16)
            selected = [(packed >> (slot * 4)) & 0xf
                        for slot in range(count)]
            require(request_tag == request_tag_expected,
                    "line {} response-tag sequence mismatch".format(lineno))
            request_tag_expected = (request_tag_expected + 1) & 0xffff
            require(count == len(active["contexts"]) and
                    selected == active["contexts"],
                    "line {} active context identity mismatch".format(lineno))
            bank_valid = int(bank_s, 16)
            packed_addr = int(addr_s, 16)
            got_valid = int(valid_s, 16)
            got_subtract = int(subtract_s, 16)
            exp_bank = exp_valid = exp_subtract = 0
            chosen = []
            for bank in range(BANKS):
                candidates = sorted(source for ctx in selected
                                    for source in (contexts[ctx]["add"] |
                                                   contexts[ctx]["sub"])
                                    if source % BANKS == bank)
                if not candidates:
                    continue
                source = candidates[0]
                exp_bank |= 1 << bank
                require(((packed_addr >> (bank * 5)) & 0x1f) ==
                        source // BANKS,
                        "line {} bank{} not lowest union row".format(
                            lineno, bank))
                users = []
                for slot, ctx in enumerate(selected):
                    row = contexts[ctx]
                    if source in row["add"] or source in row["sub"]:
                        exp_valid |= 1 << (slot * BANKS + bank)
                        subtract = source in row["sub"]
                        if subtract:
                            exp_subtract |= 1 << (slot * BANKS + bank)
                        users.append((ctx, subtract))
                chosen.append((bank, source, users))
            require(bank_valid == exp_bank and got_valid == exp_valid and
                    got_subtract == exp_subtract and bank_valid != 0,
                    "line {} bank/destination/polarity mismatch".format(lineno))
            for _bank, source, users in chosen:
                physical_reads += 1
                active["physical_reads"] += 1
                if len(users) > 1:
                    shared_reads += 1
                for ctx, subtract in users:
                    row = contexts[ctx]
                    sign = -1 if subtract else 1
                    for lane in range(LANES):
                        row["acc"][lane] += sign * model_weight(source, lane)
                        require(-(1 << (ACC_W-1)) <= row["acc"][lane]
                                < (1 << (ACC_W-1)),
                                "line {} signed19 overflow".format(lineno))
                    row["add"].discard(source)
                    row["sub"].discard(source)
                    row["count"] += 1
                    logical_updates += 1
                    active["logical_updates"] += 1
            request_count += 1
            active["requests"] += 1
            union_left = set()
            for ctx in selected:
                union_left |= contexts[ctx]["add"] | contexts[ctx]["sub"]
            expected_last = not union_left
            require(int(last_s) == int(expected_last),
                    "line {} last mismatch".format(lineno))
            if expected_last:
                last_requests += 1
                for ctx in selected:
                    del contexts[ctx]
                active = None
        elif kind == "O":
            require(len(fields) == 5, "line {} malformed O".format(lineno))
            _, _cycle, tag_s, count_s, acc_s = fields
            tag, count = int(tag_s, 16), int(count_s)
            require(output_index < len(expected_order),
                    "line {} unexpected output".format(lineno))
            require(tag == expected_order[output_index] and
                    tag not in output_tags and tag in expected_by_tag,
                    "line {} output tag/order/uniqueness".format(lineno))
            model = expected_by_tag[tag]
            require(count == model["count"] and
                    unpack_lanes(int(acc_s, 16)) == model["acc"],
                    "line {} output count/value mismatch".format(lineno))
            output_tags.add(tag)
            output_index += 1
        elif kind == "END":
            require(len(fields) == 3, "line {} malformed END".format(lineno))
            commands = int(fields[1].split("=", 1)[1])
            outputs = int(fields[2].split("=", 1)[1])
            require(commands == command_count and outputs == output_index,
                    "END population mismatch")
            end_seen = True
        else:
            raise ValueError("line {} unknown record {}".format(lineno, kind))

    require(end_seen and active is None and not contexts,
            "ledger incomplete/active contexts remain")
    require(output_index == len(expected_order) == command_count,
            "command/launch/output conservation mismatch")
    relations = Counter(group["relation"] for group in groups)
    for name in ("K1", "K2_FULL_SHARE", "K2_PARTIAL_SHARE",
                 "K2_NO_SHARE", "K3_FULL_SHARE", "K3_PARTIAL_SHARE",
                 "K3_NO_SHARE", "K4_FULL_SHARE", "K4_PARTIAL_SHARE",
                 "K4_NO_SHARE"):
        require(relations[name] > 0, "missing {} group".format(name))
    require(physical_reads < logical_updates and shared_reads > 0,
            "union sharing not exercised")
    return {
        "schema": "m54_vcs_handshake_ledger_replay_v1",
        "status": "PASS_STANDALONE_M54_K4_C16_EXACT_LEDGER",
        "ledger": {"path": str(Path(path).resolve()), "sha256": sha256(path)},
        "commands": command_count,
        "groups": len(groups),
        "group_relations": dict(sorted(relations.items())),
        "requests": request_count,
        "last_requests": last_requests,
        "physical_unique_weight_row_issues": physical_reads,
        "logical_destination_updates": logical_updates,
        "shared_physical_reads": shared_reads,
        "outputs": output_index,
        "response_tags_monotonic_modulo_16b": True,
        "context_ids_finite_4b_and_reuse_checked": True,
        "mismatch_count": 0,
        "claim_scope": [
            "standalone accepted-handshake K1..K4 union behavior",
            "lowest union bank-row and independent add/sub/bypass",
            "signed19x96 output tag/order/value conservation",
        ],
        "not_admitted": [
            "M52 transaction cycles as RTL cycles", "DC/PPA/power/energy",
            "system speedup", "checkpoint/full-network equivalence",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.ledger)
    require(not args.output.exists(), "refusing replay output overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("PASS M54 LEDGER commands={} groups={} requests={} outputs={} reads={} updates={}".format(
        result["commands"], result["groups"], result["requests"],
        result["outputs"], result["physical_unique_weight_row_issues"],
        result["logical_destination_updates"]))


if __name__ == "__main__":
    main()
