#!/usr/bin/python3.12
"""One fixed two-stage source interface; actual RF payload, no RTL claim."""
from __future__ import annotations
from collections import Counter, deque
from fractions import Fraction
from pathlib import Path
import copy
import csv
import json
import random
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import schedule_source as base

CONTRACT = json.loads((HERE / "resource_contract.json").read_text())
LATENCY = CONTRACT["pipeline"]["result_read_latency_slots"]
WORDS = CONTRACT["work_RF"]["words_per_lane"]


def graph_variant(axis, fused):
    nodes, params = base.graphs(axis)
    if not fused:
        return nodes, params, 0
    users = Counter(p for n in nodes for p in {r["node"] for r in n["parents"]})
    removed = set()
    replacements = {}
    for node in nodes:
        if node["kind"] != "sat" or len(node["parents"]) != 1:
            continue
        p = node["parents"][0]
        rounded = nodes[p["node"]]
        if (p["shift"] == 0 and p["sign"] == 1
                and rounded["kind"] == "round" and users[rounded["id"]] == 1):
            replacements[node["id"]] = dict(node, kind="norm24",
                parents=copy.deepcopy(rounded["parents"]), rne_shift=rounded["rne_shift"])
            removed.add(rounded["id"])
    kept = [replacements.get(n["id"], n) for n in nodes if n["id"] not in removed]
    remap = {n["id"]: i for i, n in enumerate(kept)}
    result = []
    for n in kept:
        x = copy.deepcopy(n)
        x["id"] = remap[n["id"]]
        for p in x["parents"]:
            p["node"] = remap[p["node"]]
        result.append(x)
    return result, params, len(removed)


def compile_program(nodes, policy):
    parents = {n["id"]: {p["node"] for p in n["parents"]} for n in nodes}
    use = Counter(p for n in nodes[10:] for p in parents[n["id"]])
    consumers = {n["id"]: [] for n in nodes}
    for n in nodes[10:]:
        for p in parents[n["id"]]:
            consumers[p].append(n["id"])
    height = {}
    for n in reversed(nodes):
        height[n["id"]] = LATENCY + max((height[c] for c in consumers[n["id"]]), default=0)
    regs = {i: i for i in range(10)}
    free = set(range(10, WORDS))
    ready_at = {i: i + LATENCY for i in range(10)}
    pending = set(range(10, len(nodes)))
    program = [dict(kind="load", dst=i, source_t=i, logical_node=i, issue_slot=i) for i in range(10)]
    peak = 10
    c = 10
    while pending:
        ready = [i for i in pending if all(p in ready_at and ready_at[p] <= c for p in parents[i])]
        if not ready:
            program.append(dict(kind="nop", reason="RAW_wait", issue_slot=c))
            c += 1
            continue
        def score(i):
            return (nodes[i]["kind"] == "gate", sum(use[p] == 1 for p in parents[i]), height[i], -i)
        i = min(ready) if policy == "official_ready" else max(ready, key=score)
        n = nodes[i]
        operands = [dict(reg=regs[p["node"]], node=p["node"], shift=p["shift"], sign=p["sign"])
                    for p in n["parents"]]
        for p in parents[i]:
            use[p] -= 1
            if use[p] == 0:
                free.add(regs.pop(p))
        ins = {k: v for k, v in n.items() if k not in {"id", "parents"}}
        ins.update(operands=operands, logical_node=i, issue_slot=c)
        if n["kind"] != "gate":
            if not free:
                raise RuntimeError(f"RF allocation failed: {policy}, node {i}")
            dst = min(free)
            free.remove(dst)
            regs[i] = dst
            ins["dst"] = dst
        program.append(ins)
        ready_at[i] = c + LATENCY
        pending.remove(i)
        peak = max(peak, len(regs))
        c += 1
    final_ready = max(ready_at.values())
    while c < final_ready:
        program.append(dict(kind="nop", reason="pipeline_drain", issue_slot=c))
        c += 1
    program.append(dict(kind="commit", issue_slot=c))
    assert len(program) <= CONTRACT["instruction_ROM"]["words"]
    return program, dict(policy=policy, instructions=len(program), peak_live_words=peak,
        operation_slots=dict(Counter(n["kind"] for n in program)),
        RAW_nops=sum(n.get("reason") == "RAW_wait" for n in program),
        drain_nops=sum(n.get("reason") == "pipeline_drain" for n in program))


def stage_A(ins, work, tags, raw_vector):
    kind = ins["kind"]
    if kind == "load":
        return int(raw_vector[ins["source_t"]])
    for p in ins["operands"]:
        assert tags[p["reg"]] == p["node"], ("RF RAW/lifetime", ins, p, tags[p["reg"]])
    args = [(work[p["reg"]] << p["shift"]) * p["sign"] for p in ins["operands"]]
    assert all(-(1 << 47) <= x < (1 << 47) for x in args)
    if kind == "addsub":
        value = sum(args)
    elif kind in {"round", "norm24"}:
        value = base.rne(args[0], ins["rne_shift"])
    elif kind == "sat":
        value = args[0]
    elif kind == "gate":
        value = base.gate(args[0], ins["threshold"], ins["direction"], ins["constant"])
    else:
        raise ValueError(kind)
    assert -(1 << 47) <= value < (1 << 47)
    return value


def stage_B(ins, value):
    return base.sat(value) if ins["kind"] in {"sat", "norm24"} else value


def run_vector(program, vector):
    work = [None] * WORDS
    tags = [None] * WORDS
    out = [None] * 10
    pipe = None
    for ins in program:
        kind = ins["kind"]
        incoming = None
        if kind == "commit":
            assert pipe is None and None not in out
        elif kind != "nop":
            incoming = (ins, stage_A(ins, work, tags, vector))
        # This write is at slot end, after this slot's operand reads.
        if pipe is not None:
            old, value = pipe
            result = stage_B(old, value)
            if old["kind"] == "gate":
                out[old["output_t"]] = result
            else:
                work[old["dst"]] = result
                tags[old["dst"]] = old["logical_node"]
        pipe = incoming
    assert pipe is None and None not in out
    return out


def validate_program(axis, program, params):
    vectors = [[0] * 10, [base.MIN24] * 10, [base.MAX24] * 10, [base.MIN24, base.MAX24] * 5]
    for t in range(10):
        for v in [base.MIN24, base.MAX24, -2049, -2048, -1, 1, 2048, 2049]:
            x = [0] * 10
            x[t] = v
            vectors.append(x)
    rand = random.Random(10)
    vectors += [[rand.randint(base.MIN24, base.MAX24) for _ in range(10)] for _ in range(512)]
    for x in vectors:
        assert run_vector(program, x) == base.reference(axis, params, x), (axis, x)
    return dict(vectors=len(vectors), mismatches=0, register_tag_and_two_slot_readiness_checked=True)


def check_norm_boundary():
    checks = 0
    for q in [base.MIN24 - 1, base.MIN24, base.MIN24 + 1, -2, -1, 0, 1, 2,
              base.MAX24 - 1, base.MAX24, base.MAX24 + 1]:
        for rem in [-1, 0, 2047, 2048, 2049, 4095, 4096]:
            value = q * 4096 + rem
            expected = min(base.MAX24, max(base.MIN24, round(Fraction(value, 4096))))
            assert base.sat(base.rne(value, 12)) == expected
            checks += 1
    return dict(signed_tie_and_saturation_cases=checks, mismatches=0)


def simulate(program, scenario, path, input_vectors, expected_words):
    lanes = CONTRACT["lanes"]
    vectors = len(input_vectors)
    batches = vectors // lanes
    conf = CONTRACT["backpressure"][scenario]
    raw = [[None] * 10 for _ in range(vectors)]
    work = [[None] * WORDS for _ in range(lanes)]
    tags = [[None] * WORDS for _ in range(lanes)]
    fifo = deque()
    pipe = None  # Exactly one registered interstage payload, never a host event queue.
    gate_words = [0] * lanes
    valid_mask = 0
    admitted = set()
    commit_fetched = False
    pc = batch = cycle = 0
    counts = Counter()
    wb_counts = Counter()
    raw_reads = raw_writes = rf_reads = rf_writes = rom_reads = 0
    peak_fifo = 0
    popped = []
    log = []
    while batch < batches or fifo or pipe is not None:
        accepted = enqueued = accepted_word = enqueued_word = ""
        ready = cycle % conf["period"] >= conf["blocked_prefix"]
        if ready and fifo:
            accepted, accepted_word = fifo.popleft()
            assert accepted_word == expected_words[accepted]
            popped.append(accepted)
            counts["output_transfers"] += 1
        incoming = None
        issue = "drain"
        logical_node = ""
        if batch < batches:
            ins = program[pc]
            kind = ins["kind"]
            if kind == "load" and cycle <= batch * 10 + ins["source_t"]:
                issue = "input_wait"
            elif kind == "commit":
                assert pipe is None and valid_mask == 1023
                if not commit_fetched:
                    rom_reads += 1
                    commit_fetched = True
                if len(fifo) < CONTRACT["output"]["FIFO_entries"]:
                    lane = min(set(range(lanes)) - admitted)
                    enqueued = batch * lanes + lane
                    enqueued_word = gate_words[lane]
                    fifo.append((enqueued, enqueued_word))
                    admitted.add(lane)
                    issue = "commit_enqueue"
                else:
                    issue = "fifo_full_wait"
                if len(admitted) == lanes:
                    batch += 1
                    pc = 0
                    admitted.clear()
                    commit_fetched = False
                    valid_mask = 0
                    gate_words = [0] * lanes
            else:
                rom_reads += 1
                if kind == "nop":
                    issue = ins["reason"]
                else:
                    issue = kind
                    logical_node = ins["logical_node"]
                    if kind == "load":
                        raw_reads += lanes
                    else:
                        rf_reads += lanes * len({p["reg"] for p in ins["operands"]})
                    values = [stage_A(ins, work[lane], tags[lane], raw[batch * lanes + lane])
                              for lane in range(lanes)]
                    incoming = (ins, values)
                pc += 1
        wb_kind = wb_dst = ""
        if pipe is not None:
            old, values = pipe
            wb_kind = old["kind"]
            wb_counts[wb_kind] += 1
            if wb_kind == "gate":
                assert not (valid_mask & (1 << old["output_t"]))
                for lane, value in enumerate(values):
                    gate_words[lane] |= stage_B(old, value) << old["output_t"]
                valid_mask |= 1 << old["output_t"]
            else:
                wb_dst = old["dst"]
                for lane, value in enumerate(values):
                    work[lane][wb_dst] = stage_B(old, value)
                    tags[lane][wb_dst] = old["logical_node"]
                rf_writes += lanes
        pipe = incoming
        input_words = lanes if cycle < batches * 10 else 0
        if input_words:
            input_batch, input_t = divmod(cycle, 10)
            for lane in range(lanes):
                v = input_batch * lanes + lane
                raw[v][input_t] = input_vectors[v][input_t]
            raw_writes += lanes
        counts[issue] += 1
        peak_fifo = max(peak_fifo, len(fifo))
        log.append([cycle, batch, pc, issue, logical_node, wb_kind, wb_dst, input_words,
                    enqueued, enqueued_word, accepted, accepted_word, len(fifo), valid_mask, int(pipe is not None)])
        cycle += 1
    assert popped == list(range(vectors))
    assert raw == input_vectors
    assert raw_reads == raw_writes == vectors * 10
    assert counts["gate"] == wb_counts["gate"] == batches * 10
    with path.open("w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["cycle", "batch_after", "pc_after", "issue", "logical_node", "WB_kind", "WB_dst",
                         "I24_input_words", "enqueued_vector", "enqueued_word", "accepted_vector", "accepted_word",
                         "FIFO_count", "gate_valid_mask", "interstage_valid"])
        writer.writerows(log)
    return dict(service_slots=cycle, event_slots=dict(counts), writeback_slots=dict(wb_counts),
                peak_FIFO_entries=peak_fifo, interstage_capacity_entries=1,
                raw_input_bytes=raw_writes * 3, raw_I_read_bytes=raw_reads * 3,
                work_RF_read_bytes=rf_reads * 6, work_RF_write_bytes=rf_writes * 6,
                instruction_fetch_bytes=rom_reads * 16,
                gates_accepted=len(popped), payload_mismatches=0, raw_I_unchanged=True)


def main():
    measurements = {}
    for control in CONTRACT["controls"]:
        for axis in CONTRACT["axes"]:
            key = f"{axis}_{control}"
            nodes, params, fused_pairs = graph_variant(axis, control == "fused")
            choices = [compile_program(nodes, p) for p in CONTRACT["compiler"]["policies_for_all_arms"]]
            program, selected = min(choices, key=lambda x: (x[1]["instructions"], x[1]["peak_live_words"]))
            validations = [validate_program(axis, p, params) for p, _ in choices]
            (HERE / f"{key}_program.json").write_text(json.dumps(program, indent=2) + "\n")
            vectors, expected = base.captured_tile(axis)
            measurements[key] = dict(fused_pairs=fused_pairs, selected=selected,
                choices=[info for _, info in choices], validations=validations,
                scenarios={s: simulate(program, s, HERE / f"{key}_{s}_timeline.csv", vectors, expected)
                           for s in CONTRACT["backpressure"]})
    comparisons = {}
    for scenario in CONTRACT["backpressure"]:
        row = {key: value["scenarios"][scenario]["service_slots"] for key, value in measurements.items()}
        row["unfused_structure_reduction_percent"] = 100 * (1 - row["lifting_raw_unfused"] / row["ordinary_unfused"])
        row["fused_structure_reduction_percent"] = 100 * (1 - row["lifting_raw_fused"] / row["ordinary_fused"])
        row["lifting_generic_fusion_reduction_percent"] = 100 * (1 - row["lifting_raw_fused"] / row["lifting_raw_unfused"])
        comparisons[scenario] = row
    result = dict(resource_contract="resource_contract.json", measurements=measurements,
        comparisons=comparisons, normalization_boundary_validation=check_norm_boundary(),
        added_interstage_allocated_bytes_per_arm=50,
        decision="SOURCE_INTERFACE_ONLY; full-chain service and relative AEE gate remain open")
    (HERE / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(comparisons=comparisons,
        selected={k: v["selected"] for k, v in measurements.items()}), indent=2))


if __name__ == "__main__":
    main()
