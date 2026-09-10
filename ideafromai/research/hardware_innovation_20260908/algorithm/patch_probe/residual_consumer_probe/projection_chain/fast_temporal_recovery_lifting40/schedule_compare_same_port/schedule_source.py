#!/usr/bin/python3.12
"""Fixed-resource source-PSN service experiment; not RTL or complete r1 service."""
from __future__ import annotations
from collections import Counter, deque
from pathlib import Path
import csv
import json
import random
import numpy as np

HERE = Path(__file__).resolve().parent
LIFT = HERE.parent
CHAIN = LIFT.parent
MIN24, MAX24 = -(1 << 23), (1 << 23) - 1
RESOURCE = dict(
    scope="stream I24 into persistent P2 -> complete source gate words; consumers not scheduled",
    lanes=8, channels=96, positions=2, work_words_per_lane=96, work_word_bits=48,
    work_rf="8 private flop RF banks, each 2 combinational reads and 1 end-of-slot write",
    instruction_rom=dict(words=512, bits=128, read_ports=1, read_latency="combinational ROM and decode, not a clock-qualified SRAM", control="SIMD broadcast; held while COMMIT stalls"),
    alu="one signed48 add/sub/round-increment/sat24/compare issue per lane per slot; result next slot",
    shifts="same dual programmable operand shift/select network on both axes; no timing/area claim",
    raw_I_bytes=5760, raw_I_bank="8 banks, each 1R1W signed24; I retained for both actual consumers",
    input="192 bits per slot; batch-major then t; values written at slot end; not initially resident",
    output=dict(lane_gate_bits=10, lane_valid_bits=10, fifo_entries=16, payload_bits=10, tag_bits=8,
                enqueue_per_slot=1, dequeue_per_slot=1, pop_before_push=True,
                commit="one pending mask per 8-lane group; wait until all admitted, then reuse collector"),
    scenarios=dict(ready=dict(period=1, blocked_prefix=0),
                   blocked=dict(period=1024, blocked_prefix=896)),
    caveat="service slots in a CPU model, without VCS/DC/PT/Formality, SRAM .db or clock claim",
)

def ref(node, shift=0, sign=1):
    return dict(node=int(node), shift=int(shift), sign=int(sign))

def shifted(r, extra=0, sign=1):
    return ref(r["node"], r["shift"] + int(extra), r["sign"] * int(sign))

def rne(v, shift):
    q, rem = divmod(int(v), 1 << shift)
    return q + int(rem > (1 << (shift - 1)) or (rem == (1 << (shift - 1)) and (q & 1)))

def sat(v):
    return min(MAX24, max(MIN24, int(v)))

def gate(v, threshold, direction, constant):
    if constant >= 0:
        return int(constant)
    return int(v >= threshold if direction > 0 else v <= threshold)

def folded_cutoff(threshold, direction, constant, shift):
    if constant >= 0:
        return 0, constant
    if direction > 0:
        if threshold <= MIN24: return 0, 1
        if threshold > MAX24: return 0, 0
        return (threshold << shift) - (1 << (shift - 1)) + (threshold & 1), -1
    if threshold >= MAX24: return 0, 1
    if threshold < MIN24: return 0, 0
    return (threshold << shift) + (1 << (shift - 1)) - (threshold & 1), -1

def graphs(axis):
    nodes = [dict(id=t, kind="input", source=t, parents=[]) for t in range(10)]
    def add(kind, parents, **kw):
        idx = len(nodes)
        nodes.append(dict(id=idx, kind=kind, parents=parents, **kw))
        return ref(idx)
    def import_graph(g, inputs):
        aliases = {}
        for n in g["nodes"]:
            if n["kind"] == "input":
                aliases[n["id"]] = inputs[n["source"]]
            else:
                aliases[n["id"]] = add("addsub", [
                    shifted(aliases[n["lhs"]], n["lhs_shift"]),
                    shifted(aliases[n["rhs"]], n["rhs_shift"], -1 if n["subtract"] else 1)])
        return [shifted(aliases[n["node"]], n["shift"], n["sign"]) for n in g["outputs"]]
    if axis == "ordinary":
        g = json.loads((LIFT / "ordinary_source_cmvm.integer_dag.json").read_text())
        outputs = import_graph(g, [ref(i) for i in range(10)])
        post = json.loads((LIFT / "ordinary_source_cmvm.json").read_text())["postprocess"]["rows"]
        for out, row in zip(outputs, post):
            add("gate", [out], output_t=row["t"], threshold=row["direct_dot_cutoff"],
                direction=row["direction"], constant=row["constant_gate"])
        params = np.load(CHAIN / "temporal_structured_recovery/fixed_valid825/identity_permuted_base_coordinate_constants.npz")
    else:
        bundle = json.loads((LIFT / "constant_compilation_graphs.json").read_text())["whole_halfstage_graphs"]
        params = np.load(LIFT / "fixed_lifting_valid825/fast_raw_diagonal_fixed_constants.npz")
        current = [ref(i) for i in range(10)]
        final_numerators = set()
        for half in range(8):
            item = bundle[f"fast_raw_diagonal/forward/{half}"]
            outputs = import_graph(item["whole_five_pairs_graph"], current)
            for out, coordinate in zip(outputs, item["write_time_indices"]):
                if half == 7:
                    current[coordinate] = out
                    final_numerators.add(coordinate)
                else:
                    rounded = add("round", [out], rne_shift=12)
                    current[coordinate] = add("sat", [rounded])
        for t, coordinate in enumerate(params["source_permutation"].tolist()):
            threshold = int(params["source_threshold"][t])
            direction = int(params["source_direction"][t])
            constant = int(params["source_constant"][t])
            if coordinate in final_numerators:
                threshold, constant = folded_cutoff(threshold, direction, constant, 12)
            add("gate", [current[coordinate]], output_t=t, threshold=threshold,
                direction=direction, constant=constant)
    return nodes, {k: params[k].tolist() for k in params.files}

def compile_program(nodes, policy):
    """Both axes receive the same topology/last-use scheduling and allocator."""
    use = Counter(p["node"] for n in nodes[10:] for p in {r["node"]: r for r in n["parents"]}.values())
    consumers = {n["id"]: [] for n in nodes}
    for n in nodes[10:]:
        for p in {r["node"] for r in n["parents"]}: consumers[p].append(n["id"])
    height = {}
    for n in reversed(nodes):
        height[n["id"]] = 1 + max((height[c] for c in consumers[n["id"]]), default=0)
    regs = {i: i for i in range(10)}
    free = set(range(10, RESOURCE["work_words_per_lane"]))
    done = set(range(10))
    pending = set(range(10, len(nodes)))
    program = [dict(kind="load", dst=i, source_t=i, logical_node=i) for i in range(10)]
    peak = len(regs)
    while pending:
        ready = [i for i in pending if all(p["node"] in done for p in nodes[i]["parents"])]
        def score(i):
            parents = {p["node"] for p in nodes[i]["parents"]}
            killed = sum(use[p] == 1 for p in parents)
            return (nodes[i]["kind"] == "gate", killed, height[i], -i)
        i = min(ready) if policy == "official" else max(ready, key=score)
        n = nodes[i]
        parents = {p["node"] for p in n["parents"]}
        operands = [dict(reg=regs[p["node"]], shift=p["shift"], sign=p["sign"]) for p in n["parents"]]
        for p in parents:
            use[p] -= 1
            if use[p] == 0:
                free.add(regs.pop(p))
        inst = {k: v for k, v in n.items() if k not in {"parents", "id"}}
        inst.update(operands=operands, logical_node=i)
        if n["kind"] != "gate":
            if not free:
                return None, dict(policy=policy, feasible=False, reason="work RF exhausted", logical_node=i)
            dest = min(free)
            free.remove(dest)
            regs[i] = dest
            inst["dst"] = dest
            if use[i] == 0: free.add(regs.pop(i))
        program.append(inst)
        done.add(i)
        pending.remove(i)
        peak = max(peak, len(regs))
    program.append(dict(kind="commit"))
    assert len(program) <= RESOURCE["instruction_rom"]["words"]
    return program, dict(policy=policy, feasible=True, peak_live_words=peak, instructions=len(program))

def run_program(program, vector):
    rf = [None] * RESOURCE["work_words_per_lane"]
    out = [None] * 10
    for ins in program:
        k = ins["kind"]
        if k == "load":
            rf[ins["dst"]] = int(vector[ins["source_t"]])
            continue
        if k == "commit": continue
        args = [(rf[p["reg"]] << p["shift"]) * p["sign"] for p in ins["operands"]]
        assert all(-(1 << 47) <= x < (1 << 47) for x in args)
        if k == "addsub": result = sum(args)
        elif k == "round": result = rne(args[0], ins["rne_shift"])
        elif k == "sat": result = sat(args[0])
        elif k == "gate":
            out[ins["output_t"]] = gate(args[0], ins["threshold"], ins["direction"], ins["constant"])
            continue
        else: raise ValueError(k)
        assert -(1 << 47) <= result < (1 << 47)
        rf[ins["dst"]] = result
    assert None not in out
    return out

def reference(axis, params, vector):
    if axis == "ordinary":
        values = [sat(rne(sum(int(a) * int(x) for a, x in zip(row, vector)), int(params["As_exponent"])))
                  for row in params["As_q16"]]
    else:
        values = list(vector)
        for layer, matching in enumerate(params["lifting_matchings"]):
            for half in range(2):
                for pair, (i, j) in enumerate(matching):
                    dst, src = (i, j) if half == 0 else (j, i)
                    q = params["lifting_q12"][layer][pair][half]
                    values[dst] = sat(rne((values[dst] << 12) + q * values[src], 12))
        values = [values[i] for i in params["source_permutation"]]
    return [gate(v, int(thr), int(d), int(c)) for v, thr, d, c in
            zip(values, params["source_threshold"], params["source_direction"], params["source_constant"])]

def validate_program(axis, program, params):
    vectors = [[0] * 10, [MIN24] * 10, [MAX24] * 10, [MIN24, MAX24] * 5]
    for t in range(10):
        for v in [MIN24, MAX24, -2049, -2048, -1, 1, 2048, 2049]:
            x = [0] * 10
            x[t] = v
            vectors.append(x)
    rand = random.Random(10)
    vectors += [[rand.randint(MIN24, MAX24) for _ in range(10)] for _ in range(512)]
    for x in vectors:
        assert run_program(program, x) == reference(axis, params, x), (axis, x)
    boundary_checks = 0
    for d in [-1, 1]:
        for threshold in [MIN24-1, MIN24, MIN24+1, -2, -1, 0, 1, 2, MAX24-1, MAX24, MAX24+1]:
            cutoff, const = folded_cutoff(threshold, d, -1, 12)
            for n in [cutoff-1, cutoff, cutoff+1, (threshold<<12)-2048, (threshold<<12)+2048]:
                assert gate(n, cutoff, d, const) == gate(sat(rne(n, 12)), threshold, d, -1)
                boundary_checks += 1
    return dict(source_vectors=len(vectors), source_mismatches=0, round_cutoff_boundary_checks=boundary_checks)

def simulate(program, scenario, trace_path, input_vectors, expected_words):
    lanes = RESOURCE["lanes"]
    vectors = RESOURCE["channels"] * RESOURCE["positions"]
    batches = vectors // lanes
    conf = RESOURCE["scenarios"][scenario]
    fifo = deque()
    counts = Counter()
    raw_reads = raw_writes = rf_reads = rf_writes = rom_reads = 0
    max_fifo = cycle = batch = pc = 0
    admitted = set()
    commit_fetched = False
    valid_mask = 0
    popped = []
    log = []
    raw_state = [[None] * 10 for _ in range(vectors)]
    work = [[None] * RESOURCE["work_words_per_lane"] for _ in range(lanes)]
    gate_words = [0] * lanes
    payload_checks = 0
    # All instructions have one-slot result latency. RF allocation happens at
    # compilation; reads precede writes, including last-use overwrite.
    while batch < batches or fifo:
        popped_id = enqueued_id = ""
        popped_word = enqueued_word = ""
        input_words = lanes if cycle < batches * 10 else 0
        raw_writes += input_words
        ready = (cycle % conf["period"]) >= conf["blocked_prefix"]
        if fifo and ready:
            popped_id, popped_word = fifo.popleft()
            assert popped_word == expected_words[popped_id], (popped_id, popped_word, expected_words[popped_id])
            payload_checks += 1
            popped.append(popped_id)
            counts["output_transfers"] += 1
        event = "drain"
        if batch < batches:
            ins = program[pc]
            kind = ins["kind"]
            if kind == "load" and cycle <= batch * 10 + ins["source_t"]:
                event = "input_wait"
            elif kind == "commit":
                if not commit_fetched:
                    rom_reads += 1
                    commit_fetched = True
                assert valid_mask == (1 << 10) - 1
                if len(fifo) < RESOURCE["output"]["fifo_entries"]:
                    lane = min(set(range(lanes)) - admitted)
                    enqueued_id = batch * lanes + lane
                    enqueued_word = gate_words[lane]
                    fifo.append((enqueued_id, enqueued_word))
                    admitted.add(lane)
                    event = "commit_enqueue"
                else:
                    event = "fifo_full_wait"
                if len(admitted) == lanes:
                    batch += 1
                    pc = 0
                    admitted.clear()
                    commit_fetched = False
                    valid_mask = 0
                    gate_words = [0] * lanes
            else:
                rom_reads += 1
                event = kind
                if kind == "load":
                    raw_reads += lanes
                    rf_writes += lanes
                    for lane in range(lanes):
                        value = raw_state[batch * lanes + lane][ins["source_t"]]
                        assert value is not None
                        work[lane][ins["dst"]] = value
                else:
                    rf_reads += lanes * len({p["reg"] for p in ins["operands"]})
                    for lane in range(lanes):
                        args = [(work[lane][p["reg"]] << p["shift"]) * p["sign"] for p in ins["operands"]]
                        if kind == "addsub": value = sum(args)
                        elif kind == "round": value = rne(args[0], ins["rne_shift"])
                        elif kind == "sat": value = sat(args[0])
                        elif kind == "gate":
                            bit = gate(args[0], ins["threshold"], ins["direction"], ins["constant"])
                            gate_words[lane] |= bit << ins["output_t"]
                            continue
                        else: raise ValueError(kind)
                        assert -(1 << 47) <= value < (1 << 47)
                        work[lane][ins["dst"]] = value
                    if kind == "gate":
                        valid_mask |= 1 << ins["output_t"]
                    else:
                        rf_writes += lanes
                pc += 1
        if input_words:
            input_batch, input_t = divmod(cycle, 10)
            for lane in range(lanes):
                v = input_batch * lanes + lane
                raw_state[v][input_t] = input_vectors[v][input_t]
        counts[event] += 1
        max_fifo = max(max_fifo, len(fifo))
        log.append([cycle, batch, pc, event, input_words, enqueued_id, enqueued_word, popped_id, popped_word, len(fifo)])
        cycle += 1
        if cycle > 1000000: raise RuntimeError("finite P2 schedule did not finish")
    assert popped == list(range(vectors))
    assert raw_state == input_vectors
    assert raw_reads == raw_writes == vectors * 10
    with trace_path.open("w") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["cycle", "batch_after", "pc_after", "event", "I24_input_words", "enqueued_vector", "enqueued_gate_word", "accepted_vector", "accepted_gate_word", "fifo_count"])
        writer.writerows(log)
    return dict(service_slots=cycle, event_slots=dict(counts), peak_fifo_entries=max_fifo,
                raw_input_bytes=raw_writes * 3, raw_I_read_bytes=raw_reads * 3,
                work_RF_read_bytes=rf_reads * 6, work_RF_write_bytes=rf_writes * 6,
                instruction_fetch_bytes=rom_reads * 16,
                gate_payload_bits=vectors * 10, fifo_tagged_transfer_bits=vectors * 18,
                gates_accepted=vectors, gate_payload_checks=payload_checks, gate_payload_mismatches=0, raw_I_unchanged=True, boundary="one P2 tile; raw I still alive for unscheduled consumers")

def captured_tile(axis):
    fixture = json.loads((HERE / "source_tile_fixture.json").read_text())
    return fixture["I24"], fixture["gate_words"][axis]

def main():
    HERE.mkdir(exist_ok=True)
    (HERE / "resource_contract.json").write_text(json.dumps(RESOURCE, indent=2) + "\n")
    results = {}
    for axis in ["ordinary", "lifting_raw"]:
        nodes, params = graphs(axis)
        input_vectors, expected_words = captured_tile(axis)
        choices = [compile_program(nodes, p) for p in ["official", "pressure"]]
        legal = [(program, info) for program, info in choices if program is not None]
        if not legal: raise RuntimeError(f"No finite RF schedule for {axis}; do not score as baseline defeat")
        program, chosen = min(legal, key=lambda x: (x[1]["instructions"], x[1]["peak_live_words"]))
        validations = [validate_program(axis, p, params) for p, _ in legal]
        (HERE / f"{axis}_program.json").write_text(json.dumps(program, indent=2) + "\n")
        results[axis] = dict(operations=dict(Counter(n["kind"] for n in nodes)),
                             compiler_choices=[info for _, info in choices], selected=chosen,
                             validation=validations,
                             programs_use_identical_privileges=True,
                             program_used_bytes=len(program) * 16,
                             scenarios={scenario: simulate(program, scenario, HERE / f"{axis}_{scenario}_timeline.csv", input_vectors, expected_words)
                                        for scenario in RESOURCE["scenarios"]})
    assert captured_tile("ordinary")[0] == captured_tile("lifting_raw")[0]
    comparisons = {}
    for scenario in RESOURCE["scenarios"]:
        a, b = [results[axis]["scenarios"][scenario]["service_slots"] for axis in ["ordinary", "lifting_raw"]]
        comparisons[scenario] = dict(ordinary=a, lifting_raw=b, service_reduction_percent=100*(1-b/a), ratio=a/b)
    result = dict(scope=RESOURCE["scope"], resource_contract="resource_contract.json",
                  work_RF_allocated_bytes=8*96*6, raw_I_reserved_bytes=5760, instruction_ROM_allocated_bytes=512*16,
                  gate_collectors_bits=8*20, fifo_storage_bits=16*18, commit_pending_bits=8,
                  measurements=results, comparisons=comparisons,
                  capture=dict(frame="zurich_city_09_a_0001.npy", positions_yx=[[0,0],[0,1]], equal_I24_between_axes=True),
                  decision="SOURCE_ONLY; full r1/dual-consumer Stage B pending")
    (HERE / "source_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(comparisons=comparisons, selected={k:v["selected"] for k,v in results.items()}), indent=2))

if __name__ == "__main__":
    main()
