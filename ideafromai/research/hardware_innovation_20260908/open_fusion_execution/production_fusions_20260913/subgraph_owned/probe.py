#!/opt/anaconda3/bin/python3.12
"""Fixed source-pair exact common-subgraph inheritance on archived compiled DAGs.

No compiler import, training, GPU, RTL mutation, or capacity search. The optional
new operations are a CPU resource model, not instructions supported by the RTL.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OPEN = HERE.parents[1]
BREADTH = OPEN / "breadth_20260912"
PAIRS = [(i, i + 1) for i in range(0, 10, 2)]
ARMS = {
    "dense_two_term": BREADTH / "source_constant_probe/dense",
    "lifting40_two_term": BREADTH / "source_constant_probe/lifting40",
    "contiguous34": BREADTH / "source_execution/contiguous34",
}
LOW, HIGH = -(1 << 23), (1 << 23) - 1


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist()
                               if hasattr(x, "tolist") else int(x)) + "\n")


def signature(d):
    return tuple(sorted((int(k), int(v)) for k, v in d.items() if v))


def contract_signature(sig, pair):
    a, b = pair
    d = dict(sig)
    d[a] = d.get(a, 0) + d.pop(b, 0)
    return signature(d)


def inspect_dag(program):
    """Expand only additions. Every real norm24 is a distinct opaque leaf."""
    sig, arithmetic = {}, []
    for pc, ins in enumerate(program):
        kind = ins["kind"]
        if kind == "load":
            sig[ins["logical_node"]] = ((ins["source_t"], 1),)
        elif kind == "addsub":
            d = defaultdict(int)
            for p in ins["operands"]:
                for k, v in sig[p["node"]]:
                    d[k] += v * p["sign"] * (1 << p["shift"])
            s = signature(d)
            sig[ins["logical_node"]] = s
            arithmetic.append(dict(node=ins["logical_node"], pc=pc, signature=s,
                                   dst=ins["dst"]))
        elif kind == "norm24":
            sig[ins["logical_node"]] = ((1000 + ins["logical_node"], 1),)

    # Any unconditional expression duplicate is deliberately excluded from X.
    static_groups = defaultdict(list)
    for n in arithmetic:
        static_groups[n["signature"]].append(n["node"])
    static_duplicates = [v for v in static_groups.values() if len(v) > 1]
    all_families, selected = [], []
    for pair_index, pair in enumerate(PAIRS):
        groups = defaultdict(list)
        for n in arithmetic:
            groups[contract_signature(n["signature"], pair)].append(n)
        families = []
        for key, group in groups.items():
            # Exact original signature equality is ordinary CSE, never a hit.
            if len({n["signature"] for n in group}) < 2:
                continue
            anchor = group[0]
            followers = [n for n in group[1:] if n["signature"] != anchor["signature"]]
            if not followers:
                continue
            families.append(dict(pair_index=pair_index, pair=list(pair),
                                 anchor=anchor, followers=followers,
                                 contracted_signature=key))
        families.sort(key=lambda f: (f["followers"][0]["pc"], f["anchor"]["pc"]))
        all_families.extend(families)
        if families:
            chosen = dict(families[0], cache_reg=90 + pair_index)
            selected.append(chosen)
    unary = [n for n in arithmetic if len(n["signature"]) == 1]
    return dict(arithmetic_nodes=len(arithmetic), raw_unary_nodes=[n for n in unary
                if n["signature"][0][0] < 10], static_duplicate_groups=static_duplicates,
                all_families=all_families, selected=selected,
                opaque_norm_boundaries=sum(n["kind"] == "norm24" for n in program),
                eligible_node_pair_relations=sum(len(f["followers"]) for f in all_families),
                selected_node_pair_relations=sum(len(f["followers"]) for f in selected))


def rne(x, shift):
    if shift == 0:
        return x.copy()
    q = x >> shift
    rem = x & ((1 << shift) - 1)
    half = 1 << (shift - 1)
    return q + ((rem > half) | ((rem == half) & ((q & 1) != 0)))


def gate(x, ins):
    if ins["constant"] >= 0:
        return np.full(x.shape, bool(ins["constant"]), dtype=bool)
    return x >= ins["threshold"] if ins["direction"] > 0 else x <= ins["threshold"]


def evaluate(program, x, plan=None):
    """Real signed48 operands/results, actual physical RF tags, exact reuse."""
    shape = (len(x), 8)
    work = [None] * 96
    tags = [None] * 96
    values, out = {}, np.zeros((len(x), 10, 8), bool)
    selected = [] if plan is None else plan["selected"]
    by_follow = defaultdict(list)
    by_anchor = defaultdict(list)
    for f in selected:
        by_anchor[f["anchor"]["node"]].append(f)
        for n in f["followers"]:
            by_follow[n["node"]].append(f)
    eq = [x[:, a] == x[:, b] for a, b in PAIRS]
    cache = {}
    checked_writes = changed_lanes = 0
    for ins in program:
        kind = ins["kind"]
        if kind in {"nop", "commit"}:
            continue
        node = ins["logical_node"]
        if kind == "load":
            v = x[:, ins["source_t"]].astype(np.int64)
        else:
            args = []
            for p in ins["operands"]:
                assert tags[p["reg"]] == p["node"], (node, p, tags[p["reg"]])
                operand = work[p["reg"]] * p["sign"] * (1 << p["shift"])
                assert operand.min() >= -(1 << 47) and operand.max() < (1 << 47)
                args.append(operand)
            if kind == "addsub":
                v = args[0] + args[1]
            elif kind == "norm24":
                v = np.clip(rne(args[0], ins["rne_shift"]), LOW, HIGH)
            elif kind == "gate":
                out[:, ins["output_t"]] = gate(args[0], ins)
                continue
            else:
                raise ValueError(kind)
        assert v.min() >= -(1 << 47) and v.max() < (1 << 47)
        if node in by_follow:
            # Fixed priority by pair id; overlapped masks are not double counted.
            used = np.zeros(shape, bool)
            for f in sorted(by_follow[node], key=lambda z: z["pair_index"]):
                mask = eq[f["pair_index"]] & ~used
                shared = cache[f["pair_index"]]
                assert np.array_equal(v[mask], shared[mask])
                v = np.where(mask, shared, v)
                used |= mask
            changed_lanes += int(used.sum())
        work[ins["dst"]] = v
        tags[ins["dst"]] = node
        values[node] = v
        checked_writes += v.size
        for f in by_anchor.get(node, []):
            cache[f["pair_index"]] = v.copy()
    return out, values, dict(rf_lane_writes_checked=checked_writes,
                            inherited_addsub_lanes=changed_lanes)


def ready_cycles(program):
    """Read-only CPU transcription of existing RTL ready handshake/pipeline.

    COPY/MERGE/CMP extensions below use identical 2R1W and latency rules but are
    hypothetical CPU operations. Only the original program is RTL validated.
    """
    pc, state, beat, store_beat = 0, "EXEC", 0, 0
    p1 = p2 = None
    cycles = 1  # IDLE/start edge before EXEC, also counted by archived tb.cpp.
    counts = Counter()
    while True:
        ins = program[pc]
        k = ins["kind"]
        sources = [p["reg"] for p in ins.get("operands", [])]
        dst = 95 if k == "gate" else ins.get("dst", -1000)
        hazard = any(p is not None and (p == dst or p in sources) for p in (p1, p2))
        arithmetic = k in {"addsub", "norm24", "gate", "copy", "merge", "cmp"}
        issue = (state == "EXEC" and arithmetic and not hazard) or (
            state == "LOAD_ISSUE" and p1 is None and p2 is None)
        nxt = None
        if issue:
            nxt = dst
            counts["issue_" + k] += 1
        if state == "EXEC":
            if k == "load":
                beat, state = 0, "RD_REQ"
            elif k == "nop":
                pc += 1
                counts["nop"] += 1
            elif k == "commit":
                state = "COMMIT_WAIT"
            elif issue:
                if k == "gate":
                    state = "GATE_WAIT"
                else:
                    pc += 1
            elif hazard:
                counts["hazard_wait"] += 1
        elif state == "RD_REQ":
            state = "RD_RSP"
        elif state == "RD_RSP":
            if beat == 2:
                state = "LOAD_ISSUE"
            else:
                beat += 1
                state = "RD_REQ"
        elif state == "LOAD_ISSUE":
            if issue:
                state = "LOAD_WAIT"
        elif state == "LOAD_WAIT":
            if p1 is None and p2 is None:
                pc += 1
                state = "EXEC"
        elif state == "GATE_WAIT":
            if p1 is None and p2 is None:
                state = "GATE_COLLECT"
        elif state == "GATE_COLLECT":
            pc += 1
            state = "EXEC"
        elif state == "COMMIT_WAIT":
            if p1 is None and p2 is None:
                store_beat, state = 0, "STORE"
        elif state == "STORE":
            if store_beat:
                cycles += 1
                break
            store_beat = 1
        p2, p1 = p1, nxt
        cycles += 1
        assert cycles < 10000
    return cycles, dict(counts)


def simplify_references(program, plan):
    """Ordinary liveness control: existing unoverwritten RF is already saved."""
    selected = []
    for original in plan["selected"]:
        f = dict(original)
        anchor_pc = f["anchor"]["pc"]
        final_pc = max(n["pc"] for n in f["followers"])
        reg = f["anchor"]["dst"]
        overwritten = any(n.get("dst") == reg for n in program[anchor_pc+1:final_pc])
        f["needs_save"] = overwritten
        f["reference_node"] = "cache"+str(f["pair_index"]) if overwritten else f["anchor"]["node"]
        if not overwritten:
            f["cache_reg"] = reg
        selected.append(f)
    return selected


def augment(program, plan, mask_classes, ordinary_simplify=False):
    """Fixed five compares, selected saves, full copy or partial 2R1W merge."""
    out = []
    anchors, followers = defaultdict(list), defaultdict(list)
    families = simplify_references(program, plan) if ordinary_simplify else [
        dict(f, needs_save=True, reference_node="cache"+str(f["pair_index"])) for f in plan["selected"]]
    for f in families:
        anchors[f["anchor"]["node"]].append(f)
        for n in f["followers"]:
            followers[n["node"]].append(f)
    for pc, ins in enumerate(program):
        node = ins.get("logical_node")
        chosen = [f for f in followers.get(node, []) if mask_classes[f["pair_index"]]]
        # This implementation treats each family sequentially. If two masks
        # cover all lanes jointly, no free multi-parent select is assumed.
        whole = next((f for f in chosen if mask_classes[f["pair_index"]] == 2), None)
        if whole:
            out.append(dict(kind="copy", dst=ins["dst"], logical_node=node,
                            operands=[dict(reg=whole["cache_reg"], node=whole["reference_node"], shift=0, sign=1)]))
        else:
            out.append(ins)
            for f in chosen:
                out.append(dict(kind="merge", dst=ins["dst"], logical_node=node,
                                pair_index=f["pair_index"], operands=[
                    dict(reg=ins["dst"], node=node, shift=0, sign=1),
                    dict(reg=f["cache_reg"], node=f["reference_node"], shift=0, sign=1)]))
        for f in anchors.get(node, []):
            if f["needs_save"]:
                out.append(dict(kind="copy", dst=f["cache_reg"], logical_node="cache"+str(f["pair_index"]),
                                operands=[dict(reg=ins["dst"], node=node, shift=0, sign=1)]))
        if pc == 9:
            assert ins["kind"] == "load"
            for j, (a, b) in enumerate(PAIRS):
                if ordinary_simplify and j not in {f["pair_index"] for f in families}:
                    continue
                # Flags use 40 explicitly charged state bits, not extra RF.
                out.append(dict(kind="cmp", dst=-1-j, pair_index=j,
                                operands=[dict(reg=a, node=a, shift=0, sign=1),
                                          dict(reg=b, node=b, shift=0, sign=1)]))
    return out


def execute_paid(program, x):
    """Execute actual saved RF90..94 values and 2R1W merge/copy operations."""
    work, tags, flags = [None]*96, [None]*96, [None]*5
    out = np.zeros((len(x), 10, 8), bool)
    counts = Counter()
    for ins in program:
        kind = ins["kind"]
        if kind in {"nop", "commit"}:
            continue
        args = []
        for p in ins.get("operands", []):
            assert tags[p["reg"]] == p["node"], (ins, p, tags[p["reg"]])
            args.append(work[p["reg"]] * p["sign"] * (1 << p["shift"]))
        counts[kind+"_vector_issues"] += len(x)
        counts["rf_vector_reads"] += len(args)*len(x)
        if kind == "cmp":
            flags[ins["pair_index"]] = args[0] == args[1]
            counts["flag_lane_writes"] += 8*len(x)
            continue
        if kind == "load":
            v = x[:, ins["source_t"]].astype(np.int64)
        elif kind == "addsub":
            v = args[0] + args[1]
        elif kind == "norm24":
            v = np.clip(rne(args[0], ins["rne_shift"]), LOW, HIGH)
        elif kind == "gate":
            out[:, ins["output_t"]] = gate(args[0], ins)
            continue
        elif kind == "copy":
            v = args[0].copy()
        elif kind == "merge":
            assert flags[ins["pair_index"]] is not None
            v = np.where(flags[ins["pair_index"]], args[1], args[0])
        else:
            raise ValueError(kind)
        assert v.min() >= -(1<<47) and v.max() < (1<<47)
        work[ins["dst"]] = v
        tags[ins["dst"]] = ins["logical_node"]
        counts["rf_vector_writes"] += len(x)
    return out, dict(counts)


def literal_reference(x, params, lifting):
    value = x.transpose(1, 0, 2).reshape(10, -1).astype(np.int64)
    if lifting:
        for layer in range(4):
            for half in [0, 1]:
                for j, pair in enumerate(params["lifting_matchings"][layer]):
                    a, b = int(pair[half]), int(pair[1-half])
                    q = int(params["lifting_q12"][layer, j, half])
                    value[a] = np.clip(rne(4096*value[a]+q*value[b], 12), LOW, HIGH)
        value = value[params["source_permutation"]]
    else:
        value = np.clip(rne(params["As_q16"].astype(np.int64) @ value,
                            int(params["As_exponent"])), LOW, HIGH)
    result = np.empty_like(value, bool)
    for t in range(10):
        result[t] = gate(value[t], dict(threshold=int(params["source_threshold"][t]),
                                       direction=int(params["source_direction"][t]),
                                       constant=int(params["source_constant"][t])))
    return result.reshape(10, len(x), 8).transpose(1, 0, 2)


def exercise_paid(program, x, plan, ordinary_simplify=False):
    eq = [x[:, a] == x[:, b] for a, b in PAIRS]
    classes = np.stack([np.where(np.all(m, axis=1), 2, np.any(m, axis=1).astype(int)) for m in eq], axis=1)
    patterns = Counter(map(tuple, classes.tolist()))
    total_cycles, profiles, counts = 0, [], Counter()
    gates = np.zeros((len(x), 10, 8), bool)
    for pattern, repetitions in sorted(patterns.items()):
        changed = augment(program, plan, pattern, ordinary_simplify=ordinary_simplify)
        indices = np.all(classes == np.asarray(pattern), axis=1)
        gates[indices], operations = execute_paid(changed, x[indices])
        counts.update(operations)
        cost, events = ready_cycles(changed)
        total_cycles += cost*repetitions
        profiles.append(dict(classes=pattern, tiles=repetitions, cycles_per_tile=cost,
                             program_words=len(changed), event_counts=events))
    return gates, total_cycles, profiles, dict(counts)


def stats_for_mask(eq, all_families, selected):
    def count(families):
        followers = defaultdict(list)
        for f in families:
            for n in f["followers"]:
                followers[n["node"]].append(f["pair_index"])
        lane_hits = full_hits = partial_vectors = 0
        per_node = []
        for node, pair_ids in followers.items():
            mask = np.logical_or.reduce([eq[i] for i in pair_ids])
            full, some = np.all(mask, axis=1), np.any(mask, axis=1)
            lane_hits += int(mask.sum())
            full_hits += int(full.sum())
            partial_vectors += int((some & ~full).sum())
            per_node.append(dict(node=node, lane_hits=int(mask.sum()),
                                 full_vectors=int(full.sum()), partial_vectors=int((some & ~full).sum())))
        return dict(distinct_follower_nodes=len(followers), lane_hits=lane_hits,
                    full_vectors=full_hits, partial_vectors=partial_vectors, per_node=per_node)
    return dict(all_eligible=count(all_families), fixed_selected=count(selected))


def load_inputs(label):
    with np.load(BREADTH / "source_constant_probe/dense/cpu_gold.npz") as z:
        a = z[label + "_I24"]
    with np.load(BREADTH / "source_constant_probe/lifting40/cpu_gold.npz") as z:
        assert np.array_equal(a, z[label + "_I24"])
    t, c, h, w = a.shape
    assert (t, c) == (10, 96)
    x = a.reshape(10, 12, 8, h, w).transpose(3, 4, 1, 0, 2).reshape(-1, 10, 8)
    return x, dict(height=h, width=w, tiles=len(x), input_values=x.size)


def boundary_checks():
    n = 0
    for q in [LOW-1, LOW, LOW+1, -2, -1, 0, 1, 2, HIGH-1, HIGH, HIGH+1]:
        for rem in [-1, 0, 2047, 2048, 2049, 4095, 4096]:
            v = q * 4096 + rem
            want = min(HIGH, max(LOW, round(Fraction(v, 4096))))
            assert int(np.clip(rne(np.array([v], np.int64), 12), LOW, HIGH)[0]) == want
            n += 1
    return dict(negative_tie_and_sat24_cases=n, mismatches=0)


def main():
    report = dict(interface="one fixed raw-input-pair substitution; norm24 opaque",
                  source_pairs=PAIRS, result_cache_capacity_vectors=5,
                  mask_state_bits=40, boundary_checks=boundary_checks(), arms={})
    table = []
    for arm, folder in ARMS.items():
        program = json.loads((folder / "program.json").read_text())
        old = json.loads((folder / "results.json").read_text())
        plan = inspect_dag(program)
        baseline, cycle_counts = ready_cycles(program)
        archived = {r["cycles"] // r["tiles"] for r in old["rows"] if r["mode"] == "ready"}
        assert archived == {baseline}, (arm, baseline, archived)
        assert max(i.get("dst", 0) for i in program) < 90
        assert len(program) + 5 + len(plan["selected"]) + plan["selected_node_pair_relations"] <= 512
        detail = dict(archived_program=str(folder.relative_to(OPEN) / "program.json"),
                      plan=plan, baseline_ready_cycles=baseline, baseline_cycle_counts=cycle_counts,
                      baseline_cycles_exactly_match_archived_RTL=True,
                      baseline_work_rf_peak=old["accounting"]["peak_live_words"],
                      reserved_result_rf_vectors=len(plan["selected"]),
                      conservative_work_rf_peak=old["accounting"]["peak_live_words"]+len(plan["selected"]),
                      gate_collector_rf=95, windows=[])
        simplified = simplify_references(program, plan)
        detail["ordinary_liveness_reference_control"] = simplified
        detail["ordinary_control_added_result_RF_vectors"] = sum(f["needs_save"] for f in simplified)
        detail["ordinary_control_flag_bits"] = 8*len(simplified)
        assert detail["conservative_work_rf_peak"] + 1 <= 96
        params_file = (folder / "deployed_constants.npz") if arm != "contiguous34" else (
            BREADTH / "algorithm/matched_training/contiguous34/stage320/deployed_constants.npz")
        params = dict(np.load(params_file))
        rng = np.random.default_rng(13)
        directed = rng.integers(LOW, HIGH+1, size=(12, 10, 8), dtype=np.int64)
        for j, (a, b) in enumerate(PAIRS):
            directed[j, b] = directed[j, a]  # full-vector inherited path
            directed[5+j, b, :4] = directed[5+j, a, :4]  # partial merge
        directed[10] = LOW
        directed[11] = HIGH
        wanted = literal_reference(directed, params, arm.startswith("lifting"))
        direct_gates, _, _, direct_ops = exercise_paid(program, directed, plan)
        assert np.array_equal(direct_gates, wanted)
        direct_ordinary, _, _, direct_ordinary_ops = exercise_paid(program, directed, plan, ordinary_simplify=True)
        assert np.array_equal(direct_ordinary, wanted)
        detail["directed_positive_negative_full_partial_checks"] = dict(
            tiles=len(directed), gate_bits=int(wanted.size), mismatches=0,
            operations=direct_ops, ordinary_liveness_control_operations=direct_ordinary_ops,
            independent_literal_matrix_or_40_halfstep_reference=True)
        for label in ["corner", "interior"]:
            x, metadata = load_inputs(label)
            expected = np.fromfile(folder / (label + "_gates.bin"), dtype="<u2").reshape(-1, 8)
            baseline_gates, baseline_values, baseline_check = evaluate(program, x)
            inherited_gates, inherited_values, inherited_check = evaluate(program, x, plan)
            packed = sum(baseline_gates[:, t].astype(np.uint16) << t for t in range(10))
            assert np.array_equal(packed, expected)
            assert np.array_equal(baseline_gates, literal_reference(x, params, arm.startswith("lifting")))
            assert np.array_equal(inherited_gates, baseline_gates)
            for node, value in inherited_values.items():
                assert np.array_equal(value, baseline_values[node])
            eq = [x[:, a] == x[:, b] for a, b in PAIRS]
            hits = stats_for_mask(eq, plan["all_families"], plan["selected"])
            paid_gates, modeled_cycles, profile, paid_ops = exercise_paid(program, x, plan)
            assert np.array_equal(paid_gates, baseline_gates)
            ordinary_gates, ordinary_cycles, ordinary_profile, ordinary_ops = exercise_paid(
                program, x, plan, ordinary_simplify=True)
            assert np.array_equal(ordinary_gates, baseline_gates)
            row = dict(window=label, **metadata, gate_bits=int(baseline_gates.size),
                       gate_mismatches=0, intermediate_mismatches=0,
                       baseline_check=baseline_check, inherited_check=inherited_check,
                       equal_input_lanes_per_pair=[int(m.sum()) for m in eq],
                       equal_input_whole_vectors_per_pair=[int(np.all(m, axis=1).sum()) for m in eq],
                       hit_accounting=hits, baseline_total_ready_cycles=baseline*len(x),
                       hypothetical_paid_total_ready_cycles=modeled_cycles,
                       hypothetical_paid_change_pct=100*(modeled_cycles/(baseline*len(x))-1),
                       fixed_compare_vector_issues=5*len(x), fixed_compare_rf_vector_reads=10*len(x),
                       result_save_vector_writes=len(plan["selected"])*len(x),
                       result_cache_peak_bits=len(plan["selected"])*8*48,
                       paid_physical_RF_and_flag_operations=paid_ops,
                       paid_physical_RF_tag_checked=True, paid_gate_mismatches=0,
                       paid_addsub_lane_executions_avoided=(plan["arithmetic_nodes"]*len(x)-paid_ops.get("addsub_vector_issues", 0))*8,
                       load_SR64_reads=30*len(x), commit_SW64_writes=2*len(x),
                       mask_class_profiles=profile,
                       upper_bound_note="Even free recognition/storage/alias cannot remove any addsub if all-eligible full_vectors is zero. Lane hits do not imply vector service.")
            row["ordinary_liveness_control"] = dict(
                paid_total_ready_cycles=ordinary_cycles,
                paid_change_pct=100*(ordinary_cycles/(baseline*len(x))-1),
                operations=ordinary_ops, mask_class_profiles=ordinary_profile,
                physical_RF_tags_checked=True, gate_mismatches=0,
                compare_vector_issues=len(simplified)*len(x),
                extra_result_save_vector_writes=sum(f["needs_save"] for f in simplified)*len(x))
            row["matched_input_zero_lanes_per_pair"] = [int(((x[:, a] == 0) & m).sum()) for (a, _), m in zip(PAIRS, eq)]
            row["full_match_spatial_locations_per_pair"] = [
                [dict(y=int(t)//12//metadata["width"], x=int(t)//12%metadata["width"], channel_group=int(t)%12)
                 for t in np.flatnonzero(np.all(m, axis=1))] for m in eq]
            # Every actual addsub execution occupies <=3 cycles including both
            # pipeline hazards. This deliberately generous 3-cycle/node bound
            # grants free alias, RF, comparisons, and control, for the SAME
            # original schedule; it does not credit deletion of load/gate I/O.
            row["all_eligible_generous_source_cycle_saving_bound"] = 3*hits["all_eligible"]["full_vectors"]
            row["all_eligible_generous_source_saving_bound_pct"] = 100*row["all_eligible_generous_source_cycle_saving_bound"]/(baseline*len(x))
            detail["windows"].append(row)
            table.append(dict(arm=arm, window=label, baseline_ready_per_tile=baseline,
                              raw_unary_nodes=len(plan["raw_unary_nodes"]),
                              all_candidate_relations=plan["eligible_node_pair_relations"],
                              selected_relations=plan["selected_node_pair_relations"],
                              selected_families=len(plan["selected"]),
                              selected_lane_hits=hits["fixed_selected"]["lane_hits"],
                              all_full_vector_hits=hits["all_eligible"]["full_vectors"],
                              paid_cpu_cycles=modeled_cycles, baseline_cycles=baseline*len(x),
                              paid_change_pct=row["hypothetical_paid_change_pct"],
                              ordinary_control_cycles=ordinary_cycles,
                              ordinary_control_change_pct=row["ordinary_liveness_control"]["paid_change_pct"],
                              generous_saving_bound_pct=row["all_eligible_generous_source_saving_bound_pct"]))
        report["arms"][arm] = detail
    dump(HERE / "results.json", report)
    with (HERE / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(table)
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
