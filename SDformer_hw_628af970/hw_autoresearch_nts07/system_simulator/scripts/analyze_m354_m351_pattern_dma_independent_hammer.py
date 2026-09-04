#!/usr/bin/env python3
"""Independent M354 hammer review of the frozen M351 correction overlay."""

from __future__ import division

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def assignment_attributes(tree, base_name):
    attributes = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if (isinstance(target, ast.Attribute) and
                    isinstance(target.value, ast.Name) and
                    target.value.id == base_name):
                attributes.append(target.attr)
    return attributes


def corrected_function_audit(tree):
    matches = [node for node in ast.walk(tree)
               if isinstance(node, ast.FunctionDef) and
               node.name == "corrected_tile_load_cycles"]
    require(len(matches) == 1, "corrected tile function count drift")
    function = matches[0]
    loaded_names = {node.id for node in ast.walk(function)
                    if isinstance(node, ast.Name) and
                    isinstance(node.ctx, ast.Load)}
    subscript_keys = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.Subscript):
            continue
        key_node = node.slice
        if isinstance(key_node, ast.Index):
            key_node = key_node.value
        if isinstance(key_node, ast.Str):
            subscript_keys.add(key_node.s)
        elif (isinstance(key_node, ast.Constant) and
              isinstance(key_node.value, str)):
            subscript_keys.add(key_node.value)
    expected_keys = {
        "used_pwp_patterns",
        "pwp_vector_bytes_per_output_block",
        "partition_bits",
        "weight_vector_bytes",
        "dram_bytes_per_cycle",
    }
    require("q" not in loaded_names,
            "corrected tile DMA still depends on q/pattern payload")
    require(subscript_keys == expected_keys,
            "corrected tile DMA field set drift: {}".format(subscript_keys))
    require("pattern_bytes" not in subscript_keys,
            "pattern bytes remain in corrected output-tile DMA")
    return {
        "function_count": 1,
        "q_or_pattern_term_in_tile_dma": False,
        "retained_dynamic_pwp_term": True,
        "retained_weight_term": True,
        "retained_dram_rounding": True,
        "observed_model_fields": sorted(subscript_keys),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--m351-replay", required=True, type=Path)
    parser.add_argument("--m339-replay", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output_dir.exists(), "refusing M354 output overwrite")

    contract = strict_json(args.contract)
    require(contract.get("schema") ==
            "m354_m351_pattern_dma_independent_hammer_contract_v1",
            "M354 contract schema drift")
    require(contract.get("status") == "FROZEN_BEFORE_M354_REVIEW",
            "M354 contract is not frozen")
    root = args.contract.resolve().parents[1]
    identities = {}
    paths = {}
    for name, identity in contract["inputs"].items():
        path = root / identity["path"]
        require(path.is_file(), "missing M354 input: " + str(path))
        observed = sha256(path)
        require(observed == identity["sha256"], "SHA drift for " + name)
        identities[name] = {"path": identity["path"], "sha256": observed}
        paths[name] = path

    frozen_m351_bytes = paths["m351_result"].read_bytes()
    frozen_m339_bytes = paths["m339_result"].read_bytes()
    require(args.m351_replay.read_bytes() == frozen_m351_bytes,
            "M351 exact-SHA replay is not byte-identical")
    require(args.m339_replay.read_bytes() == frozen_m339_bytes,
            "M339 exact-SHA replay is not byte-identical")

    m351_contract = strict_json(paths["m351_contract"])
    m344_contract = strict_json(paths["m344_contract"])
    m351 = strict_json(paths["m351_result"])
    m344 = strict_json(paths["m344_result"])
    m339 = strict_json(paths["m339_result"])
    m347 = strict_json(paths["m347_result"])
    require(m351_contract["cycle_model"] == m344_contract["cycle_model"],
            "M351 changed the M344 cycle model")
    require(m351["status"] ==
            "PASS_M351_PATTERN_DMA_CORRECTED_ANALYTICAL_RECURRENCE_UNADMITTED",
            "M351 status drift")
    require(m339["status"] ==
            "PASS_M339_EXACT_WORK_AND_PINNED_KFIRST_CYCLE_UPPER_UNADMITTED",
            "M339 status drift")
    require(m347["verdict"]["p0_count"] == 0 and
            any(issue["id"] == "P1-1" for issue in m347["issues"]["p1"]),
            "M347 review trigger drift")

    source = paths["m351_analyzer"].read_text(encoding="utf-8")
    tree = ast.parse(source)
    monkeypatches = Counter(assignment_attributes(tree, "m344"))
    require(monkeypatches == Counter({
        "candidate_tile_load_cycles": 1,
        "strict_json": 1,
    }), "unexpected M344 monkeypatch surface: {}".format(monkeypatches))
    function_audit = corrected_function_audit(tree)

    model = m351_contract["cycle_model"]
    require(model["samples"] == 10 and model["operators"] == 4 and
            model["partitions_per_operator"] == 432,
            "phase geometry drift")
    phases = (model["samples"] * model["operators"] *
              model["partitions_per_operator"])
    require(phases == 17280, "phase count drift")
    q_to_tile = {int(q): int(tile) for q, tile in
                 m351["mechanism"]["q_to_output_block_tile"].items()}
    require(q_to_tile == {16: 8, 32: 4, 64: 2, 128: 1},
            "q/output-tile mapping drift")
    require(all(q * tile == 128 for q, tile in q_to_tile.items()),
            "q times output tile invariant failure")

    work_by_q = {row["q_capacity"]: row
                 for row in m339["exact_runtime_working_set"]}
    work_rows = []
    capacity_rows = []
    for q, output_tile in sorted(q_to_tile.items()):
        work = work_by_q[q]
        require(work["candidate_vector_ops_per_block"] ==
                work["pwp_ops_per_block"] +
                work["correction_ops_per_block"],
                "M339 candidate work conservation failure")
        maximum_patterns = work["used_patterns_per_phase"]["maximum"]
        require(maximum_patterns == q, "maximum used-pattern drift")
        pwp_bytes = (maximum_patterns *
                     model["pwp_vector_bytes_per_output_block"] * output_tile)
        weight_bytes = (model["partition_bits"] *
                        model["weight_vector_bytes"] * output_tile)
        pattern_bytes = q * model["pattern_bytes"]
        per_context = pwp_bytes + weight_bytes + pattern_bytes
        double_context = 2 * per_context
        require(double_context <= model["pwp_weight_pattern_cache_bytes"],
                "capacity overflow")
        capacity_rows.append({
            "q": q,
            "output_block_tile": output_tile,
            "maximum_used_patterns": maximum_patterns,
            "pwp_bytes_per_context": pwp_bytes,
            "weight_bytes_per_context": weight_bytes,
            "pattern_bytes_reserved_per_context": pattern_bytes,
            "per_context_bytes": per_context,
            "double_context_bytes": double_context,
            "unused_bytes_per_32kib_context": 32768 - per_context,
            "fits_two_equal_32kib_contexts": True,
        })
        work_rows.append({
            key: work[key] for key in (
                "q_capacity", "bit_sparse_vector_ops_per_block",
                "candidate_vector_ops_per_block", "pwp_ops_per_block",
                "correction_ops_per_block", "exact_vector_op_speedup",
                "full_table_pwp_bytes", "selective_pwp_bytes",
                "selective_traffic_reduction", "used_patterns_per_phase")
        })

    descriptor_bytes = (2 * model["rows_per_operator"] *
                        model["descriptor_bytes_per_row"])
    tile_cache_bytes = model["pwp_weight_pattern_cache_bytes"]
    physical_bytes = tile_cache_bytes + descriptor_bytes
    require(descriptor_bytes == 36000 and tile_cache_bytes == 65536 and
            physical_bytes == 101536,
            "fixed physical allocation arithmetic failure")
    require(m351["mechanism"]["fixed_tile_cache_bytes"] == tile_cache_bytes and
            m351["mechanism"]["separate_two_context_descriptor_sram_bytes"] ==
            descriptor_bytes and
            m351["mechanism"]["fixed_physical_cache_plus_descriptor_bytes"] ==
            physical_bytes,
            "M351 physical allocation reporting drift")

    old_rows = {(row["q_capacity"], row["output_block_tile"], row["port"],
                 row["matcher_architecture"]): row
                for row in m344["cycle_bounds"]}
    new_rows = {(row["q_capacity"], row["output_block_tile"], row["port"],
                 row["matcher_architecture"]): row
                for row in m351["analytical_recurrences"]}
    expected_keys = {(q, output_tile, port, matcher)
                     for q, output_tile in q_to_tile.items()
                     for port in ("WIDE144_PWP_96_WEIGHT", "SHARED96")
                     for matcher in ("SYSTOLIC_Q_II1", "SERIAL16_II1")}
    require(set(old_rows) == expected_keys and set(new_rows) == expected_keys,
            "q/O/port/matcher row Cartesian product drift")

    static_fields = (
        "q_capacity", "output_block_tile", "output_tiles_per_partition",
        "port", "matcher_architecture",
        "maximum_context_bytes_including_weight_pattern_pwp",
        "double_context_bytes", "fits_64kb",
        "descriptor_sram_bytes_two_contexts", "bit_sparse_cycles",
        "cycle_admitted",
    )
    row_audit = []
    for key in sorted(expected_keys):
        old = old_rows[key]
        new = new_rows[key]
        for field in static_fields:
            require(old[field] == new[field],
                    "non-DMA row field changed: {} {}".format(key, field))
        q, output_tile, port, matcher = key
        capacity = next(row for row in capacity_rows if row["q"] == q)
        require(new["maximum_context_bytes_including_weight_pattern_pwp"] ==
                capacity["per_context_bytes"] and
                new["double_context_bytes"] == capacity["double_context_bytes"],
                "row capacity mismatch")
        require(new["bit_sparse_cycles"] == 543784143,
                "bit-sparse cycle drift")
        strict_cycles = new["analytical_serial_first_tile_cycles"]
        overlap_cycles = new["analytical_last_first_overlap_cycles"]
        require(overlap_cycles <= strict_cycles,
                "analytical overlap/serial ordering failure")
        require(math.isclose(new["analytical_serial_first_tile_speedup"],
                             new["bit_sparse_cycles"] / float(strict_cycles),
                             rel_tol=0.0, abs_tol=1e-12),
                "serial speedup division failure")
        require(math.isclose(new["analytical_last_first_overlap_speedup"],
                             new["bit_sparse_cycles"] / float(overlap_cycles),
                             rel_tol=0.0, abs_tol=1e-12),
                "overlap speedup division failure")
        strict_delta = (old["strict_first_tile_serial_cycles"] -
                        strict_cycles)
        overlap_delta = (old["last_tile_first_tile_overlap_cycles"] -
                         overlap_cycles)
        pattern_service_per_phase = int(math.ceil(
            q * model["pattern_bytes"] /
            float(model["dram_bytes_per_cycle"])))
        tiles = model["output_blocks"] // output_tile
        strict_min = phases * pattern_service_per_phase
        total_tile_pattern_service = strict_min * tiles
        overlap_min = model["samples"] * pattern_service_per_phase
        require(strict_min <= strict_delta <= total_tile_pattern_service,
                "strict correction outside pattern-only bounds")
        require(overlap_min <= overlap_delta <= total_tile_pattern_service,
                "overlap correction outside pattern-only bounds")
        row_audit.append({
            "q": q,
            "output_block_tile": output_tile,
            "port": port,
            "matcher": matcher,
            "bit_sparse_cycles": new["bit_sparse_cycles"],
            "analytical_serial_cycles": strict_cycles,
            "analytical_serial_speedup":
                new["analytical_serial_first_tile_speedup"],
            "analytical_overlap_cycles": overlap_cycles,
            "analytical_overlap_speedup":
                new["analytical_last_first_overlap_speedup"],
            "strict_cycles_removed_vs_m344": strict_delta,
            "overlap_cycles_removed_vs_m344": overlap_delta,
            "pattern_only_strict_delta_bounds": [strict_min,
                                                  total_tile_pattern_service],
            "pattern_only_overlap_delta_bounds": [overlap_min,
                                                   total_tile_pattern_service],
            "all_non_dma_row_fields_unchanged": True,
            "speedup_redivision_exact": True,
        })

    pattern_ledger = []
    for q, output_tile in sorted(q_to_tile.items()):
        service_per_phase = q * model["pattern_bytes"] // model[
            "dram_bytes_per_cycle"]
        require(service_per_phase * model["dram_bytes_per_cycle"] ==
                q * model["pattern_bytes"],
                "pattern transfer is not an integral service-cycle count")
        tile_loads_per_phase = model["output_blocks"] // output_tile
        pattern_ledger.append({
            "q": q,
            "output_block_tile": output_tile,
            "phases": phases,
            "pattern_loads_per_phase_after_m351": 1,
            "pattern_bytes_per_phase": q * model["pattern_bytes"],
            "pattern_bytes_total_after_m351":
                phases * q * model["pattern_bytes"],
            "pattern_dma_service_cycles_per_phase": service_per_phase,
            "pattern_dma_service_cycles_total_after_m351":
                phases * service_per_phase,
            "output_tile_loads_per_phase": tile_loads_per_phase,
            "raw_duplicate_tile_pattern_service_removed":
                phases * tile_loads_per_phase * service_per_phase,
            "pattern_reserved_in_capacity": True,
            "weight_retained_in_every_tile_dma": True,
            "selective_pwp_retained_in_every_tile_dma": True,
        })

    require(m351["correction"] == {
        "pattern_bytes_removed_from_each_output_tile_dma": True,
        "pattern_bytes_retained_in_each_context_capacity": True,
        "pattern_loads_per_phase": 1,
        "review_trigger": "M347 P1 duplicated pattern DMA",
        "sealed_parent_mutated": False,
        "weight_and_selective_pwp_bytes_retained_per_output_tile_dma": True,
    }, "M351 correction declaration drift")
    require(m351["admission"]["cycle_bound"] is False and
            m351["admission"]["finite_queue_executable_cycle"] is False and
            m351["admission"]["rtl_cycle_match"] is False and
            m351["admission"]["area_matched"] is False and
            m351["admission"]["energy"] is False and
            m351["admission"]["system_speedup"] is False and
            m351["admission"]["date_headline"] is False and
            all(row["analytical_recurrence_only"] is True and
                row["cycle_admitted"] is False
                for row in m351["analytical_recurrences"]),
            "analytical recurrence demotion boundary drift")

    findings = {
        "p0": [],
        "p1": [],
        "p2": [
            {
                "id": "M354-P2-01",
                "title": "Pattern physical ownership remains implicit",
                "impact": "M351 conservatively reserves pattern bytes in both tile-context capacity calculations but transfers one table per phase; a shared pattern SRAM versus one-context residence and release is not defined. This does not break capacity arithmetic, but it blocks executable scheduling.",
            },
            {
                "id": "M354-P2-02",
                "title": "The author result does not retain a per-phase DMA/component ledger",
                "impact": "The once-per-phase transfer and exposed max-term deltas require source inspection and a full 40-record replay. M354 reconstructs aggregate ledgers, but a future finite simulator should emit accepted DMA transactions directly.",
            },
            {
                "id": "M354-P2-03",
                "title": "M339 exact work is SHA-linked rather than copied into M351",
                "impact": "M351 truthfully gates and reproduces M339 but only emits a boolean and identity hash. M354 reruns M339 byte-identically and republishes all q16/q32/q64/q128 work rows for auditability.",
            },
        ],
    }
    score = 91
    verdict = {
        "go": [
            "M351 pattern-DMA duplicate-charge correction",
            "the four fixed-64KiB tile-cache capacity points",
            "101536-byte fixed tile-cache-plus-descriptor allocation statement",
            "all 16 corrected values when labeled analytical recurrences only",
            "q128/O1 SHARED96 SERIAL16 as the finite-context implementation seed",
        ],
        "no_go": [
            "calling either recurrence an executable cycle bound",
            "using any analytical speedup as measured or area-normalized performance",
            "claiming total module storage is only 64KiB",
            "promoting the wide/systolic peak to system speedup or DATE headline",
        ],
        "p0_count": 0,
        "p1_count": 0,
        "p2_count": 3,
        "limited_scope_verdict": "GO_M351_CORRECTION_AND_CAPACITY__NO_GO_EXECUTABLE_PERFORMANCE_OR_HEADLINE",
    }

    payload = {
        "schema": "m354_m351_pattern_dma_independent_hammer_v1",
        "status": "PASS_M354_INDEPENDENT_M351_HAMMER",
        "milestone": "M354",
        "date": "2026-08-25",
        "score_0_to_100": score,
        "verdict": verdict,
        "identity": identities,
        "exact_sha_replay": {
            "m351_byte_identical": True,
            "m351_sha256": sha256(args.m351_replay),
            "m339_byte_identical": True,
            "m339_sha256": sha256(args.m339_replay),
            "runtime_payload_records_rehashed_by_each_replay": 40,
        },
        "wrapper_audit": {
            "m344_attribute_assignments": dict(monkeypatches),
            "only_numeric_monkeypatch": "candidate_tile_load_cycles",
            "schema_compatibility_monkeypatch": "strict_json",
            "unexpected_m344_monkeypatches": [],
            "corrected_tile_load_function": function_audit,
            "finding": "The numeric wrapper changes only output-tile DMA: it removes q*pattern_bytes and retains weight, selective-PWP and DRAM rounding. M344 candidate_tile_bytes remains frozen and therefore still prices pattern capacity.",
        },
        "pattern_dma_ledger": pattern_ledger,
        "m339_exact_work_reproduction": {
            "byte_identical_replay": True,
            "all_q_rows": work_rows,
        },
        "capacity_recompute": {
            "formula": "max_used_patterns*144*O + 16*96*O + q*2 bytes per context",
            "rows": capacity_rows,
            "fixed_tile_cache_bytes": tile_cache_bytes,
            "separate_two_context_descriptor_sram_bytes": descriptor_bytes,
            "fixed_physical_cache_plus_descriptor_bytes": physical_bytes,
            "all_pairs_fit_two_equal_32kib_contexts": True,
        },
        "all_q_output_tile_port_matcher_rows": row_audit,
        "recurrence_boundary_audit": {
            "all_rows_analytical_recurrence_only": True,
            "all_rows_cycle_admitted_false": True,
            "top_level_cycle_bound_false": True,
            "finite_queue_executable_cycle_false": True,
            "missing_for_promotion": [
                "finite cache ownership and release states",
                "one shared 32-byte-per-cycle DMA server",
                "descriptor SRAM ports and banks",
                "bank-conflict and queue backpressure trace",
                "RTL cycle match",
                "area/Fmax normalization",
                "energy and system integration",
            ],
        },
        "findings": findings,
        "claim_boundary": "M354 admits M351's pattern-DMA correction, exact M339 work inheritance, fixed capacity arithmetic, 101536-byte cache-plus-descriptor allocation and analytical recurrence arithmetic only. It does not admit executable cycles, area-normalized speedup, energy, system speedup or a DATE headline.",
    }
    args.output_dir.mkdir(parents=True, exist_ok=False)
    result_path = (args.output_dir /
                   "m354_m351_pattern_dma_independent_hammer_r1.json")
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")

    selected = next(row for row in row_audit
                    if row["q"] == 128 and row["output_block_tile"] == 1 and
                    row["port"] == "SHARED96" and
                    row["matcher"] == "SERIAL16_II1")
    readme = """# M354：M351 pattern-DMA correction 独立打铁评审

结论：**91/100，P0/P1/P2 = 0/0/3。有限口径 GO M351 correction、固定容量与 analytical recurrence arithmetic；NO-GO executable performance、面积公平性能、系统倍速和 DATE headline。**

M351 与 M339 均由冻结合同完整重跑并和封存 JSON 字节一致；M339/M344/M347/M351 的两层 SHA 封存也由 runner 逐项验证。M351 wrapper 对 M344 只有两个 attribute assignment：`candidate_tile_load_cycles` 是唯一数值 monkeypatch，`strict_json` 只兼容 overlay contract schema。修正函数保留 weight、selective-PWP 和 32 B/cycle rounding，只从每个 output-tile DMA 删除 pattern；原 `candidate_tile_bytes` 未改，所以 pattern 仍保留在容量证明中。

10×4×432 = 17,280 个 phase 中，q16/q32/q64/q128 pattern 分别仍每 phase 搬一次，即 32/64/128/256 bytes，合计 DMA service 为 17,280/34,560/69,120/138,240 cycles。output-tile DMA 不再重复搬 pattern；weight 与实际使用的 PWP 没有漏收。16 个 q/O/port/matcher 组合的非 DMA 字段均与 M344 相同，修正周期差全部落在仅删除 pattern 可产生的严格边界内，speedup 重新除法逐项一致。

独立容量重算为：q16/O8 30,752 B/context、q32/O4 24,640 B、q64/O2 21,632 B、q128/O1 20,224 B；双 context 均小于 65,536 B。固定配置应报告 65,536 B tile cache 加 36,000 B descriptor SRAM，合计 **101,536 B**，不能简写成整个模块 64 KiB。

最可信实现种子 q128/O1 + SHARED96 + SERIAL16 的串行 analytical recurrence 为 {serial_cycles:,} cycles、{serial_speedup:.6f}x；乐观 overlap 为 {overlap_cycles:,} cycles、{overlap_speedup:.6f}x。这两项仍不是 executable bound，因为 pattern 物理归属、两个 cache slot 的释放时刻、单 DMA 仲裁、descriptor 端口、bank conflict、有限 queue、RTL cycle match、面积/Fmax 和能量均未实现。

三个 P2 是可审计性与后续实现缺口：pattern 只传一次但物理 residence 未明确；作者结果没有 per-phase DMA/component ledger；M339 work 在 M351 中以 SHA+boolean 继承而未展开。M354 已重放并公开所有 q work rows，但真正晋级仍需 finite-context simulator 的 accepted-transaction ledger。
""".format(
        serial_cycles=selected["analytical_serial_cycles"],
        serial_speedup=selected["analytical_serial_speedup"],
        overlap_cycles=selected["analytical_overlap_cycles"],
        overlap_speedup=selected["analytical_overlap_speedup"],
    )
    (args.output_dir / "README.md").write_text(readme, encoding="utf-8")
    print("M354_PASS score={} p0=0 p1=0 p2=3 q128_shared_serial={:.6f}x "
          "physical={}B".format(score,
                                selected["analytical_serial_speedup"],
                                physical_bytes), flush=True)


if __name__ == "__main__":
    main()
