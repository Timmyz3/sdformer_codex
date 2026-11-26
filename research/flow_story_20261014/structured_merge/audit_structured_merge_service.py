#!/usr/bin/env python3
"""Read-only source/results audit; integer geometry, not model/PE simulation.

Uses stdlib only. No torch, checkpoint, GPU, server, or old-code mutation.
Writes its own JSON receipt when --output is supplied; no scientific-quality fit.
"""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESEARCH = HERE.parents[1]
MODULE = HERE / "structured_merge_mlp.py"
RUNNER = HERE / "run_structured_merge_screen.py"
STATIC = RESEARCH / "flow_story_20261011/preflight/execution_contract/execution_services_results.json"
RESULT_SHA = "70208e3c0378bed2127d41e47745a373f1c02aad49b07da2a79af0ad9be1e9fb"
MODULE_SHA = "26454de4009e20d45edc186eaf255ef1df918b2250f270a239713efb90fe72f0"
RUNNER_SHA = "f23667f986ce6f83d6a51a7056ff25e82432d05def3bbd174530f17afc2551bf"
STATIC_SHA = "3cc67c3c331dc8475cd1647003a684bb6a36843e8b6ce00faa5b41f1f30968e8"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def runner_constants():
    values = {}
    for node in ast.parse(RUNNER.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ("STAGE_SHAPES", "DEPTHS", "ARMS"):
                    values[target.id] = ast.literal_eval(node.value)
    require(set(values) == {"STAGE_SHAPES", "DEPTHS", "ARMS"}, "Missing runner constants")
    return values


def extracted_source_ledger():
    # Compile only this pure counting function; do not import the torch module.
    nodes = [n for n in ast.parse(MODULE.read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name == "service_ledger"]
    require(len(nodes) == 1, "Missing source ledger")
    tree = ast.Module(body=nodes)
    tree.type_ignores = []
    env = {"math": math, "TILE": 9}
    exec(compile(tree, str(MODULE), "exec"), env)
    return env["service_ledger"]


def geometry(shape, mask):
    t, b, h, w, c = shape
    require(t == 10 and b == 1, "Not full T10/B1")
    require(len(mask) == (h+8)//9 and all(len(row) == (w+8)//9 for row in mask), "Mask extent")
    fine = coarse = reps = 0
    fine_shapes, coarse_shapes, areas = [], [], Counter()
    for iy, row in enumerate(mask):
        for ix, keep_fine in enumerate(row):
            require(type(keep_fine) is bool, "Nonboolean decision")
            vh, vw = min(9, h-9*iy), min(9, w-9*ix)
            if keep_fine:
                fine += vh*vw
                fine_shapes.append([t, b, vh, vw, c])
            else:
                coarse += vh*vw
                rr = 0
                for cy in range(0, vh, 3):
                    for cx in range(0, vw, 3):
                        area = min(3, vh-cy)*min(3, vw-cx)
                        areas[area] += 1
                        rr += 1
                reps += rr
                coarse_shapes.append([t, b, (vh+2)//3, (vw+2)//3, c])
    require(fine+coarse == h*w, "Coverage differs")
    require(sum(k*v for k, v in areas.items()) == coarse, "Cell areas differ")
    return dict(fine=fine, coarse=coarse, reps=reps, fine_shapes=fine_shapes,
                coarse_shapes=coarse_shapes, coarse_cell_area_histogram=dict(sorted(areas.items())))


def counts(shape, hidden, points):
    t, b, _, _, c = shape
    rows = t*b*points
    return dict(fc_rows_each=rows, fc_weighted_terms_both=2*rows*c*hidden,
                psn_real_macs_both=t*t*b*points*(c+hidden),
                bn_elements_both=rows*(c+hidden), threshold_elements_both=rows*(c+hidden))


def added_counts(shape, geom):
    t, b, h, w, c = shape
    factor = t*b*c
    q, r = geom["coarse"], geom["reps"]
    return dict(pool_input_elements=factor*q, pool_reduction_additions=factor*(q-r),
                pool_constant_divisions=factor*r, representative_elements=factor*r,
                repeat_dense_write_elements=factor*q, repeat_unique_source_elements=factor*r,
                common_assembly_write_elements=factor*h*w,
                residual_additions_unchanged=factor*h*w,
                local_contiguous_copy_upper_bound_elements=factor*h*w)


def aggregate(rows, field):
    keys = rows[0][field]
    return {k: sum(row["depth"]*row[field][k] for row in rows) for k in keys}


def audit_record(record, shape, hidden, source_ledger):
    require(record["shape_TBHWC"] == list(shape) and record["merge_stride"] == 3, "Record geometry")
    g = geometry(shape, record["fine_mask"])
    expected_ledger = source_ledger(shape, record["fine_mask"], 3, hidden)
    require(record["selected_service"] == expected_ledger, "Source ledger mismatch")
    selected = counts(shape, hidden, g["fine"]+g["reps"])
    dense = counts(shape, hidden, shape[2]*shape[3])
    require(expected_ledger["selected_fc_rows_per_layer"] == selected["fc_rows_each"], "Independent row count")
    for key, ourkey in (("cancelled_fc_weight_products_dense_equivalent", "fc_weighted_terms_both"),
                        ("cancelled_psn_temporal_macs_nominal", "psn_real_macs_both"),
                        ("cancelled_bn_elements", "bn_elements_both")):
        require(expected_ledger[key] == dense[ourkey]-selected[ourkey], "Independent service " + key)
    mode = record["mode"]
    require(mode in ("actual", "dual_reference"), "Bad mode")
    actual_g = g
    if mode == "dual_reference":
        h, w = shape[2:4]
        all_fine = geometry(shape, [[True]*((w+8)//9) for _ in range((h+8)//9)])
        all_coarse = geometry(shape, [[False]*((w+8)//9) for _ in range((h+8)//9)])
        actual_g = dict(fine=h*w, coarse=h*w, reps=all_coarse["reps"],
                        fine_shapes=all_fine["fine_shapes"], coarse_shapes=all_coarse["coarse_shapes"])
    t, b, _, _, c = shape
    for side, points in (("fine", actual_g["fine"]), ("coarse", actual_g["reps"])):
        rows = t*b*points
        expected = dict(native_mlp_calls=len(actual_g[side+"_shapes"]), fc1_rows=rows, fc2_rows=rows,
                        sn1_elements=rows*c, sn2_elements=rows*hidden,
                        bn1_elements=rows*hidden, bn2_elements=rows*c)
        require(record["observed"][side] == expected, "Observed hooks " + side)
        require(record["observed"]["native_input_shapes"][side] == actual_g[side+"_shapes"], "Native shapes " + side)
    work = added_counts(shape, actual_g)
    require(record["executed_fc_rows_per_layer"] == t*b*(actual_g["fine"]+actual_g["reps"]), "Executed rows")
    for key, ourkey in (("executed_pool_input_elements", "pool_input_elements"),
                        ("executed_pool_reduction_additions_logical", "pool_reduction_additions"),
                        ("executed_pool_divisions_logical", "pool_constant_divisions"),
                        ("executed_repeat_write_elements", "repeat_dense_write_elements")):
        require(record[key] == work[ourkey], "Executed added work " + key)
    require(record["external_residual_added_here"] is False and record["temporal_pooling"] is False,
            "Residual/time contract")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results", type=Path, default=HERE/"screen_v1/results.json")
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    for path, pin in ((MODULE, MODULE_SHA), (RUNNER, RUNNER_SHA), (STATIC, STATIC_SHA), (a.results, RESULT_SHA)):
        require(sha(path) == pin, "Frozen input SHA differs: " + str(path))
    constants = runner_constants()
    static = json.loads(STATIC.read_text())["configurations"][1]
    require(static["name"] == "full_480x640_window9_current_config", "Static configuration")
    base = static["cross_group_packed_dense_baseline_issue_steps"]
    ledger = extracted_source_ledger()
    stages = []
    shape_by_layer, hidden_by_layer = {}, {}
    for s in static["stages"]:
        index = s["stage"]
        shape = constants["STAGE_SHAPES"][index]
        t, b, h, w, c = shape
        depth = constants["DEPTHS"][index]
        require(s["feature_HWC"] == [h,w,c] and s["depth"] == depth, "Static/native shape")
        services = s["variants"][0]["baseline_MLP_services"]
        hidden = services["mlp.fc1"]["output_elements"]//(t*b*h*w)
        require(hidden == 4*c, "Audited native hidden width differs")
        all_coarse = [[False]*((w+8)//9) for _ in range((h+8)//9)]
        g = geometry(shape, all_coarse)
        fine, coarse = counts(shape, hidden, h*w), counts(shape, hidden, g["reps"])
        require(fine["fc_weighted_terms_both"] == services["mlp.fc1"]["scalar_products"]+services["mlp.fc2"]["scalar_products"], "Dense FC ledger")
        require(fine["psn_real_macs_both"] == sum(v["scalar_products"] for k,v in services.items() if k.endswith("temporal")), "Dense PSN ledger")
        residual = services["mlp.residual_add"]["static_issue_steps"]
        private = sum(v["static_issue_steps"] for k,v in services.items() if k != "mlp.residual_add")
        remain = Fraction(private*g["reps"], h*w)
        require(remain.denominator == 1, "Noninteger old template scaling")
        added = added_counts(shape, g)
        pool_illustration = Fraction(added["pool_reduction_additions"]+added["pool_constant_divisions"], 16)
        require(pool_illustration.denominator == 1, "Illustrative vector padding needed")
        stages.append(dict(stage=index, depth=depth, shape_TBHWC=list(shape), hidden=hidden,
                           dense_spatial_points=h*w, coarse_representatives=g["reps"],
                           cell_area_histogram=g["coarse_cell_area_histogram"], fine=fine, coarse=coarse,
                           added=added, template=dict(dense_private_issue=private, coarse_private_issue=int(remain),
                           residual_issue_unchanged=residual, cancelled_private_issue=private-int(remain),
                           illustrative_pool16_issue=int(pool_illustration))))
        for block in range(depth):
            name = "sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp".format(index,block)
            shape_by_layer[name], hidden_by_layer[name] = shape, hidden
    scopes = {}
    for name, members in (("stage0", (0,)), ("stage2", (2,)), ("stage0_plus_stage2_UNEXECUTED", (0,2)), ("all12", (0,1,2,3))):
        rows = [s for s in stages if s["stage"] in members]
        delta = sum(s["depth"]*s["template"]["cancelled_private_issue"] for s in rows)
        pool = sum(s["depth"]*s["template"]["illustrative_pool16_issue"] for s in rows)
        scopes[name] = dict(layers=sum(s["depth"] for s in rows), fine=aggregate(rows,"fine"),
                            coarse=aggregate(rows,"coarse"), added=aggregate(rows,"added"),
                            template_cancelled_private_issue=delta, template_whole_reduction_percent=100*delta/base,
                            template_10percent_extra_issue_allowance=float(Fraction(10*delta-base,10)),
                            illustrative_pool16_issue=pool,
                            illustrative_after_pool_whole_percent=100*(delta-pool)/base,
                            minimum_fraction_of_scope_saving_for_10percent_without_overhead=base/(10*delta),
                            minimum_fraction_under_uniform_illustrative_pool_scaling=base/(10*(delta-pool)))
    s2 = stages[2]
    loss_one_full_tile = s2["template"]["dense_private_issue"]//1200*(81-9)
    avoided_pool_one_tile = 10*384*81//16
    d2 = scopes["stage2"]["template_cancelled_private_issue"]
    p2 = scopes["stage2"]["illustrative_pool16_issue"]
    scopes["stage2"]["one_full_9x9_tile_returned_to_fine_in_one_of_six_layers"] = dict(
        private_issue_saving_lost=loss_one_full_tile,
        illustrative_pool_issue_avoided=avoided_pool_one_tile,
        pre_pool_whole_percent=100*(d2-loss_one_full_tile)/base,
        illustrative_after_pool_whole_percent=100*(d2-loss_one_full_tile-p2+avoided_pool_one_tile)/base,
        measured_or_synthesized=False)
    raw = json.loads(a.results.read_text())
    require(raw["status"] == "STRUCTURED_MERGE_DEV8_COMPLETE" and len(raw["frames"]) == 8 and
            raw["full_forward_count"] == 67 and raw["BN_buffers_unchanged"] and raw["original_model_unchanged"] and
            raw["original_modules_restored"], "Run incomplete or mutation")
    require(raw["binding"]["module_sha256"] == MODULE_SHA and raw["binding"]["runner_sha256"] == RUNNER_SHA,
            "Observed source binding differs")
    expected_arms = {arm: (kind, scope) for arm,kind,scope in constants["ARMS"]}
    audited_records = dual_records = all12_layer_checks = 0
    per_arm_rows = {}
    for frame in raw["frames"]:
        require(set(frame["arms"]) == set(expected_arms), "Missing/extra arm")
        for arm, (kind, scope) in expected_arms.items():
            execution = frame["arms"][arm]["execution"]
            require(set(execution["all12_observed_fc_rows"]) == set(shape_by_layer), "All12 row inventory")
            scoped_names = {n for n in shape_by_layer if any(".layers.{}.".format(i) in n for i in scope)}
            require(set(execution["layers"]) == scoped_names, "Wrapped layer inventory")
            for name, shape in shape_by_layer.items():
                rows = shape[0]*shape[1]*shape[2]*shape[3]
                if name in scoped_names:
                    record = execution["layers"][name]
                    if kind == "zero":
                        rows = 0
                        require(record["mode"] == "zero" and record["executed_fc_rows_per_layer"] == 0, "Zero service")
                    else:
                        require(record["mode"] == "actual", "Reference used as deploy arm")
                        require(all(v == (kind == "fine") for row in record["fine_mask"] for v in row), "Fixed scope mask")
                        audit_record(record, shape, hidden_by_layer[name], ledger)
                        audited_records += 1
                        rows = record["executed_fc_rows_per_layer"]
                require(execution["all12_observed_fc_rows"][name] == {"fc1_rows":rows,"fc2_rows":rows}, "Independent FC hooks")
                all12_layer_checks += 1
            total = sum(v["fc1_rows"] for v in execution["all12_observed_fc_rows"].values())
            require(total == execution["summed_fc1_rows"] == execution["summed_fc2_rows"], "Aggregate hooks")
            per_arm_rows.setdefault(arm, []).append(total)
    for arm, pair in raw["first_frame_pairs"].items():
        require(arm in ("coarse_stage0","coarse_stage2","coarse_all12"), "Unexpected reference")
        require(all(v["numerically_equal"] for v in pair["flow_comparisons"]), "Reference flow mismatch")
        for name, check in pair["layers"].items():
            require(check["input_sha_equal"] and check["branch_comparison"]["numerically_equal"], "Reference branch mismatch")
        for record_name in ("actual_execution","dual_execution"):
            for name, record in pair[record_name]["layers"].items():
                audit_record(record, shape_by_layer[name], hidden_by_layer[name], ledger)
                dual_records += 1
    require(set(raw["first_frame_pairs"]) == {"coarse_stage0","coarse_stage2","coarse_all12"}, "Reference scope incomplete")
    out = dict(schema="structured_merge_service_audit_v1", status="PASS", model_or_GPU_executed=False,
               source_sha256={str(p):sha(p) for p in (Path(__file__),MODULE,RUNNER,STATIC,a.results)},
               static_template=dict(whole_issue_denominator=base, calibrated_cycles=False,
                    caveat="All BN folded in old template; observed graph freezes MLP only. Not same-graph PE gain.",
                    pool_illustration="16 element lanes, add/constant division each II1, no fusion/latency/traffic; hypothetical only"),
               stages=stages, scopes=scopes,
               observed_audit=dict(frames=8, arms=8, model_forwards_in_source_run=67,
                    selected_records_checked=audited_records, reference_side_records_checked=dual_records,
                    independent_all12_layer_row_checks=all12_layer_checks,
                    per_arm_all12_fc_rows_each_min_max={k:[min(v),max(v)] for k,v in per_arm_rows.items()},
                    hooks_verify_elements=True, MAC_terms_derived_from_shapes=True,
                    BN_unchanged=raw["BN_buffers_unchanged"], quality_recomputed=False),
               claim="Exact logical counts and source-hook consistency; not measured PE cycles, physical traffic, energy or quality.")
    text = json.dumps(out,indent=2,sort_keys=True)+"\n"
    if a.output:
        require(a.output.resolve().parent == HERE, "Output must be this new audit directory")
        a.output.write_text(text)
    print(json.dumps({"status":"PASS", "selected_records":audited_records,
                     "reference_records":dual_records, "all12_layer_checks":all12_layer_checks,
                     "scopes":{k:v["template_whole_reduction_percent"] for k,v in scopes.items()}},sort_keys=True))


if __name__ == "__main__":
    main()
