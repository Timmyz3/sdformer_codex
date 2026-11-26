#!/usr/bin/env python3
"""R10 preparation/default preflight; 366 explicit full forwards only when launched.

Reuse a private frozen R6 collector/probe. Only intervention location changes
from stage2/block0 to block2; new own checks are mandatory, old receipts are
inherited data/numerics evidence only. No old source file is edited.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import signal
import traceback
from source_contract import (HERE, TARGET, OLD_TARGET, PREFIX_NAMES, load_source,
                             require, sha, source_path, verify_all)
from select_members import verify_membership

BUDGET = 366
STATUS = "ROUND10_BLOCK2_SINGLETON_COLLECTION_COMPLETE"


def expected_rows(shapes, selected, mode):
    rows = 0
    for name, shape in shapes.items():
        t, b, h, w, c = shape
        if not any(".layers.{}.".format(s) in name for s in (0, 2)):
            rows += t * b * h * w
            continue
        for y in range(0, h, 9):
            for x in range(0, w, 9):
                hh, ww = min(9, h-y), min(9, w-x)
                representatives = math.ceil(hh/3) * math.ceil(ww/3)
                fine = name == TARGET and (y//9, x//9) in selected
                points = hh*ww+representatives if mode == "dual_reference" else (hh*ww if fine else representatives)
                rows += t*b*points
    return rows


def audited_forward(raw_forward):
    baseline = {}

    def forward(model, functional, targets, shapes, chunk, selected, merge, torch,
                mode="actual", capture=False, capture_u=False, expected_u_sha=None,
                capture_tiles=(), capture_bn=False, deadline=lambda: None):
        require(TARGET in targets and all(name in targets for name in PREFIX_NAMES), "Block2 prefix targets missing")
        out = raw_forward(model, functional, targets, shapes, chunk, selected, merge, torch,
                          mode=mode, capture=capture, capture_u=capture_u, expected_u_sha=expected_u_sha,
                          capture_tiles=capture_tiles, capture_bn=capture_bn, deadline=deadline)
        flows, execution, branches, captured = out
        layers = execution["layers"]
        expected_names = {name for name in targets if any(".layers.{}.".format(s) in name for s in (0, 2))}
        require(set(layers) == expected_names and len(layers) == 8, "All eight stage02 wrappers required")
        for name, layer in layers.items():
            h, w = shapes[name][2:4]
            mask = [[name == TARGET and (y, x) in selected for x in range(math.ceil(w/9))]
                    for y in range(math.ceil(h/9))]
            require(layer["fine_mask"] == mask, "Incorrect block2 mask or tail: " + name)
        if mode == "actual" and not selected and expected_u_sha is None:
            baseline.clear()
            baseline.update({name: {key: layers[name][key] for key in ("input_sha256", "output_sha256")}
                             for name in PREFIX_NAMES})
        require(set(baseline) == set(PREFIX_NAMES), "A current-frame CC baseline must precede interventions")
        require(all(layers[name]["input_sha256"] == baseline[name]["input_sha256"] for name in PREFIX_NAMES),
                "Block2 or earlier prefix input changed")
        require(all(layers[name]["output_sha256"] == baseline[name]["output_sha256"] for name in PREFIX_NAMES[:-1]),
                "Block2 earlier prefix output changed")
        require(execution["summed_fc1_rows"] == execution["summed_fc2_rows"] == expected_rows(shapes, selected, mode),
                "Independent full12 row service differs")
        execution["r10_target_layer"] = TARGET
        execution["r10_prefix_layer_names_checked"] = list(PREFIX_NAMES)
        execution["r10_current_CC_prefix_input_exact"] = True
        execution["r10_earlier_prefix_output_exact"] = True
        return flows, execution, branches, captured
    return forward


def load_runtime():
    collector = load_source("collector", "_round10_private_frozen_collector")
    probe = collector.load_probe()
    require(probe.TARGET == OLD_TARGET, "Unexpected inherited target")
    probe.TARGET = TARGET
    probe.forward_probe = audited_forward(probe.forward_probe)
    parent = probe.load_scope_runner()
    original_write = parent.write
    def write(path, value):
        if value.get("schema") == "flow_round10_block2_singletons_v1" and "BN_scope" in value:
            value["BN_scope"]["execution"] = (
                "Eight stage02 tiled MLP wrappers; target stage2/block2; all-coarse baseline "
                "or one complete 9x9 target tile fine, all tails coarse; actual/dual service distinguished")
        original_write(path, value)
    parent.write = write
    return collector, probe, parent


def prepare(a, collector, probe, parent):
    require(a.group == "all" and 0 < a.wall_minutes <= 30, "Fixed all28/30-minute maximum contract")
    inherited_pins = verify_all()
    member = verify_membership()
    require(sha(a.selection_manifest) == a.selection_manifest_sha256, "Compatibility manifest SHA differs")
    compat = json.loads(a.selection_manifest.read_text())
    require(compat.get("compatibility_only") is True and compat["groups"] == member["groups"]
            and compat["r10_membership_sha256"] == sha(HERE/"membership_v1.json"), "R10 membership link differs")
    own_check = parent.checked_cpu_receipt(a.runner_checks,
        [Path(__file__), HERE/"test_block2_cpu.py", HERE/"source_contract.py", HERE/"select_members.py"])
    inherited = argparse.Namespace(**vars(a))
    inherited.runner_checks = a.collector_runner_checks
    prepared = list(collector.prepare(inherited, probe, parent))
    binding = dict(prepared[7])
    require(binding["target"] == TARGET and binding["full_forward_budget"] == BUDGET, "New target/budget differs")
    binding.update(schema="flow_round10_block2_collection_binding_v1", runner_sha256=sha(__file__),
                   runner_checks_sha256=own_check, inherited_collector_source_sha256=inherited_pins["collector"],
                   inherited_collector_receipt_sha256=sha(a.collector_runner_checks),
                   inherited_source_pins_sha256=sha(HERE/"source_pins.json"),
                   r10_membership_sha256=sha(HERE/"membership_v1.json"),
                   old_target=OLD_TARGET, target=TARGET, prefix_layer_names=list(PREFIX_NAMES),
                   mechanism_change="Only target block0->block2; recollect current CC-prefix u, actual singleton outputs and branch MSE",
                   primary_analysis_candidate="flow11_task", full_forward_budget=BUDGET,
                   scope="Reused fit20 and new identities in exposed train-side sequences. No backbone/branch optimization; later fixed CPU predictor fitting only. Not native/FF quality evidence or hardware benefit.")
    prepared[7] = binding
    return tuple(prepared)


def main(argv=None):
    collector, probe, parent = load_runtime()
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parent-contract", "identity-bank", "prior-screen-results", "prior-scope-results",
                 "selection-manifest", "fit-membership", "prior-check-membership"):
        parser.add_argument("--"+name, type=Path, required=True)
        parser.add_argument("--"+name+"-sha256", required=True)
    parser.add_argument("--group", choices=("all",), default="all")
    parser.add_argument("--module-checks", type=Path, default=probe.BASE/"cpu_results.json")
    parser.add_argument("--scope-runner-checks", type=Path, default=probe.BASE/"scope_extension_v1/runner_cpu_results.json")
    parser.add_argument("--probe-runner-checks", type=Path, default=probe.HERE/"runner_cpu_results.json")
    parser.add_argument("--collector-runner-checks", type=Path, default=source_path("collector_receipt"))
    parser.add_argument("--runner-checks", type=Path, default=HERE/"collector_cpu_results.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path)
    parser.add_argument("--wall-minutes", type=float, default=30)
    parser.add_argument("--execute", action="store_true")
    a = parser.parse_args(argv)
    prepared = prepare(a, collector, probe, parent)
    result = {"schema": "flow_round10_block2_singletons_v1", "status": "PREFLIGHT_PASS_NO_INFERENCE",
              "binding": prepared[7], "native_preflight": prepared[2][-1],
              "inference_executed": False, "training_performed": False}
    if not a.execute:
        require(a.preflight_output and not a.preflight_output.exists(), "Fresh preflight output required")
        parent.write(a.preflight_output, result)
        print(result["status"])
        return
    require(parent.GPU_LOCK.parent.is_dir(), "Shared GPU lock directory required")
    with parent.GPU_LOCK.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        a.output_dir.mkdir(parents=True, exist_ok=False)
        previous = signal.getsignal(signal.SIGALRM)
        def timeout(signum, frame):
            raise TimeoutError("R10 fixed wall budget exhausted")
        signal.signal(signal.SIGALRM, timeout)
        signal.setitimer(signal.ITIMER_REAL, a.wall_minutes*60)
        try:
            result["inference_executed"] = True
            collector.execute(a, prepared, result, probe, parent)
            require(result["full_forward_count"] == BUDGET and result["completed_targets"] == 28,
                    "R10 complete count differs")
            result.update(status=STATUS, inherited_completion_status="ROUND6_SINGLETON_COLLECTION_COMPLETE",
                          scope=prepared[7]["scope"])
        except BaseException as exc:
            result.update(status="INCOMPLETE_NO_SCIENTIFIC_CONCLUSION",
                          error=type(exc).__name__+": "+str(exc), traceback=traceback.format_exc())
            raise
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
            parent.write(a.output_dir/"results.json", result)


if __name__ == "__main__":
    main()
