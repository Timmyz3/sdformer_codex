"""Summarize this round's actual receipts without treating plans as completions."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SPECS = [
    ("E1", "e1_v1", 80, 4, 0,
     "independent_review/E1_ACTUAL_RAW_QUALITY_V1.json"),
    ("E0_diagnostic", "e0_diagnostic_v1", 64, 4, 0,
     "independent_review/E0_DIAGNOSTIC_RAW_QUALITY_V1.json"),
    ("E0_smoke", "e0_smoke_v1", 0, 5, 1,
     "independent_review/E0_SMOKE_ACTUAL_AUDIT_V1.json"),
    ("E0_formal", "e0_recovery_v1", 320, 9, 512,
     "independent_review/E0_FORMAL_ACTUAL_AUDIT_V1.json"),
    ("E3", "e3_rne_probe_v1", 112, 6, 0,
     "independent_review/E3_ACTUAL_AUDIT_V1.json"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


rows = []
for name, folder, quality, engineering, updates, audit in SPECS:
    result = ROOT/"actual"/folder/"results.json"
    row = dict(
        experiment=name, result_file=str(result.relative_to(ROOT)),
        planned_quality_predictions=quality,
        planned_engineering_forwards=engineering,
        planned_optimizer_updates=updates,
    )
    if result.exists():
        data = json.loads(result.read_text())
        row.update(
            result_sha256=digest(result), status=data["status"],
            actual_model_forwards=data.get("model_forwards"),
            actual_optimizer_updates=data.get("completed_optimizer_steps", 0),
            process_internal_wall_seconds=data.get("wall_seconds"),
            max_allocated_bytes=data.get("max_allocated_bytes"),
            max_reserved_bytes=data.get("max_reserved_bytes"),
        )
        review = ROOT/audit
        if review.exists():
            row.update(independent_review=audit, review_sha256=digest(review),
                       review_status=json.loads(review.read_text()).get("status"))
    else:
        row["status"] = "NO_LOCAL_COMPLETE_RESULT"
    rows.append(row)
rows.append(dict(experiment="E2_legacy_drop4", status="DEFERRED_NOT_EXECUTED",
                 actual_model_forwards=0, actual_optimizer_updates=0,
                 reason="Ordinary RNE control invalidated the old large-damage premise."))
out = dict(
    schema="flow_20261021_actual_run_registry_v1", experiments=rows,
    summed_recorded_process_seconds=sum(
        r.get("process_internal_wall_seconds") or 0 for r in rows),
    recorded_model_forwards=sum(r.get("actual_model_forwards") or 0 for r in rows),
    wall_time_scope="Sum of process-internal recorded wall time, not rental billing "
                    "time, CUDA kernel time or deployment latency.",
    hardware_results=[
        "e4_hardware/README.md", "e4_hardware/e2_patch6_v1/README.md",
        "e4_hardware/rne4_supply_v1/README.md",
        "e4_hardware/rne4_macro6_v1/README.md",
    ],
    dataset_scope="Previously exposed DSEC training-domain development; "
                  "not official DSEC test or completed baseline reproduction.",
    generator_sha256=digest(Path(__file__)),
)
cpu_result = ROOT/"e3_response_diagnostic_v1/RESULTS_V1.json"
cpu_review = ROOT/"independent_review/E3_RESPONSE_INDEPENDENT_AUDIT_V1.json"
if cpu_result.exists():
    cpu = json.loads(cpu_result.read_text())
    out["CPU_diagnostics"] = [
        dict(experiment="E3_response_representability",
             result_file=str(cpu_result.relative_to(ROOT)),
             result_sha256=digest(cpu_result), status=cpu["status"],
             model_forwards=cpu["model_forwards"], GPU_used=cpu["GPU_used"],
             head_fitting=cpu["head_fitting"],
             independent_review=str(cpu_review.relative_to(ROOT))
             if cpu_review.exists() else None,
             independent_review_sha256=digest(cpu_review)
             if cpu_review.exists() else None)
    ]
path = ROOT/"RUN_REGISTRY_V1.json"
path.write_text(json.dumps(out, indent=2, ensure_ascii=False)+"\n")
print(json.dumps({"output": str(path), "runs": len(rows),
                  "recorded_model_forwards": out["recorded_model_forwards"],
                  "summed_recorded_process_seconds":
                  out["summed_recorded_process_seconds"]}, indent=2))
