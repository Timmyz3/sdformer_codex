"""Fixed diagnostic16 / remaining48 views of complete E0 recovery raw metrics.

No new GT read, no new model call, no changes to the original reducer.
All64 stays primary; both subsets are exposed training-domain development.
"""
import argparse
import json
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reduce_e1_quality_v1 as base


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quality", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    d = base.read(a.quality)
    assert d["schema"] == "flow_e0_independent_raw_quality_v1" and d["status"] == "COMPLETE" and d["kind"] == "recovery"
    assert d["execution_review"]["status"] == "PASS" and d["execution_review"]["quality_analysis_allowed"]
    assert d["execution_review"]["counts"] == dict(optimizer_updates=512, quality_predictions=320, formal_model_forwards=1353)
    assert len(d["frames"]) == 64 and len({f["target_id"] for f in d["frames"]}) == 64
    roles_path = HERE.parent/"selections_v1.json"
    roles = base.read(roles_path)
    assert base.sha(roles_path) == "60b7d6d525d4130f5f0efecaff95511808b2186c3aaba9e9f37a957db32ef31f"
    original = set(roles["localization_ids"])
    assert len(original) == 16 and original <= {f["target_id"] for f in d["frames"]}
    subsets = {
        "all64_primary": d["frames"],
        "diagnostic16_already_examined": [f for f in d["frames"] if f["target_id"] in original],
        "remaining48_development": [f for f in d["frames"] if f["target_id"] not in original],
    }
    reports = {}
    pairs = [(v["candidate"], v["reference"]) for v in d["comparisons"].values()]
    pairs += [("endpoint_Q4", "endpoint_Q8")]
    for name, frames in subsets.items():
        assert len(frames) == {"all64_primary": 64, "diagnostic16_already_examined": 16, "remaining48_development": 48}[name]
        seqs = list(dict.fromkeys(f["sequence"] for f in frames))
        assert len(seqs) == 8
        summaries = {s: {scope: {r: base.mathcore.aggregate([f["states"][s][r] for f in frames if scope == "overall" or f["sequence"] == scope])
                                for r in base.mathcore.REGIONS} for scope in ["overall"] + seqs} for s in d["states"]}
        comparisons = {c+"__vs__"+r: base.mathcore.comparison(frames,summaries,c,r,seqs) for c,r in pairs}
        reports[name] = dict(target_ids=[f["target_id"] for f in frames], count=len(frames),
                             summaries=summaries, comparisons=comparisons)
    out = dict(schema="flow_e0_predeclared_subset_readout_v1", status="COMPLETE",
               source_quality_sha256=base.sha(a.quality), selections_sha256=base.sha(roles_path),
               raw_metrics_recomputed=False, additional_GT_read=False, model_forwards=0,
               subset_selection="Fixed localization16 and its complement in complete64, frozen before formal outcome.",
               primary="all64_primary", independent_confirmation=False,
               scope="All subsets are exposed training-domain development; remaining48 is not new official or unseen-scene test.",
               reports=reports, script_sha256=base.sha(__file__))
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status="COMPLETE", output=str(a.output),
                         counts={k:v["count"] for k,v in reports.items()}), indent=2))


if __name__ == "__main__":
    main()
