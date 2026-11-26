"""Independent R33 scalar evaluation checks, synthetic arrays only."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import evaluate as e
from test_evaluate_synthetic import fixture


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    output = HERE / "EVALUATOR_INDEPENDENT_CPU_V1.json"
    assert not output.exists()
    pack, pred, model, frames, hw = fixture()
    out = e.evaluate_data(pack, pred, model, frames, hw)
    definitions = out["policy_definitions"]
    assert len(definitions) == 32
    # The implementation reports selected action probabilities. Independently
    # enumerate each outcome and every region; never reuse mixed_metrics or mc.
    checked = 0
    for policy, definition in definitions.items():
        for region in e.REGIONS:
            counts, sums, means = [], [], []
            expected_cost = []
            for i, row in enumerate(out["per_identity"]):
                detail = row["policies"][policy]
                probs = np.asarray(detail["action_probabilities"])
                assert probs.shape == (7,) and np.isclose(probs.sum(), 1)
                source_rows = [frames[i]["states"][s][region] for s in e.STATES]
                n = source_rows[0]["valid_pixels"]
                total = sum(p * r["error_sum"] for p, r in zip(probs, source_rows))
                reported = detail["regions"][region]
                assert reported["valid_pixels"] == n
                assert np.isclose(reported["error_sum"], total)
                assert (reported["AEE"] is None) if n == 0 else np.isclose(reported["AEE"], total / n)
                counts.append(n)
                sums.append(total)
                if n:
                    means.append(total / n)
                column_actions = [6, 0, 1, 2, 3, 4, 5]
                control = definition["operator_control"]
                expected_cost.append(4 * sum(p * hw["candidates"][a][control + "_affine_BN_ticks"]
                                             for p, a in zip(probs, column_actions)))
                checked += 1
            aggregate = out["quality_summaries"][policy]["overall"][region]
            assert aggregate["valid_pixels"] == sum(counts)
            assert np.isclose(aggregate["error_sum"], sum(sums))
            assert (aggregate["pixel_weighted_AEE"] is None) if not sum(counts) else np.isclose(aggregate["pixel_weighted_AEE"], sum(sums) / sum(counts))
            assert (aggregate["frame_mean_AEE"] is None) if not means else np.isclose(aggregate["frame_mean_AEE"], np.mean(means))
            cost = out["decision_summaries"][policy]["overall"]["operator_only_service"]
            assert np.isclose(cost["mean_four_conv_ticks"], np.mean(expected_cost))

    # A single realized action can be a bad tail while the expectation is safe.
    f = copy.deepcopy(frames[0])
    bad = {"all_valid": {0}, "B": {0, 1}, "L": {1, 2}}
    for state in e.STATES:
        for region in ("all_valid", "B", "L"):
            a = -1 if state == "background_Q4" else int(state.split("_")[-1])
            aee = 14.0 if a in bad[region] else 10.0
            f["states"][state][region] = dict(valid_pixels=400, AEE=aee, error_sum=400*aee)
    risk = e.uniform_tail_risk([f])
    for region, actions in bad.items():
        row = risk["frames"][0]["regions"][region]
        assert row["triggering_actions"] == sorted(actions)
        assert np.isclose(row["trigger_probability"], len(actions)/6)
        assert np.isclose(risk["summaries"]["overall"]["regions"][region]["expected_bad_frame_count"], len(actions)/6)
    assert risk["frames"][0]["any_primary_region_triggering_actions"] == [0, 1, 2]
    assert risk["frames"][0]["any_primary_region_trigger_probability"] == .5
    assert (14 + 5*10)/6 < 12
    assert (14 + 5*10)/6 - 10 < 1
    # Support gating is evaluated per frame/region, never inferred from averages.
    low = copy.deepcopy(f)
    for state in e.STATES:
        for region in ("all_valid", "B", "L"):
            r = low["states"][state][region]
            r["valid_pixels"] = 255
            r["error_sum"] = 255*r["AEE"]
    assert e.uniform_tail_risk([low])["frames"][0]["any_primary_region_trigger_probability"] == 0
    result = dict(schema="flow_r33_evaluator_independent_cpu_v1", status="PASS",
        synthetic_policy_region_identity_checks=checked, policies=32, regions=len(e.REGIONS),
        checks=["independent outcome-enumerated 32-policy regional metrics",
                "pixel-weighted versus nonempty-frame mean", "static/dynamic operator action mapping",
                "random policy tail probability and action-union risk", "per-region support tail gating"],
        source_sha256={"evaluate.py": sha(e.__file__), "test_evaluate_synthetic.py": sha(HERE.parent/"test_evaluate_synthetic.py"),
                       "independent_test": sha(__file__)},
        new_GT_read=False, new_quality_read=False, real_head_fit=False, GPU_used=False)
    output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"status": "PASS", "checks": checked, "policies": 32}))


if __name__ == "__main__":
    main()
