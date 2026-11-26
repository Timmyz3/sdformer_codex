"""Bind actual E3 masks/counts to frozen macro6 cost geometry; no new timing."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return json.loads((ROOT/rel).read_text())


def sha(rel):
    return hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()


prepared = read("e3_rne_probe/PREPARED_CONTRACT_V1.json")
bound = read("e3_rne_probe/ENDPOINT_CONTRACT_V1.json")
actual = read("actual/e3_rne_probe_v1/results.json")
cost = read("e4_hardware/rne4_macro6_v1/results/SUPPLY_REPLAY_V1.json")
sources = read("e4_hardware/rne4_macro6_v1/SOURCE_BINDINGS_V1.json")
assert sources["flow_story_20261021/e3_rne_probe/PREPARED_CONTRACT_V1.json"] == sha(
    "e3_rne_probe/PREPARED_CONTRACT_V1.json")
assert {k: v for k, v in prepared.items() if k not in ("status", "endpoint")} == {
    k: v for k, v in bound.items() if k not in ("status", "endpoint")}
assert actual["action_contract"] == bound
assert actual["status"] == "E3_RNE112_AND6_DENSE_REFERENCES_COMPLETE"
records = []
frame_target_checks = 0
for action in range(-1, 6):
    label = "background_Q4" if action < 0 else f"upgrade_{action}"
    evaluation = actual["evaluations"][label]
    costs = [x for x in cost["results"] if x["selected_macro"] == action]
    assert len(costs) == (1 if action < 0 else 2)
    for name in bound["family_targets"]:
        g = bound["geometries"][name]
        assert g == dict(T=10, H=240, W=320, K=864, Cout=96)
        mask = np.zeros((240, 320), dtype=np.bool_)
        if action >= 0:
            for y, x in bound["groups"][action]["tiles"][name]:
                mask[y*9:(y+1)*9, x*9:(x+1)*9] = True
        assert hashlib.sha256(mask.tobytes()).hexdigest() == evaluation[
            "mask_layout"]["mask_sha256"][name]
        q8 = 30900 + int(mask.sum())*10
        q4 = 768000-q8
        beats = (8*q8+4*q4)*7*6
        for c in costs:
            assert (c["q8_rows"], c["q4_rows"], c["arithmetic_accepted_beats"]) == (
                q8, q4, beats)
        for frame in evaluation["frames"]:
            work = frame["work"][name]
            assert (work["high8_rows"], work["low4_rows"]) == (q8, q4)
            frame_target_checks += 1
    records.append(dict(action=action, matched_cost_arms=len(costs),
                        per_conv_q8_rows=q8, per_conv_q4_rows=q4,
                        per_conv_accepted_beats=beats))
out = dict(
    schema="flow_e3_actual_macro6_hardware_binding_root_v1", status="PASS",
    source_result_sha256=sha("actual/e3_rne_probe_v1/results.json"),
    cost_result_sha256=sha("e4_hardware/rne4_macro6_v1/results/SUPPLY_REPLAY_V1.json"),
    endpoint_contract_sha256=sha("e3_rne_probe/ENDPOINT_CONTRACT_V1.json"),
    independently_rebuilt_spatial_masks=28,
    actual_frame_conv_domain_checks=frame_target_checks,
    matched_cost_arms=13, records=records,
    scope="Actual action geometry, masks and accepted arithmetic counts match "
          "the frozen finite cost model. No new cycle simulation, endpoint "
          "operand RTL bridge, FP reduction equivalence or network timing.",
)
dest = ROOT/"independent_review/E3_HARDWARE_BINDING_ROOT_V1.json"
dest.write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps({"status": out["status"], "frame_conv_checks": frame_target_checks,
                  "matched_cost_arms": 13}))
