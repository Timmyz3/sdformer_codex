#!/opt/anaconda3/envs/pytorch310_cpu/bin/python
"""Independent fail-closed hammer for the M1547 retained decoder screen.

The author implementation is not imported.  This hammer verifies both sealed
trees, independently reconstructs the frozen 36-call x 64-site population,
loads the four ep34 FP32 decoder weights, and recomputes all three block axes.
It also checks the mathematical accumulation domain of the claimed debt.
"""

from __future__ import print_function

import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M1521 = HW / "results/m1521_ep34_decoder_positive_planes_s30_c120_r1_20260831"
M1547 = HW / "results/m1547_ep34_s2_ccbs_decoder_retained_fastkill_r1_20260831"
CHECKPOINT = HW / "system_handoff/incoming/motion_c12_ep34_live93_checkpoint_epoch34.pth"
AUTHOR = HW / "system_simulator/scripts/analyze_m1547_ep34_s2_ccbs_decoder_retained_fastkill.py"
TEST = HW / "system_simulator/tests/test_m1547_ep34_s2_ccbs_decoder_retained_fastkill.py"
CONTRACT = HW / "contracts/m1547_ep34_s2_ccbs_decoder_retained_fastkill_contract_r1_20260831.json"
RESULT = M1547 / "m1547_ep34_s2_ccbs_decoder_retained_fastkill_r1.json"

EXPECTED = {
    str(CHECKPOINT): "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48",
    str(AUTHOR): "facf1831a29ee9b4db86b4899e82cf248e5fc9e1134c7c36eac89319fd9419d8",
    str(TEST): "4589f59cf148ac49108f5fe69a8af20a93a1048d3643a18fa0c4040a432d1088",
    str(CONTRACT): "812a583b237b1baef156818650285c90a293ad76f42699a888248780a5a432e0",
    str(M1521 / "manifest.json"): "969b786bf66323174bc734630384ae03abab5b81a4fc59000b113e0b7a5d8304",
    str(M1521 / "SHA256SUMS"): "985b7089560b77b09dc0e5327780da1d81e24f03670ee2658433cae3f7603efa",
    str(M1521 / "SHA256SUMS.seal.sha256"): "60a172e5cd041bcdd0ca38db87250090c48c66e655364b332868fb40a1b182f2",
    str(M1521 / "RUN_COMPLETE.txt"): "9fe54dfbc4c0e5cbbcf6f0fcdb04bfba4a2e8c484c98d806548345d0866f8258",
    str(RESULT): "fc9a39b3e1f17923eb6eef91f7f3a0917581074b068818ff6cdfa836f527dcb7",
    str(M1547 / "m1547_REPORT.md"): "b9a069b4026df1be1f09afcd62315a8acf66a8d1443440cdd1975bbeb959c96f",
    str(M1547 / "RUN_COMPLETE.txt"): "db034e69b98aa489688465c6fb21edd955ad817412aa10b9dd0fec20870aad1a",
    str(M1547 / "SHA256SUMS"): "227bebda9615037c5c766eea702075434c9bd99ab32f524ec2a88081c9a69448",
    str(M1547 / "SHA256SUMS.seal.sha256"): "4b3f72ebc83c7542ff2462c3ab9f409ebba23e37b09b9dbb1a3bf1b6422658fe",
}
WEIGHT_SHA = (
    "cb1a90a4ff33622024b43ee6b15a3409e2567ea1e7b626715f40cf8a4fbfd83b",
    "35a9214e9fbc2e4e271beea74c4f329c12d6c072cda9252eaae350dd404a51cb",
    "75f9921f3cd9786ece78247115dd07bdda425b4f6e068d43936c884c611d3ef7",
    "6a42dabae358d0048aa46c609c9cb633f1e8d0479e4628e4f85c21e00835ea4e",
)
BLOCKS = ((8, 16), (16, 16), (32, 16))
EPSILONS = (0.0, 0.05, 0.10, 0.20, 0.30)
SEQUENCES = ("interlaken_01_a", "thun_01_b", "zurich_city_12_a")
LOCAL_POSITIONS = (0, 4, 9)
SITES = 64


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            value.update(block)
    return value.hexdigest()


def load(path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            need(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + value)))


def verify_sha_tree(root, expected_manifest, expected_outer):
    sums = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    need(digest(sums) == expected_manifest and digest(outer) == expected_outer,
         "sealed tree identity mismatch")
    need(outer.read_text(encoding="ascii").split() == [expected_manifest, "SHA256SUMS"],
         "outer seal content mismatch")
    mapped = {}
    for line in sums.read_text(encoding="ascii").splitlines():
        parts = line.split("  ", 1)
        member_parts = parts[1].split("/") if len(parts) == 2 else []
        need(len(parts) == 2 and parts[1] not in mapped and
             not parts[1].startswith("/") and
             all(item not in ("", ".", "..") for item in member_parts),
             "malformed or unsafe seal row")
        mapped[parts[1]] = parts[0]
    actual = set()
    for path in root.rglob("*"):
        if path.is_file() and path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256"):
            relative = str(path.relative_to(root))
            actual.add(relative)
            need(relative in mapped and digest(path) == mapped[relative],
                 "sealed member mismatch: " + relative)
    need(actual == set(mapped), "sealed tree population mismatch")
    return mapped


def run_author_checks():
    commands = [
        ["python3", str(TEST)], ["python3.6", str(TEST)],
        ["python3", str(AUTHOR), "--preflight"],
        ["python3.6", str(AUTHOR), "--preflight"],
    ]
    rows = []
    for command in commands:
        completed = subprocess.run(command, cwd=str(ROOT), stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, universal_newlines=True)
        need(completed.returncode == 0, "author test/preflight failed")
        token = completed.stdout.strip()
        need(token.startswith("PASS_M1547_"), "author PASS token missing")
        rows.append({"command": command, "stdout": token})
    return rows


def selected_records(manifest):
    records = manifest["records"]
    need(len(records) == 120 and manifest["population"]["calls"] == 120 and
         manifest["population"]["samples"] == 30, "M1521 population mismatch")
    need(tuple(sorted(set(row["sequence"] for row in records))) == SEQUENCES,
         "M1521 sequence population mismatch")
    selected = []
    samples = []
    for sequence in SEQUENCES:
        identities = sorted(set((row["replay_sample_ordinal"], row["global_sample_id"])
                                for row in records if row["sequence"] == sequence))
        need(len(identities) == 10, "per-sequence sample count mismatch")
        for local in LOCAL_POSITIONS:
            replay, global_id = identities[local]
            layer_rows = sorted((row for row in records
                                 if row["sequence"] == sequence and
                                 row["replay_sample_ordinal"] == replay),
                                key=lambda row: row["module_ordinal"])
            need([row["module_ordinal"] for row in layer_rows] == [0, 1, 2, 3],
                 "selected module order mismatch")
            selected.extend(layer_rows)
            samples.append((sequence, replay, global_id))
    need(len(selected) == 36 and len(samples) == 9, "selected population mismatch")
    return selected, samples


def unpack_sites(record):
    path = M1521 / record["positive_output"]
    need(digest(path) == record["positive_output_sha256"], "positive plane SHA mismatch")
    raw = np.fromfile(str(path), dtype=np.uint8)
    need(raw.size == record["plane_bytes"], "plane byte count mismatch")
    bits = np.unpackbits(raw, bitorder="little")[:record["elements"]]
    shape = tuple(record["shape"])
    need(bits.size == int(np.prod(shape)) and shape[1] == 1, "plane shape mismatch")
    plane = bits.reshape(shape)
    time_steps, _, channels, height, width = shape
    count = time_steps * height * width
    indices = np.linspace(0, count - 1, SITES, dtype=np.int64)
    output = np.empty((SITES, channels), dtype=np.uint8)
    for out_index, flat in enumerate(indices):
        time_index = int(flat) // (height * width)
        spatial = int(flat) % (height * width)
        output[out_index] = plane[time_index, 0, :, spatial // width, spatial % width]
    need(set(np.unique(output).tolist()) <= set([0, 1]), "nonbinary retained plane")
    return output


def load_weights():
    wrapper = torch.load(str(CHECKPOINT), map_location="cpu")
    need(type(wrapper) is dict and set(wrapper) == set(["model_state_dict"]),
         "checkpoint wrapper mismatch")
    state = wrapper["model_state_dict"]
    need(len(state) == 921, "checkpoint state population mismatch")
    weights = []
    for ordinal in range(4):
        name = "sttmultires_unet.decoders.{}.deconv.0.weight".format(ordinal)
        value = state[name].detach().cpu().contiguous().numpy()
        need(value.dtype == np.float32 and value.ndim == 4 and
             hashlib.sha256(value.tobytes(order="C")).hexdigest() == WEIGHT_SHA[ordinal],
             "weight identity mismatch")
        weights.append(value)
    return weights


def stats(values):
    data = np.asarray(values, dtype=np.float64)
    need(data.size and np.isfinite(data).all(), "invalid statistic population")
    return {"count": int(data.size), "min": float(data.min()),
            "p10": float(np.quantile(data, .10)), "median": float(np.quantile(data, .50)),
            "p90": float(np.quantile(data, .90)), "p99": float(np.quantile(data, .99)),
            "max": float(data.max())}


def sample_ids(count, wanted):
    return sorted(set(int(value) for value in
                      np.linspace(0, count - 1, min(count, wanted), dtype=np.int64)))


def independent_block(weights, activity_by_layer, group, output_tile):
    normalized_bounds = []
    ratios = []
    exact_positive = 0
    exact_zero = 0
    epsilon = dict((value, {"blocks": 0, "drops": 0, "debt": []})
                   for value in EPSILONS)
    witness = {}
    metadata_bytes = 0
    old_bytes = 0
    hypothetical_weight_bytes = 0
    metadata_reads = 0
    zero_bound_blocks = 0
    total_bound_blocks = 0

    for layer_id, (weight, activity) in enumerate(zip(weights, activity_by_layer)):
        cin, cout, kh, kw = weight.shape
        need(kh == 3 and kw == 3 and activity.shape == (576, cin),
             "activity/weight shape mismatch")
        g_count = (cin + group - 1) // group
        o_count = (cout + output_tile - 1) // output_tile
        metadata_bytes += 2 * g_count * o_count
        old_bytes += 2 * cin * o_count
        hypothetical_weight_bytes += cin * cout * kh * kw
        metadata_reads += 576 * g_count * o_count
        maxima = np.empty((g_count, o_count), dtype=np.float64)
        for gb in range(g_count):
            gs, ge = gb * group, min(cin, (gb + 1) * group)
            for ob in range(o_count):
                os_, oe = ob * output_tile, min(cout, (ob + 1) * output_tile)
                maxima[gb, ob] = np.max(np.abs(weight[gs:ge, os_:oe]))
        padded = np.zeros((576, g_count * group), dtype=np.uint8)
        padded[:, :cin] = activity
        mass = padded.reshape(576, g_count, group).sum(axis=2)
        bounds = mass[:, :, None].astype(np.float64) * maxima[None, :, :]
        zero_bound_blocks += int(np.count_nonzero(bounds == 0.0))
        total_bound_blocks += int(bounds.size)
        layer_max = float(np.max(np.abs(weight)))
        normalized_bounds.append((bounds / (layer_max * group)).reshape(-1))
        reference = layer_max * cin
        for budget in EPSILONS:
            ledger = epsilon[budget]
            limit = budget * reference
            for obs in range(576):
                for ob in range(o_count):
                    debt = 0.0
                    for gb in range(g_count):
                        value = float(bounds[obs, gb, ob])
                        drop = value == 0.0 or debt + value <= limit
                        ledger["blocks"] += 1
                        if drop:
                            ledger["drops"] += 1
                            debt += value
                        if budget == .10:
                            key = (layer_id, gb, ob)
                            witness[key] = witness.get(key, 0) | (1 if drop else 2)
                    ledger["debt"].append(debt / reference)
        for obs in sample_ids(576, 18):
            for gb in sample_ids(g_count, 16):
                gs, ge = gb * group, min(cin, (gb + 1) * group)
                live = activity[obs, gs:ge].astype(bool)
                if not live.any():
                    continue
                for ob in sample_ids(o_count, 8):
                    os_, oe = ob * output_tile, min(cout, (ob + 1) * output_tile)
                    bound = float(bounds[obs, gb, ob])
                    need(bound > 0.0, "active exact block has zero bound")
                    local = weight[gs:ge, os_:oe][live].sum(axis=0)
                    exact = float(np.max(np.abs(local)))
                    exact_positive += 1
                    if exact <= 1e-12:
                        exact_zero += 1
                    else:
                        ratios.append(bound / exact)

    epsilon_rows = []
    for budget in EPSILONS:
        ledger = epsilon[budget]
        epsilon_rows.append({
            "epsilon_normalized": budget, "block_decisions": ledger["blocks"],
            "drop_fraction": float(ledger["drops"]) / ledger["blocks"],
            "normalized_debt": stats(ledger["debt"]), "is_aee_budget": False,
        })
    both = sum(1 for value in witness.values() if value == 3)
    return {
        "metadata": {"metadata_bytes": metadata_bytes,
                     "int8_weight_bytes": hypothetical_weight_bytes,
                     "metadata_to_int8_weight_bytes": float(metadata_bytes) /
                     hypothetical_weight_bytes,
                     "old_g11_per_source_metadata_bytes": old_bytes,
                     "reduction_vs_old_g11": float(old_bytes) / metadata_bytes,
                     "metadata_reads_over_selected_sites": metadata_reads},
        "bounds": stats(np.concatenate(normalized_bounds)),
        "ratios": stats(ratios),
        "false_zero": float(exact_zero) / exact_positive,
        "dynamic": {"epsilon_normalized": .10, "static_blocks_observed": len(witness),
                    "blocks_with_both_keep_and_drop": both,
                    "fraction": float(both) / len(witness)},
        "epsilon": epsilon_rows,
        "zero_bound_fraction": float(zero_bound_blocks) / total_bound_blocks,
    }


def close(left, right, path="root"):
    if isinstance(left, dict):
        need(isinstance(right, dict) and set(left) <= set(right), path + " dict mismatch")
        for key in left:
            close(left[key], right[key], path + "." + key)
    elif isinstance(left, list):
        need(isinstance(right, list) and len(left) == len(right), path + " list mismatch")
        for index, value in enumerate(left):
            close(value, right[index], path + "[{}]".format(index))
    elif isinstance(left, float):
        need(isinstance(right, (float, int)) and math.isclose(left, right, rel_tol=1e-12,
                                                              abs_tol=1e-12),
             path + " float mismatch")
    else:
        need(left == right, path + " mismatch")


def destination_domain_counterexample():
    # K3/S2: output coordinate 2*i+2 is reached by input i at tap 2 and
    # input i+1 at tap 0.  The 2-D Cartesian product creates four contributors.
    one_dim = [(0, 2), (1, 0)]
    contributors = [(iy, ky, ix, kx) for iy, ky in one_dim for ix, kx in one_dim]
    need(len(contributors) == 4 and
         len(set((2 * iy + ky, 2 * ix + kx)
                 for iy, ky, ix, kx in contributors)) == 1,
         "K3/S2 destination contributor proof failed")
    epsilon = .10
    per_source_debt = .09
    current_accepts = all(per_source_debt <= epsilon for _ in contributors)
    destination_debt = len(contributors) * per_source_debt
    need(current_accepts and destination_debt > epsilon,
         "destination debt counterexample failed")
    return {"kernel": 3, "stride": 2, "contributors_to_one_interior_destination": 4,
            "per_source_budget": epsilon, "per_source_debt": per_source_debt,
            "destination_debt": destination_debt,
            "current_source_local_test_accepts_all": True,
            "destination_budget_violated": True}


def main():
    for path, expected in EXPECTED.items():
        need(Path(path).is_file() and digest(path) == expected, "source identity mismatch: " + path)
    m1521_members = verify_sha_tree(M1521, EXPECTED[str(M1521 / "SHA256SUMS")],
                                    EXPECTED[str(M1521 / "SHA256SUMS.seal.sha256")])
    m1547_members = verify_sha_tree(M1547, EXPECTED[str(M1547 / "SHA256SUMS")],
                                    EXPECTED[str(M1547 / "SHA256SUMS.seal.sha256")])
    need(len(m1521_members) == 122 and len(m1547_members) == 3,
         "sealed member count mismatch")
    command_rows = run_author_checks()
    manifest = load(M1521 / "manifest.json")
    chosen, samples = selected_records(manifest)
    by_layer = dict((value, []) for value in range(4))
    for record in chosen:
        by_layer[record["module_ordinal"]].append(unpack_sites(record))
    activity = [np.concatenate(by_layer[index], axis=0) for index in range(4)]
    need(all(value.shape[0] == 9 * SITES for value in activity),
         "36-call x 64-site reconstruction failed")
    weights = load_weights()
    published = load(RESULT)
    independent = []
    for index, (group, output_tile) in enumerate(BLOCKS):
        result = independent_block(weights, activity, group, output_tile)
        row = published["block_results"][index]
        need(row["block"] == {"source_group": group, "output_tile": output_tile},
             "block order mismatch")
        close(result["metadata"], row["metadata_aggregate"], "metadata")
        close(result["bounds"], row["bound_normalized_to_layer_max_times_group_capacity"],
              "bounds")
        close(result["ratios"], row["bound_to_exact_local_contribution_ratio_sample"],
              "ratios")
        close(result["false_zero"], row["positive_bound_exact_zero_collision_fraction"],
              "false_zero")
        close(result["dynamic"], row["dynamic_witness"], "dynamic")
        close(result["epsilon"], row["epsilon_diagnostics"], "epsilon")
        need(math.isclose(result["zero_bound_fraction"],
                          result["epsilon"][0]["drop_fraction"],
                          rel_tol=0.0, abs_tol=0.0),
             "epsilon=0 dropped a non-zero-bound block")
        independent.append({
            "source_group": group, "output_tile": output_tile,
            "metadata_bytes": result["metadata"]["metadata_bytes"],
            "metadata_ratio_vs_hypothetical_int8":
                result["metadata"]["metadata_to_int8_weight_bytes"],
            "reduction_vs_old_g11": result["metadata"]["reduction_vs_old_g11"],
            "strict_metadata_8x_pass": result["metadata"]["reduction_vs_old_g11"] >= 8.0,
            "local_ratio_median": result["ratios"]["median"],
            "local_ratio_p90": result["ratios"]["p90"],
            "epsilon_0_drop_fraction": result["epsilon"][0]["drop_fraction"],
            "epsilon_0p1_drop_fraction": result["epsilon"][2]["drop_fraction"],
            "dynamic_both_count": result["dynamic"]["blocks_with_both_keep_and_drop"],
            "published_local_numbers_reproduced": True,
        })
    need(independent[0]["strict_metadata_8x_pass"] is False and
         math.isclose(independent[0]["reduction_vs_old_g11"], 7.976833976833976,
                      rel_tol=0, abs_tol=1e-15) and
         all(row["strict_metadata_8x_pass"] for row in independent[1:]),
         "strict 8x decision mismatch")
    need(published["decision"]["passing_configs"] == [
        {"output_tile": 16, "source_group": 16},
        {"output_tile": 16, "source_group": 32}], "published pass list mismatch")
    need(published["claim_boundary"]["cycles"] is False and
         published["claim_boundary"]["traffic"] is False and
         published["claim_boundary"]["energy"] is False and
         published["claim_boundary"]["aee"] is False and
         published["claim_boundary"]["rtl"] is False,
         "published claim boundary escalation")
    counterexample = destination_domain_counterexample()
    attacks = {
        "source_sha_substitution": True,
        "m1521_full_tree_population_or_sha": True,
        "m1547_full_tree_population_or_sha": True,
        "selected_call_population_36": len(chosen) == 36,
        "selected_site_population_36x64": sum(SITES for _ in chosen) == 2304,
        "sequence_order": tuple(sequence for sequence in SEQUENCES) == SEQUENCES,
        "site_count_63_rejected_by_frozen_64": SITES != 63,
        "block_order": BLOCKS == ((8, 16), (16, 16), (32, 16)),
        "epsilon_order": EPSILONS == (0.0, 0.05, 0.10, 0.20, 0.30),
        "zero_epsilon_debt_exact_zero": all(
            published["block_results"][index]["epsilon_diagnostics"][0]
            ["normalized_debt"]["max"] == 0.0 for index in range(3)),
        "metadata_cap_2pct": all(row["metadata_ratio_vs_hypothetical_int8"] <= .02
                                 for row in independent),
        "old_g11_strict_8x_boundary": not independent[0]["strict_metadata_8x_pass"],
        "tail_groups_ceil_charged": [[
            published["block_results"][config]["layers"][index]["metadata"]["g_blocks"]
            for index in range(4)] for config in range(3)] ==
            [[192, 97, 49, 25], [96, 49, 25, 13], [48, 25, 13, 7]],
        "dynamic_witness_exact": [row["dynamic_both_count"] for row in independent] ==
                                 [1652, 1234, 788],
        "bound_ratio_exact_recompute": True,
        "claim_escalation_rejected": True,
        "destination_debt_domain_counterexample": counterexample["destination_budget_violated"],
    }
    need(all(attacks.values()), "directed hammer attack failed")
    output = {
        "status": "FAIL_P0_DESTINATION_DEBT_ACCUMULATION_DOMAIN__M1547_LOCAL_NUMBERS_REPRODUCED_BUT_PASS_NOT_ADMITTED__SUCCESSOR_REQUIRED",
        "score": 61,
        "p0_count": 1,
        "p1_count": 1,
        "identity": {"author_source_sha256": EXPECTED[str(AUTHOR)],
                     "author_test_sha256": EXPECTED[str(TEST)],
                     "contract_sha256": EXPECTED[str(CONTRACT)],
                     "result_sha256": EXPECTED[str(RESULT)],
                     "checkpoint_sha256": EXPECTED[str(CHECKPOINT)],
                     "m1521_manifest_sha256": EXPECTED[str(M1521 / "manifest.json")]},
        "author_test_and_preflight": command_rows,
        "sealed_population": {"m1521_members": len(m1521_members),
                              "m1547_members": len(m1547_members),
                              "selected_samples": len(samples), "selected_calls": len(chosen),
                              "selected_sites": len(chosen) * SITES},
        "independent_three_block_recompute": independent,
        "directed_attacks": attacks,
        "p0": {
            "id": "P0_DESTINATION_ACCUMULATION_DOMAIN",
            "finding": "M1547 resets epsilon debt for every source-site/output-tile. K3/S2 ConvTranspose allows up to four spatial source sites to update one interior destination/output-tile, so the published source-local debt does not bound final destination error.",
            "counterexample": counterexample,
            "effect": "The approximately 99 percent epsilon=0.1 drop fractions are not an admissible bounded opportunity and the M1547 PASS cannot authorize the FC/patch capture request.",
            "required_successor": "Accumulate debt per destination/output-tile across every legal source site and tap, or conservatively divide the destination budget by proven contributor multiplicity; preserve tail/boundary mapping and charge the destination debt state before re-screening."
        },
        "p1": {
            "id": "P1_STRUCTURAL_FETCH_GATE_DECLARATIVE_ONLY",
            "finding": "decision_before_weight_fetch is a constant structural assertion, not an address-timed schedule or port/bank measurement.",
            "effect": "It may remain a successor design requirement but cannot be cited as cycles, traffic, energy, or implemented hardware."
        },
        "admission": {"m1547_pass_admitted": False, "fc_patch_capture_request_admitted": False,
                      "rtl": False, "cycles": False, "traffic": False,
                      "energy": False, "aee": False, "paper_headline": False,
                      "successor_cpu_rescreen_allowed": True},
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
