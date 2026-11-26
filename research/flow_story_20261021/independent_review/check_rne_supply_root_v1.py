"""Independent count reconstruction for completed RNE supply artifacts; no GPU."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "e4_hardware/rne4_supply_v1"
MASK = ROOT.parent / "flow_story_20261020/round30/hardware/MASK_CONTRACT_V1.json"
GEOMETRY = {
    "patch.res0.conv0": (240, 320),
    "stage2.fc1": (30, 40),
    "stage2.fc2": (30, 40),
    "stage0.fc1_K96": (120, 160),
}

def main():
    freeze = json.loads((BASE / "FREEZE_V1.json").read_text())
    mapping = freeze.get("files_sha256", freeze.get("files"))
    if mapping is None:
        raise ValueError("Inspect actual freeze schema before use")
    repository = ROOT.parents[1]
    for path, expected in mapping.items():
        p = Path(path)
        if not p.is_absolute():
            p = repository / p if path.startswith("research/") else BASE / p
        assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, p
    mask = json.loads(MASK.read_text())
    rows = json.loads((BASE / "results/SUPPLY_REPLAY_V1.json").read_text())["results"]
    checked = []
    for row in rows:
        h, w = GEOMETRY[row["name"]]
        keys = [k for k, g in mask["geometries"]["full9"].items() if (g["H"], g["W"]) == (h, w)]
        sets = [set(map(tuple, mask["evaluation_tiles"][k])) for k in keys]
        assert all(s == sets[0] for s in sets)
        selected = sets[0]
        arm = row["arm"]
        parallel = arm == "Q8_parallel_single"
        q8_only = arm in ("Q8_parallel_single", "Q8_serial_single")
        pure4 = arm == "Q4_single_allrows_COMPONENT_ONLY"
        mixed = arm == "M_fixed_RNE4_Q8_dual"
        bits = []
        for r in range(10*h*w):
            y, x = divmod(r % (h*w), w)
            tail = y >= h//9*9 or x >= w//9*9
            bits.append(8 if q8_only or (not pure4 and (tail or (mixed and (y//9, x//9) in selected))) else 4)
        q8 = bits.count(8)
        q4 = len(bits)-q8
        chunks = math.ceil(row["K"] / (16 if parallel else 128))
        ob = row["Cout"]//16
        beats = len(bits)*chunks*ob if parallel else sum(bits)*chunks*ob
        assert row["q8_rows"] == q8 and row["q4_rows"] == q4
        assert row["arithmetic_accepted_beats"] == beats
        assert row["weight_RF_read_bytes"] == beats*256
        assert row["activation_RF_read_bytes"] == beats*(4 if parallel else 16)
        assert row["scale_RF_read_bytes"] == len(bits)*ob*64
        assert row["output_store_ticks"] == len(bits)*ob
        assert row["scale_vector_ticks"] == len(bits)*ob*2
        padded = chunks*(16 if parallel else 128)
        high = 0 if pure4 else padded*row["Cout"]+4*row["Cout"]
        low = 0 if q8_only else padded*row["Cout"]//2+4*row["Cout"]
        assert high == row["Q8_weight_scale_storage_bytes"]
        assert low == row["RNE4_weight_scale_storage_bytes"]
        assert high+low <= 1048576
        grouped = row["row_schedule"] == "batch_local_two_variant_partition"
        refs = []
        for first in range(0, len(bits), row["batch_rows"]):
            current = bits[first:first+row["batch_rows"]]
            kinds = sorted(set(current)) if grouped else [8] if q8_only else [4] if pure4 else [4, 8]
            refs.extend(kinds * ob)
        weight_bytes = sum(chunks*(1 if parallel else b)*256 for b in refs)
        assert row["weight_SRAM_read_bytes"] == weight_bytes
        assert row["weight_RF_prefill_busy_ticks"] == weight_bytes//256
        assert row["scale_SRAM_read_bytes"] == len(refs)*64
        assert row["scale_RF_refill_ticks"] == len(refs)
        assert row["row_index_write_bytes"] == (len(bits)*4 if grouped else 0)
        assert row["row_index_read_bytes"] == (len(bits)*ob*4 if grouped else 0)
        for trace in row["accepted_beat_trace"]:
            assert trace["bits"] == bits[trace["row"]]
            assert trace["weight_RF_address"] % 256 == 0
            assert trace["weight_RF_address"]+256 <= row["peak_weight_RF_bytes"]
        checked.append(dict(name=row["name"], arm=arm, beats=beats,
                            high_bytes=high, low_bytes=low, RF_refills=len(refs),
                            grouped=grouped))
    out = dict(
        schema="flow_rne_supply_root_review_v1",
        status="PASS_INDEPENDENT_COUNTS_BOUNDED_OPERATOR_MODEL",
        freeze_sha256=hashlib.sha256((BASE/"FREEZE_V1.json").read_bytes()).hexdigest(),
        rows=checked, full_network=False, numerical_BN_verified=False,
        new_RTL_or_DC=False,
        limits=[
            "Independent shape/mask/representation/refill/index count reconstruction; not a second cycle simulator.",
            "Floating conversion/scale/BN service is modeled; actual canonical FP reduction parity remains unproved.",
            "Four representative operators only; stage3 resource overflow is not solved.",
            "Pure Q4 without Q8 tails is a component reference and has no matching E0 quality endpoint.",
        ],
    )
    path = ROOT/"independent_review/RNE_SUPPLY_ROOT_REVIEW_V1.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(out, indent=2)+"\n")
    print("PASS", len(checked), "complete operator arms")

if __name__ == "__main__":
    main()
