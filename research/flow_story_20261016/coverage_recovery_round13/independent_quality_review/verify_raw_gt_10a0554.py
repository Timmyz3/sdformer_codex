#!/usr/bin/env python3
"""CPU-only exact original PNG/GT/mask check; never changes source arrays."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np


TARGET = "zurich_city_10_a_0554"
EXPECTED = {
    "001916.png": "698122bd8dce30cbe7d7b1704a6569e11d7005e8d26877c6e42ccc8f6563ca99",
    "forward_timestamps.txt": "67df5b6c6a028b7c12f7271e861b8cacc4db27733830e504ebea8cc59d227f43",
    TARGET + ".json": "142032232f9c333c196e71b385d450e75d95d2def9795dbd348a30a507a9ae02",
    "source_selection_receipt.json": "1b0c5181b0db754c21ca0f17f8da7701a92745164c1aadcd6a0d16ed24393952",
}
ARRAYS = {
    "events": ("saved_flow_data/event_tensors/10bins/left/zurich_city_10_a/" + TARGET + ".npy",
               "dc47271d7ffb5945236ba5b0c939c1a8b3b0240ca069354bc297cd4cffee3951"),
    "gt": ("saved_flow_data/gt_tensors/" + TARGET + ".npy",
           "46b40a3b7f4aa7a04e0c95b7137af1a81def0e8f6993cb3e6a9185929d395183"),
    "mask": ("saved_flow_data/mask_tensors/" + TARGET + ".npy",
             "60991a37c45d54e41ca48d3af5dbb44a5c6f081e2c6a5923cb7421d2aa448678"),
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(raw, snapshot):
    files = {}
    for name, expected in EXPECTED.items():
        path = raw / name
        actual = sha(path)
        assert actual == expected, (name, actual, expected)
        files[name] = {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}
    arrays = {}
    for kind, (relative, expected) in ARRAYS.items():
        path = snapshot / relative
        actual = sha(path)
        assert actual == expected, (kind, actual, expected)
        arrays[kind] = np.load(path, allow_pickle=False)
        files[kind] = {"path": str(path.resolve()), "sha256": actual, "bytes": path.stat().st_size}
    source = json.loads((raw / "source_selection_receipt.json").read_text())
    names = source["ordered_png_basenames"]
    assert names == sorted(names) and len(set(names)) == len(names) == source["png_count"] == 752
    assert all(Path(n).name == n and n.endswith(".png") for n in names)
    assert source["selected_index_zero_based"] == 553
    assert names[553] == source["selected_png"] == "001916.png"
    assert source["png_sha256"] == EXPECTED["001916.png"] and source["png_bytes"] == 203026
    assert source["timestamps_sha256"] == EXPECTED["forward_timestamps.txt"]
    assert source["timestamp_bytes"] == 18837
    ts_path = raw / "forward_timestamps.txt"
    lines = ts_path.read_text().splitlines()
    ts = np.loadtxt(ts_path, delimiter=",", dtype=np.int64)
    assert lines[0] == "# from_timestamp_us, to_timestamp_us"
    assert ts.shape == (752, 2)
    pair = [57363107613, 57363207611]
    assert ts[553].tolist() == pair
    assert [int(t.strip()) for t in lines[554].split(",")] == pair
    receipt = json.loads((raw / (TARGET + ".json")).read_text())
    assert receipt["schema_version"] == "flow_native_input_receipt_v1"
    assert receipt["target_id"] == TARGET and receipt["timestamp_index_1based"] == 554
    assert receipt["timestamp_pair_us"] == pair
    assert receipt["sha256"] == ARRAYS["events"][1]
    assert receipt["relative_path"] == ARRAYS["events"][0]
    assert receipt["shape"] == list(arrays["events"].shape) == [10, 480, 640]
    assert receipt["dtype"] == str(arrays["events"].dtype) == "float32"
    event_payload_sha = hashlib.sha256(arrays["events"].tobytes(order="C")).hexdigest()
    assert event_payload_sha == receipt["tensor_C_order_bytes_sha256"]
    ts_source = receipt["source_hashes"]["train_optical_flow/zurich_city_10_a/flow/forward_timestamps.txt"]
    assert ts_source == {"bytes": 18837, "sha256": EXPECTED["forward_timestamps.txt"]}
    bgr = cv2.imread(str(raw / "001916.png"), cv2.IMREAD_UNCHANGED)
    assert bgr is not None and bgr.shape == (480, 640, 3) and bgr.dtype == np.uint16
    decoded = np.stack([(bgr[:, :, 2].astype(np.float32) - 32768.) / 128.,
                        (bgr[:, :, 1].astype(np.float32) - 32768.) / 128.])
    valid = bgr[:, :, 0] != 0
    gt, mask = arrays["gt"], arrays["mask"]
    assert gt.shape == (2, 480, 640) and mask.shape == (480, 640)
    exact_flow = bool(np.array_equal(decoded, gt))
    exact_mask = bool(np.array_equal(valid, mask))
    delta = np.abs(decoded - gt)
    return {
        "schema": "flow_single_target_original_gt_exact_audit_v1",
        "status": "PASS" if exact_flow and exact_mask else "FAIL",
        "target_id": TARGET,
        "source_files": files,
        "script_sha256": sha(Path(__file__)),
        "runtime": {"python": sys.executable, "numpy": np.__version__, "opencv": cv2.__version__},
        "png_selection": {"count": 752, "index_zero_based": 553, "basename": names[553],
                          "selection_evidence": "Root's read-only original-server sorted filename receipt, independently bound and checked locally"},
        "timestamps": {"header": lines[0], "data_row_zero_based": 553,
                       "text_line_one_based": 555, "pair_us": pair, "interval_us": pair[1]-pair[0],
                       "raw_splitlines_index553_is_preceding_data_row": lines[553],
                       "receipt_pair_exact": True},
        "original_png": {"shape": list(bgr.shape), "dtype": str(bgr.dtype),
                         "decoder": "OpenCV BGR uint16; x=(R-32768)/128, y=(G-32768)/128, valid=B!=0"},
        "comparison": {"flow_all_pixels_exact": exact_flow, "mask_all_pixels_exact": exact_mask,
                       "flow_all_pixels_max_abs_diff": float(delta.max()),
                       "flow_valid_pixels_max_abs_diff": float(delta[:, valid].max()),
                       "valid_pixels": int(valid.sum())},
        "native_event_binding": {"npy_sha_exact": True, "C_order_payload_sha_exact": True,
                                 "timestamp_source_sha_exact": True, "receipt_index_and_pair_exact": True,
                                 "raw_event_slice_or_voxel_regenerated": False},
        "scope": "Closes this target's original PNG to saved GT/mask content and timestamp-index binding; does not prove native event regeneration, general dataset correctness, or model-failure cause. No source data, metrics, sample membership, or experiment rules changed.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.raw_dir, args.snapshot_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps({"status": result["status"], "comparison": result["comparison"], "output": str(args.output)}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
