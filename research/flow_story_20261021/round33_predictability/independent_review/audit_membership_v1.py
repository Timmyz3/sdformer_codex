"""R33 independent metadata-only membership audit. Never opens model/NPY/GT arrays."""
import collections
import csv
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROUND = HERE.parent
ROOT = ROUND.parents[2]


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text())


member_path = ROUND/"data/MEMBERSHIP_V1.json"
d = read(member_path)
assert sha(member_path) == "d860612a3ae3e4620fa2a9a616df9f8e93774efadcf7dc564881e27b7e7da8cc"
assert sha(ROUND/"data/build_membership.py") == d["builder_sha256"]
for rel, digest in d["source_pins"].items():
    assert sha(ROOT/rel) == digest
old = read(ROOT/"research/flow_story_20261019/round27/independent_review/membership_v1.json")
canonical = read(ROOT/"research/flow_story_20261011/preflight/data_protocol/artifacts/common_targets.json")["records"]
canonical_by_id = {r["target_id"]: r for r in canonical}
assert len(canonical) == len(canonical_by_id)
setup = ROOT/"research/flow_story_20261012/baseline/native_setup"
verification = read(setup/"verification_files.json")["files"]
verify_by_path = {x["relative_path"]: x for x in verification}
assert len(verification) == len(verify_by_path)
with (setup/"native_deployment_manifest.csv").open() as f:
    deployment = list(csv.DictReader(f))
by_target = collections.defaultdict(list)
for x in deployment:
    if x["category"] in ("input", "gt", "mask"):
        by_target[x["target"]].append(x)
exclusions = {k: set(v) for k, v in old["excluded_development_exposures"].items()}
exclusions["R27_R32_train256_dev64"] = {x["target_id"] for x in old["records"]}
assert {k: set(v) for k, v in d["excluded_identity_groups"].items()} == exclusions
consumed = set().union(*exclusions.values())
train_sequences = old["sequences"]
confirm_sequences = sorted({x["sequence"] for x in canonical if x["protocol_split"] == "valid"})
assert len(train_sequences) == 8 and len(confirm_sequences) == 5
assert not set(train_sequences) & set(confirm_sequences)
timestamps = {}
for item in d["timestamp_sources"]:
    p = ROOT/item["relative_path"]
    assert sha(p) == item["sha256"]
    rel = f'train_optical_flow/{item["sequence"]}/flow/forward_timestamps.txt'
    assert item["sha256"] == verify_by_path[rel]["sha256"]
    with p.open() as f:
        ts = [tuple(int(v.strip()) for v in row.split(",")) for row in f if row.strip() and not row.startswith("#")]
    assert all(len(x) == 2 and x[1] > x[0] for x in ts)
    assert all(a[0] < b[0] for a, b in zip(ts, ts[1:]))
    timestamps[item["sequence"]] = ts
assert set(timestamps) == set(train_sequences + confirm_sequences)


def interval(row):
    index = int(row["target_index_1based"])
    assert row["target_id"] == f'{row["sequence"]}_{index:04d}'
    return timestamps[row["sequence"]][index-1]


def separation(a, b):
    x, y = interval(a); u, v = interval(b)
    return max(x, u) - min(y, v)


def pick(pool, role, n):
    chronological = sorted(pool, key=lambda x: (interval(x), x["target_id"]))
    result = []
    for k in range(n):
        start, end = k*len(chronological)//n, (k+1)*len(chronological)//n
        candidates = chronological[start:end]
        assert candidates
        # Explicit digest ordering independently reproduces the declared selection.
        ranked = sorted((hashlib.sha256(f'flow-r33-20261021:{role}:{x["target_id"]}'.encode()).hexdigest(), x["target_id"]) for x in candidates)
        result.append(canonical_by_id[ranked[0][1]])
    return result


expected, pools = [], []
for seq in train_sequences:
    pool = [x for x in canonical if x["sequence"] == seq and x["protocol_split"] == "train" and x["target_id"] not in consumed]
    tune = pick(pool, "tune", 2)
    fit_pool = [x for x in pool if all(x["target_id"] != y["target_id"] and separation(x, y) >= 100000 for y in tune)]
    fit = pick(fit_pool, "fit", 8)
    expected += [(role, x) for role, rows in [("fit", fit), ("tune", tune)] for x in rows]
    pools.append({"sequence": seq, "fresh_pool": len(pool), "fit_pool_after_tune_purge": len(fit_pool)})
for seq in confirm_sequences:
    pool = [x for x in canonical if x["sequence"] == seq and x["protocol_split"] == "valid" and x["target_id"] not in consumed]
    expected += [("confirm", x) for x in pick(pool, "confirm", 8)]
    pools.append({"sequence": seq, "fresh_pool": len(pool)})
assert pools == d["candidate_pools"]
assert [(x["role"], x["target_id"]) for x in d["records"]] == [(role, x["target_id"]) for role, x in expected]
counts = dict(collections.Counter(x["role"] for x in d["records"]))
assert counts == d["counts"] == {"fit": 64, "tune": 16, "confirm": 40}
ids = {x["target_id"] for x in d["records"]}
assert len(ids) == 120 and not ids & consumed
source_paths = set()
expected_header = {"input": ("float32", [10, 480, 640]), "gt": ("float32", [2, 480, 640]), "mask": ("bool", [480, 640])}
for x in d["records"]:
    c = canonical_by_id[x["target_id"]]
    assert x["canonical"] == c and x["sequence"] == c["sequence"]
    assert (x["t0_us"], x["t1_us"]) == interval(c)
    assert x["source_files"] == by_target[x["target_id"]]
    assert len(x["source_files"]) == 3
    assert {f["category"] for f in x["source_files"]} == {"input", "gt", "mask"}
    for f in x["source_files"]:
        v = verify_by_path[f["relative_path"]]
        assert f["sequence"] == x["sequence"] and f["target"] == x["target_id"]
        assert f["split"] == c["protocol_split"]
        assert v["passed"] and v["all_elements_finite"] and v["sha256"] == f["sha256"] and v["bytes"] == int(f["bytes"])
        dtype, shape = expected_header[f["category"]]
        assert v["npy_header"]["dtype"] == dtype and v["npy_header"]["shape"] == shape
        assert not v["npy_header"]["fortran_order"]
        source_paths.add(f["relative_path"])
assert len(source_paths) == 360
gaps = [separation(a, b) for role, a in expected for other, b in expected if role == "fit" and other == "tune" and a["sequence"] == b["sequence"]]
assert min(gaps) == d["temporal_contract"]["fit_tune_minimum_gap_us"] == 200007

# Historical metadata receipts close the native T10 timestamp convention without
# opening any tensor. Canonical pending annotations are not treated as proof.
acceptance = read(setup/"acceptance.json")
native = read(setup/"native_deployment_manifest.json")
assert acceptance["passed"]
assert acceptance["deployment_manifest_sha256"] == sha(setup/"native_deployment_manifest.json")
assert native["files_csv_sha256"] == sha(setup/"native_deployment_manifest.csv")
assert acceptance["verification_files_sha256"] == sha(setup/"verification_files.json")
generator = ROOT/"research/flow_story_20261012/data_encoding/generate_native_inputs.py"
assert sha(generator) == native["encoding_contract"]["script_sha256"]
assert "stamps[row['target_index_1based'] - 1]" in generator.read_text()
assert native["encoding_contract"]["slice"] == "[t_beg,t_end)"

hist_by_seq = {s: sum(t in consumed for t, r in canonical_by_id.items() if r["sequence"] == s) for s in confirm_sequences}
out = {"schema": "flow_r33_membership_independent_audit_v1", "status": "PASS_METADATA_SCOPE",
       "membership_sha256": sha(member_path), "builder_sha256": d["builder_sha256"], "auditor_sha256": sha(__file__),
       "counts": counts, "unique_identities": len(ids), "source_file_metadata_records": len(source_paths),
       "deterministic_all120_role_order_exact": True, "known_exclusion_union_count": len(consumed), "known_exclusion_identity_overlap": 0,
       "fit_sequences": train_sequences, "confirm_sequences": confirm_sequences, "confirm_sequence_disjoint_from_fit_tune": True,
       "minimum_fit_tune_interval_gap_us": min(gaps), "timestamp_files_rehashed": len(timestamps),
       "known_historical_exposure_counts_in_confirmation_sequences": hist_by_seq,
       "encoding_contract_sha256": native["encoding_contract_sha256"],
       "metadata_receipts": {str(p.relative_to(ROOT)): sha(p) for p in [setup/"acceptance.json", setup/"native_deployment_manifest.json", generator]},
       "GT_arrays_read": False, "quality_results_read": False, "model_forwards": 0,
       "limitations": ["Only pinned historical exposure IDs were excluded; inventory not exhaustive.", "Confirmation uses repository-local validation sequences, not official DSEC test and not a claim of backbone-selection-unseen data.", "Source tensor integrity is checked against prior frozen metadata, not rehashed current remote tensor bytes; collector must bind actual bytes.", "Native T10 interval convention is backed by pinned generation source and prior acceptance metadata, not canonical pending-status text alone."],
       "scope_conclusion": "Deterministic, source-bound membership is suitable for the frozen small selector exploration. Confirmation requires head weights frozen before its collector, and no quality-dependent member replacement."}
(HERE/"MEMBERSHIP_INDEPENDENT_AUDIT_V1.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
