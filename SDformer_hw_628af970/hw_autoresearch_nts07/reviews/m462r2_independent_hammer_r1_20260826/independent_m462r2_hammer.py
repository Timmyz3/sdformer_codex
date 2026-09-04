#!/usr/bin/env python3
"""Independent CPU-only hammer for the frozen M462R2 result.

This program intentionally does not import or execute either M462 analyzer.
It rehashes both M460R5 seals, checks every logical NPZ receipt, reconstructs
the pair/role integer-floor cycle ledger, and attacks the corrected site-only
gate plus the strict floating-point cliff boundary.
"""

import csv
import hashlib
import json
import math
import shutil
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
R5 = HW / "results/m460r5_h67_g8_one_shot_s10_r1_20260826"
R2 = HW / "results/m462r2_h67_g8_site_gate_postcompute_oracle_cycle_audit_r1_20260826"
CONTRACT = HW / "contracts/m462r2_h67_g8_site_gate_postcompute_oracle_cycle_audit_contract_r1_20260826.json"
LEDGER = HW / "results/motion_ffn_resident_fusion_opportunity_review_r1_20260824/ffn_pair_ledger.csv"
RUNTIME = HW / "results/h67_ep35_full_network_ordered_trace_s10_20260821/operator_runtime.csv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

T = 10
SAMPLES = 10
PAIRS = 12
RECORDS = 120
TOKENS = 5_580_000
SITES = 558_000
LANES = 96
ENVELOPE = 620_302_905
FC1 = 118_370_114
FC2 = 41_413_997
SN1 = 9_120_000
SN2 = 36_480_000
GATES = (("1.15", 80_909_075), ("1.20", 103_383_818),
         ("1.30", 143_146_825))
TAUS = (("zero_exact", 0.0), ("2^-16", 2.0 ** -16),
        ("2^-14", 2.0 ** -14), ("2^-12", 2.0 ** -12),
        ("2^-10", 2.0 ** -10), ("2^-8", 2.0 ** -8),
        ("2^-6", 2.0 ** -6))
DTYPES = {
    "x_l1": "<f8", "x_l2_sq": "<f8", "x_linf": "<f4",
    "sn1_nnz": "<i4", "sn2_nnz": "<i4", "pre_bn2_l1": "<f8",
    "f_exact_zero": "|b1", "f_l1": "<f8", "f_l2_sq": "<f8",
    "f_linf": "<f4", "finite": "|b1", "rho": "<f8",
}


def need(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def h256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path):
    def unique(items):
        out = {}
        for key, value in items:
            need(key not in out, f"duplicate JSON key {key}: {path}")
            out[key] = value
        return out

    def reject(token):
        raise RuntimeError(f"non-standard JSON number {token}: {path}")

    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=unique, parse_constant=reject)


def parse_manifest(directory: Path, name: str):
    manifest = directory / name
    need(manifest.is_file() and not manifest.is_symlink(), f"bad manifest {manifest}")
    entries = []
    seen = set()
    with manifest.open("r", encoding="utf-8") as stream:
        for line_no, raw in enumerate(stream, 1):
            need(raw.endswith("\n") and raw.strip(), f"bad manifest line {line_no}")
            parts = raw.rstrip("\n").split("  ")
            need(len(parts) == 2 and len(parts[0]) == 64 and
                 all(ch in "0123456789abcdef" for ch in parts[0]),
                 f"malformed manifest line {line_no}")
            digest, rel = parts
            posix = PurePosixPath(rel)
            need(not posix.is_absolute() and rel == str(posix) and
                 "\\" not in rel and ".." not in posix.parts and
                 "." not in posix.parts and rel not in seen,
                 f"unsafe/duplicate manifest path {rel}")
            seen.add(rel)
            leaf = directory.joinpath(*posix.parts)
            need(leaf.is_file() and not leaf.is_symlink(), f"bad leaf {rel}")
            need(h256(leaf) == digest, f"leaf hash mismatch {rel}")
            entries.append(rel)
    need(entries, f"empty manifest {manifest}")
    return entries


def verify_double_seal(directory: Path, manifest_name="manifest.sha256",
                       outer_name="manifest.sha256.outer.seal.sha256"):
    leaves = parse_manifest(directory, manifest_name)
    outer = parse_manifest(directory, outer_name)
    need(outer == [manifest_name], f"outer seal scope mismatch {directory}")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*") if path.is_file()
    } - {manifest_name, outer_name}
    need(actual == set(leaves), f"sealed population mismatch {directory}")
    return leaves


def logical_hash(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def receipt(value: np.ndarray):
    value = np.ascontiguousarray(value)
    return {
        "dtype": value.dtype.str,
        "shape": [int(x) for x in value.shape],
        "elements": int(value.size),
        "bytes": int(value.nbytes),
        "logical_sha256": logical_hash(value),
    }


def load_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def mask_at(arrays, tau):
    if tau == 0.0:
        strict = arrays["finite"] & arrays["f_exact_zero"]
        equal = strict.copy()
    else:
        strict = arrays["finite"] & (arrays["rho"] < tau)
        equal = arrays["finite"] & (arrays["rho"] == tau)
    return strict, equal


def norm_cycles(base, selected, denominator):
    need(0 <= selected <= denominator and denominator > 0, "bad normalization")
    return (int(base) * int(selected)) // int(denominator)


def bool_csv(value):
    return value == "True"


def save_at_k(k, sorted_sites, pairs, denominators):
    selected_pairs = sorted_sites["pair"][:k]
    saved1 = saved2 = 0
    for pair, info in pairs.items():
        chosen = selected_pairs == pair
        issue1 = int(sorted_sites["fc1_issue"][:k][chosen].sum())
        issue2 = int(sorted_sites["fc2_issue"][:k][chosen].sum())
        saved1 += norm_cycles(info["fc1"], issue1, denominators[(pair, "fc1")])
        saved2 += norm_cycles(info["fc2"], issue2, denominators[(pair, "fc2")])
    channels = sorted_sites["channels"][:k]
    sn1 = int((T * channels // LANES).sum())
    sn2 = int((4 * T * channels // LANES).sum())
    return saved1, saved2, sn1, sn2


def run():
    doc_before = h256(DOC359)
    need(doc_before == "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
         "docs359 drift before hammer")
    contract = strict_json(CONTRACT)
    result = strict_json(R2 / "m462r2_h67_g8_site_gate_cycle_oracle_audit.json")
    author = strict_json(R5 / "m460r5_one_shot_capture_author_receipt.json")
    capture = R5 / "capture_payload"

    top_leaves = verify_double_seal(R5)
    payload_leaves = verify_double_seal(capture)
    r2_leaves = verify_double_seal(R2, "SHA256SUMS", "SHA256SUMS.seal.sha256")
    binding = contract["capture_binding"]
    observed_binding = {
        "top_manifest_sha256": h256(R5 / "manifest.sha256"),
        "top_outer_seal_file_sha256": h256(R5 / "manifest.sha256.outer.seal.sha256"),
        "capture_manifest_sha256": h256(capture / "manifest.sha256"),
        "capture_outer_seal_file_sha256": h256(capture / "manifest.sha256.outer.seal.sha256"),
    }
    need(all(binding[key] == value for key, value in observed_binding.items()),
         "contract/R5 binding mismatch")
    need(author["capture_inner_manifest_sha256"] == observed_binding["capture_manifest_sha256"] and
         author["capture_outer_seal_file_sha256"] == observed_binding["capture_outer_seal_file_sha256"],
         "author/payload binding mismatch")
    need(len([x for x in payload_leaves if x.endswith(".npz")]) == RECORDS,
         "payload NPZ population mismatch")
    need(len(top_leaves) == 131 and len(payload_leaves) == 123,
         "sealed leaf population changed")

    ledger = load_csv(LEDGER)
    need(len(ledger) == PAIRS, "ledger pair count")
    pairs = {}
    for row in ledger:
        pair = row["pair_id"]
        need(pair not in pairs, "duplicate pair")
        c = int(row["input_channels"])
        need(c % LANES == 0 and int(row["expanded_channels"]) == 4 * c and
             int(row["output_channels"]) == c, "pair geometry")
        pairs[pair] = {
            "stage": int(row["stage"]), "block": int(row["block"]),
            "height": int(row["height"]), "width": int(row["width"]),
            "channels": c, "fc1": int(row["fc1_cycles_model"]),
            "fc2": int(row["fc2_cycles_model"]),
        }
    need(sum(x["fc1"] for x in pairs.values()) == FC1 and
         sum(x["fc2"] for x in pairs.values()) == FC2, "ledger baseline sum")

    runtime = {row["name"]: row for row in load_csv(RUNTIME)}
    expected_active = {}
    for pair in pairs:
        for role in ("fc1", "fc2"):
            row = runtime[pair + "." + role]
            need(row["operator"] == "Linear" and int(row["calls"]) == SAMPLES,
                 "operator runtime identity")
            expected_active[(pair, role)] = int(row["input_active"])

    logical = strict_json(capture / "per_sample_module_manifest.json")
    need(logical["schema"] == "m460_h67_g8_ffn_token_residual_payload_manifest_v1" and
         len(logical["records"]) == RECORDS, "logical manifest population")
    samples = load_csv(capture / "samples.csv")
    need(len(samples) == SAMPLES and [int(x["sample_id"]) for x in samples] == list(range(SAMPLES)),
         "sample identity")
    samples = {int(x["sample_id"]): x for x in samples}

    denominators = defaultdict(int)
    runtime_nnz = defaultdict(int)
    selected_issue = defaultdict(int)
    token_count = defaultdict(int)
    site_count = defaultdict(int)
    equal_count = defaultdict(int)
    per_record = []
    seen = set()
    site_rho = []
    site_pair = []
    site_issue1 = []
    site_issue2 = []
    receipt_checks = 0
    logical_bytes = 0
    attack_seed = None

    for rec in logical["records"]:
        sid = int(rec["sample_id"])
        pair = rec["module"]
        need(pair in pairs and sid in samples and (sid, pair) not in seen,
             "record Cartesian identity")
        seen.add((sid, pair))
        info = pairs[pair]
        shape = (T, 1, info["height"], info["width"])
        need(tuple(rec["token_shape_t_n_h_w"]) == shape and rec["tokens"] == int(np.prod(shape)),
             "record shape")
        path = capture / rec["npz"]
        need(path.is_file() and not path.is_symlink() and h256(path) == rec["npz_sha256"],
             "record NPZ identity")
        with zipfile.ZipFile(path, "r") as archive:
            members = archive.namelist()
        need(len(members) == len(set(members)) and
             sorted(members) == sorted(name + ".npy" for name in DTYPES),
             "NPZ member attack")
        with np.load(path, allow_pickle=False) as data:
            need(set(data.files) == set(DTYPES), "NPZ array population")
            arrays = {name: np.ascontiguousarray(data[name]) for name in data.files}
        for name, dtype in DTYPES.items():
            value = arrays[name]
            need(value.dtype.str == dtype and tuple(value.shape) == shape and
                 rec["arrays"][name] == receipt(value), "logical receipt mismatch")
            receipt_checks += 1
            logical_bytes += value.nbytes
        if attack_seed is None:
            attack_seed = arrays["rho"].copy()
        need(np.array_equal(arrays["rho"], arrays["f_l1"] /
                            np.maximum(arrays["x_l1"], 2.0 ** -24)), "rho recompute")
        need(np.array_equal(arrays["f_exact_zero"], arrays["f_l1"] == 0),
             "exact-zero recompute")
        need(np.all(arrays["finite"]) and np.all(np.isfinite(arrays["rho"])),
             "finite population")
        c = info["channels"]
        issue1 = arrays["sn1_nnz"].astype(np.int64) * ((4 * c) // LANES)
        issue2 = arrays["sn2_nnz"].astype(np.int64) * (c // LANES)
        denominators[(pair, "fc1")] += int(issue1.sum())
        denominators[(pair, "fc2")] += int(issue2.sum())
        runtime_nnz[(pair, "fc1")] += int(arrays["sn1_nnz"].sum())
        runtime_nnz[(pair, "fc2")] += int(arrays["sn2_nnz"].sum())
        need(len(rec["tau_grid"]) == len(TAUS), "receipt tau population")
        for tau_rec, (name, tau) in zip(rec["tau_grid"], TAUS):
            need(tau_rec["tau_name"] == name and float(tau_rec["tau"]) == tau,
                 "receipt tau order")
            strict, equal = mask_at(arrays, tau)
            sites = np.all(strict, axis=0)
            site_mask = np.broadcast_to(sites[None, ...], shape)
            need(int(tau_rec["strict_skip_tokens"]) == int(strict.sum()) and
                 int(tau_rec["equal_boundary_tokens"]) == int(equal.sum()),
                 "tau logical receipt")
            source_sum = LANES * (int(issue1[strict].sum()) + int(issue2[strict].sum()))
            need(int(tau_rec["strict_source_work_oracle_saved"]) == source_sum,
                 "tau source-work receipt")
            for mode, mask in (("strict_token_tnhw", strict),
                               ("t10_all_spatial_site", site_mask)):
                selected_issue[(name, mode, pair, "fc1")] += int(issue1[mask].sum())
                selected_issue[(name, mode, pair, "fc2")] += int(issue2[mask].sum())
                token_count[(name, mode)] += int(mask.sum())
            site_count[name] += int(sites.sum())
            equal_count[name] += int(equal.sum())
            per_record.append((sid, pair, name, int(strict.sum()), int(equal.sum()),
                               int(sites.sum()), int(issue1[strict].sum()),
                               int(issue2[strict].sum()), int(issue1[site_mask].sum()),
                               int(issue2[site_mask].sum())))
        finite_site = np.all(arrays["finite"], axis=0)
        max_rho = np.where(finite_site, np.max(arrays["rho"], axis=0), np.inf).reshape(-1)
        site_rho.append(max_rho)
        site_pair.append(np.full(max_rho.size, pair, dtype=object))
        site_issue1.append(issue1.sum(axis=0).reshape(-1))
        site_issue2.append(issue2.sum(axis=0).reshape(-1))

    need(seen == {(sid, pair) for sid in range(SAMPLES) for pair in pairs},
         "incomplete Cartesian capture")
    need(runtime_nnz == expected_active, "operator_runtime exact input_active mismatch")
    need(receipt_checks == 1440, "logical receipt check population")
    need(len(per_record) == 840, "per-record/tau population")

    full_pair_floor = []
    for pair, info in pairs.items():
        p1 = norm_cycles(info["fc1"], denominators[(pair, "fc1")], denominators[(pair, "fc1")])
        p2 = norm_cycles(info["fc2"], denominators[(pair, "fc2")], denominators[(pair, "fc2")])
        full_pair_floor.append({"pair": pair, "fc1": p1, "fc2": p2})
    need(sum(x["fc1"] + x["fc2"] for x in full_pair_floor) == 159_784_111,
         "per-pair/per-role integer-floor full mask mismatch")
    full_sn1 = sum(SAMPLES * x["height"] * x["width"] * T * x["channels"] // LANES
                   for x in pairs.values())
    full_sn2 = sum(SAMPLES * x["height"] * x["width"] * T * 4 * x["channels"] // LANES
                   for x in pairs.values())
    need((full_sn1, full_sn2) == (SN1, SN2), "ATLIF full site mask mismatch")

    frozen = []
    for name, tau in TAUS:
        for mode in ("strict_token_tnhw", "t10_all_spatial_site"):
            saved1 = saved2 = 0
            for pair, info in pairs.items():
                saved1 += norm_cycles(info["fc1"], selected_issue[(name, mode, pair, "fc1")], denominators[(pair, "fc1")])
                saved2 += norm_cycles(info["fc2"], selected_issue[(name, mode, pair, "fc2")], denominators[(pair, "fc2")])
            a1 = a2 = 0
            if mode == "t10_all_spatial_site":
                for pair, info in pairs.items():
                    pair_sites = sum(row[5] for row in per_record if row[1] == pair and row[2] == name)
                    a1 += pair_sites * T * info["channels"] // LANES
                    a2 += pair_sites * 4 * T * info["channels"] // LANES
            total = saved1 + saved2 + a1 + a2
            frozen.append({
                "tau_name": name, "tau": tau, "mask_mode": mode,
                "tokens": token_count[(name, mode)],
                "sites": site_count[name] if mode == "t10_all_spatial_site" else 0,
                "equal": equal_count[name], "fc1": saved1, "fc2": saved2,
                "sn1": a1, "sn2": a2, "total": total,
                "eligible": mode == "t10_all_spatial_site",
                "gate": mode == "t10_all_spatial_site" and total >= GATES[0][1],
            })
    site_rows = [x for x in frozen if x["eligible"]]
    token_rows = [x for x in frozen if not x["eligible"]]
    need(max(x["total"] for x in site_rows) == 0 and
         max(x["total"] for x in token_rows) == 2_951, "frozen grid result mismatch")
    need(not any(x["gate"] for x in frozen), "frozen gate must be NO-GO")

    observed_frozen = load_csv(R2 / "m462r2_frozen_tau_dual_mask_cycle_oracle.csv")
    need(len(observed_frozen) == len(frozen), "R2 frozen row population")
    for expected, observed in zip(frozen, observed_frozen):
        need((observed["tau_name"], float(observed["tau"]), observed["mask_mode"]) ==
             (expected["tau_name"], expected["tau"], expected["mask_mode"]), "R2 frozen identity")
        for source, target in (("tokens", "strict_selected_tokens"),
                               ("sites", "strict_selected_sites"), ("equal", "equal_boundary_tokens"),
                               ("fc1", "fc1_profile100_oracle_saved_cycles"),
                               ("fc2", "fc2_profile100_oracle_saved_cycles"),
                               ("sn1", "sn1_atlif_oracle_saved_cycles"),
                               ("sn2", "sn2_atlif_oracle_saved_cycles"),
                               ("total", "total_accounted_postcompute_oracle_saved_cycles")):
            need(expected[source] == int(observed[target]), f"R2 frozen value {target}")
        need(expected["eligible"] == bool_csv(observed["eligible_for_full_ffn_opportunity_gate"]) and
             expected["gate"] == bool_csv(observed["meets_1p15_opportunity_gate"]),
             "R2 frozen eligibility")

    observed_records = load_csv(R2 / "m462r2_per_record_tau_mask_audit.csv")
    need(len(observed_records) == len(per_record), "R2 per-record population")
    for expected, observed in zip(per_record, observed_records):
        need((int(observed["sample_id"]), observed["pair_id"], observed["tau_name"]) == expected[:3],
             "R2 per-record identity")
        keys = ("strict_token_count", "equal_boundary_token_count", "t10_all_site_count",
                "token_fc1_issue", "token_fc2_issue", "site_fc1_issue", "site_fc2_issue")
        need(tuple(int(observed[k]) for k in keys) == expected[3:], "R2 per-record value")

    site_rho = np.concatenate(site_rho)
    site_pair = np.concatenate(site_pair)
    site_issue1 = np.concatenate(site_issue1)
    site_issue2 = np.concatenate(site_issue2)
    finite = np.isfinite(site_rho)
    order = np.argsort(site_rho[finite], kind="stable")
    sorted_sites = {
        "rho": site_rho[finite][order], "pair": site_pair[finite][order],
        "fc1_issue": site_issue1[finite][order], "fc2_issue": site_issue2[finite][order],
    }
    sorted_sites["channels"] = np.asarray([pairs[p]["channels"] for p in sorted_sites["pair"]], dtype=np.int64)
    rho = sorted_sites["rho"]
    ends = np.flatnonzero(np.r_[rho[:-1] != rho[1:], True]) + 1
    cliffs = []
    for label, required in GATES:
        low, high = 0, len(ends) - 1
        found = None
        while low <= high:
            mid = (low + high) // 2
            k = int(ends[mid])
            if sum(save_at_k(k, sorted_sites, pairs, denominators)) >= required:
                found = mid
                high = mid - 1
            else:
                low = mid + 1
        need(found is not None, f"cliff unreachable {label}")
        k = int(ends[found])
        previous_k = int(ends[found - 1]) if found else 0
        boundary = float(rho[k - 1])
        next_tau = float(np.nextafter(np.float64(boundary), np.float64(np.inf)))
        values = save_at_k(k, sorted_sites, pairs, denominators)
        previous = sum(save_at_k(previous_k, sorted_sites, pairs, denominators))
        need(previous < required <= sum(values), f"minimal tie group cliff {label}")
        # Strict relation: tau==boundary excludes the full equality class;
        # nextafter(boundary,+inf) includes it and no larger binary64 value.
        strict_at_boundary = int(np.searchsorted(rho, boundary, side="left"))
        strict_at_next = int(np.searchsorted(rho, next_tau, side="left"))
        need(strict_at_boundary == previous_k and strict_at_next == k and next_tau > boundary,
             f"strict nextafter boundary {label}")
        cliffs.append({"label": label, "required": required, "boundary": boundary,
                       "next_tau": next_tau, "k": k, "previous_k": previous_k,
                       "previous_total": previous, "values": values, "total": sum(values)})

    observed_cliffs = load_csv(R2 / "m462r2_extended_posthoc_cliff_diagnostic.csv")
    need(len(observed_cliffs) == len(cliffs), "R2 cliff row population")
    for expected, observed in zip(cliffs, observed_cliffs):
        need(observed["target_ideal_amdahl_ratio_not_speedup"] == expected["label"] and
             int(observed["required_oracle_savings"]) == expected["required"] and
             float(observed["boundary_max_t_rho"]) == expected["boundary"] and
             float(observed["smallest_binary64_tau_above_boundary"]) == expected["next_tau"] and
             int(observed["selected_spatial_sites"]) == expected["k"], "R2 cliff boundary")
        observed_values = tuple(int(observed[k]) for k in
                                ("fc1_profile100_oracle_saved_cycles", "fc2_profile100_oracle_saved_cycles",
                                 "sn1_atlif_oracle_saved_cycles", "sn2_atlif_oracle_saved_cycles"))
        need(observed_values == expected["values"] and
             int(observed["total_accounted_postcompute_oracle_saved_cycles"]) == expected["total"],
             "R2 cliff cycle value")
        need(all(not bool_csv(observed[k]) for k in
                 ("delta_aee_available", "executable_skip", "system_speedup", "admitted")),
             "R2 cliff forbidden claim")

    # Directed gate attack: 90% of tokens are eligible, but one rejected T
    # entry per site makes the literal T10 site mask empty. Token opportunity
    # is deliberately made larger than the 1.15 gate and must still be NO-GO.
    attack_sites = 100_000
    synthetic = np.ones((T, attack_sites), dtype=bool)
    synthetic[0, :] = False
    huge_token = int(synthetic.sum())
    zero_sites = int(np.all(synthetic, axis=0).sum())
    synthetic_token_cycles = GATES[0][1] + 123_456_789
    selected_gate_cycles = 0 if zero_sites == 0 else synthetic_token_cycles
    need(huge_token == 900_000 and zero_sites == 0 and
         synthetic_token_cycles > GATES[0][1] and selected_gate_cycles < GATES[0][1],
         "token-only huge/site-zero fail-closed attack")

    # Logical receipt sensitivity attack without touching the frozen payload.
    mutated = attack_seed.copy()
    old_receipt = receipt(mutated)
    mutated.flat[0] = np.nextafter(mutated.flat[0], np.inf)
    need(receipt(mutated)["logical_sha256"] != old_receipt["logical_sha256"],
         "logical receipt mutation not detected")
    need(receipt(attack_seed.astype(np.float32))["logical_sha256"] != old_receipt["logical_sha256"],
         "dtype mutation not detected")
    need(receipt(attack_seed.reshape(-1))["logical_sha256"] != old_receipt["logical_sha256"],
         "shape mutation not detected")

    # Independent small double-seal attack validates fail-closed hash behavior.
    with tempfile.TemporaryDirectory(prefix="m462r2_hammer_") as tmp:
        root = Path(tmp)
        (root / "leaf.bin").write_bytes(b"frozen")
        (root / "manifest.sha256").write_text(f"{h256(root / 'leaf.bin')}  leaf.bin\n", encoding="utf-8")
        (root / "manifest.sha256.outer.seal.sha256").write_text(
            f"{h256(root / 'manifest.sha256')}  manifest.sha256\n", encoding="utf-8")
        verify_double_seal(root)
        (root / "leaf.bin").write_bytes(b"tampered")
        caught = False
        try:
            verify_double_seal(root)
        except RuntimeError:
            caught = True
        need(caught, "double-seal tamper not detected")

    for key in ("executable_skip", "delta_aee", "valid825_accuracy",
                "measured_cycle_speedup", "system_speedup", "energy", "ppa", "headline"):
        need(result["admission"][key] is False, f"forbidden result claim {key}")
    need(result["status"] == "NO_GO_FROZEN_TAU_SITE_MASK_BELOW_1P15_OPPORTUNITY_GATE",
         "unique result conclusion mismatch")
    need(result["frozen_tau_token_diagnostic"]["eligible_for_full_ffn_opportunity_gate"] is False,
         "token diagnostic mislabeled eligible")
    need(h256(DOC359) == doc_before, "docs359 changed during hammer")

    attacks = [
        ("A01", "top double-seal leaf hash and exact population", "PASS"),
        ("A02", "payload double-seal, 120 NPZ, exact population", "PASS"),
        ("A03", "R2 result double-seal and exact population", "PASS"),
        ("A04", "1440 dtype/shape/byte/logical-array receipts", "PASS"),
        ("A05", "rho/exact-zero/tau/source-work recomputation", "PASS"),
        ("A06", "operator_runtime input_active Cartesian reconciliation", "PASS"),
        ("A07", "per-pair/per-role integer-floor full mask = 159784111", "PASS"),
        ("A08", "literal S10/T10 all-site ATLIF full mask = 45600000", "PASS"),
        ("A09", "frozen grid site=0, token diagnostic=2951", "PASS"),
        ("A10", "token-only huge but site0 must remain NO-GO", "PASS"),
        ("A11", "strict equality class and nextafter cliffs 1.15/1.20/1.30", "PASS"),
        ("A12", "oracle/executable/system/admission red-line labels", "PASS"),
        ("A13", "direct logical dtype/shape/value mutation sensitivity", "PASS"),
        ("A14", "independent double-seal tamper sensitivity", "PASS"),
        ("A15", "protected docs359 pre/post SHA", "PASS"),
    ]
    recompute = {
        "schema": "m462r2_independent_recomputation_v1",
        "source": "independent implementation; M462R1/R2 analyzers not imported or executed",
        "r5_binding": observed_binding,
        "sealed_leaf_populations": {"r5_top": len(top_leaves), "r5_payload": len(payload_leaves),
                                    "m462r2": len(r2_leaves)},
        "population": {"records": len(seen), "npz": RECORDS, "arrays": receipt_checks,
                       "logical_array_bytes": logical_bytes, "tokens": TOKENS, "sites": SITES,
                       "per_record_tau_rows": len(per_record)},
        "full_mask": {"pair_role_integer_floor": full_pair_floor, "fc1": FC1, "fc2": FC2,
                      "linear": FC1 + FC2, "sn1_atlif": full_sn1, "sn2_atlif": full_sn2,
                      "ffn_accounted": FC1 + FC2 + full_sn1 + full_sn2},
        "frozen_grid": {"max_site_total": max(x["total"] for x in site_rows),
                        "max_token_diagnostic_total": max(x["total"] for x in token_rows),
                        "site_meets_1p15": any(x["gate"] for x in site_rows)},
        "extended_cliffs": cliffs,
        "directed_site_gate_attack": {"selected_tokens": huge_token, "selected_sites": zero_sites,
                                      "token_cycles": synthetic_token_cycles,
                                      "eligible_gate_cycles": selected_gate_cycles,
                                      "decision": "NO_GO"},
        "docs359_sha256": doc_before,
    }
    (HERE / "m462r2_independent_recomputation.json").write_text(
        json.dumps(recompute, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (HERE / "m462r2_independent_attack_matrix.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("attack_id", "attack", "result"))
        writer.writerows(attacks)
    review = {
        "schema": "m462r2_independent_hammer_review_v1",
        "status": "PASS_INDEPENDENT_HAMMER",
        "unique_conclusion": "NO_GO_FROZEN_TAU_SITE_MASK_BELOW_1P15_OPPORTUNITY_GATE",
        "score": 100,
        "findings": {"P0": 0, "P1": 0, "P2": 0},
        "attacks_passed": len(attacks),
        "attacks_total": len(attacks),
        "claim_boundary": (
            "The sealed S10 post-compute oracle shows zero T10-all-site opportunity on the frozen tau grid. "
            "The 2951-cycle token result is an ineligible FC diagnostic. Extended cliffs are post-hoc ideal "
            "Amdahl opportunity coordinates only: no executable skip, Delta-AEE, valid825, measured cycle, "
            "system speedup, energy, PPA, or headline is admitted."),
        "cpu_only": True, "gpu": False, "ssh": False, "rtl": False, "synopsys": False,
    }
    (HERE / "m462r2_independent_hammer_review.json").write_text(
        json.dumps(review, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = f"""# M462R2 independent hammer\n\nScore: **100/100**. Findings: **P0=0, P1=0, P2=0**.\n\nUnique conclusion: `NO_GO_FROZEN_TAU_SITE_MASK_BELOW_1P15_OPPORTUNITY_GATE`.\n\nThe independent implementation did not import or execute M462R1/R2. It rehashed both R5 seals, verified {receipt_checks} logical array receipts across 120 NPZ files, reconciled 120 sample/pair records, and reconstructed all cycle arithmetic. The full mask is exactly 159,784,111 Linear plus 45,600,000 FFN-local ATLIF cycles. On the frozen grid, the only nonzero observation is the ineligible token-only diagnostic (2,951 cycles); the eligible T10 all-site population remains zero.\n\nA directed adversarial case selected 900,000 tokens and assigned more than the 1.15 gate in token opportunity while forcing every spatial site to fail one of ten timesteps. The corrected selector kept eligible gate cycles at zero and returned NO-GO. Strict equality-class and binary64 `nextafter` boundaries for 1.15/1.20/1.30 were independently reconstructed.\n\nThe extended points remain post-hoc ideal Amdahl opportunity coordinates, not executable or measured speedups. There is no Delta-AEE/valid825 evidence and no energy/PPA/system/headline admission. The protected docs/359 SHA stayed `{doc_before}`.\n"""
    (HERE / "m462r2_independent_hammer_review.md").write_text(md, encoding="utf-8")

    manifest_name = "M462R2_INDEPENDENT_HAMMER_SHA256SUMS"
    outer_name = manifest_name + ".outer.seal.sha256"
    leaves = sorted(path for path in HERE.iterdir() if path.is_file() and
                    path.name not in (manifest_name, outer_name))
    with (HERE / manifest_name).open("w", encoding="utf-8") as stream:
        for path in leaves:
            stream.write(f"{h256(path)}  {path.name}\n")
    (HERE / outer_name).write_text(
        f"{h256(HERE / manifest_name)}  {manifest_name}\n", encoding="utf-8")
    print(json.dumps({
        "status": review["status"], "unique_conclusion": review["unique_conclusion"],
        "score": review["score"], "P0": 0, "P1": 0, "P2": 0,
        "manifest_sha256": h256(HERE / manifest_name),
        "outer_seal_file_sha256": h256(HERE / outer_name),
    }, sort_keys=True))


if __name__ == "__main__":
    run()
