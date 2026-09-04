#!/opt/anaconda3/bin/python3
"""Freeze three non-performance-selected M2217 power windows.

The selector is deliberately read-only.  It stratifies the complete frozen
2,880-row direct-VCS population by request-reuse density before any power
measurement exists.  Each 960-row tercile contributes one real G48-island
window.  The representative is closest to that tercile's upper-median
density, then maximizes ordinary accepted bank requests, with a canonical
identity tie break.  Representatives must come from distinct DSEC sequences.

Rows with no ordinary requests remain in the population and in the fixed
tercile weights, but cannot be selected because a production SAIF window must
exercise the critical request/response/commit cones.  M2067 continuation rows
remain in the population but cannot be representatives of the standalone G48
M2018 island.  Neither filter uses cycles, power, energy, or mapped PPA.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
LOW_META = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
LOW_MEMH = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.memh"
LOW_RESULT = HW / (
    "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903/"
    "result.json")
HIGH_META = HW / "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.json"
HIGH_RESULT = HW / (
    "results/m2067_ep34_fc2_exact_continuation_vcs_r10_codex_batch_20260904/"
    "result.json")
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"

EXPECTED = {
    LOW_META: "3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5",
    LOW_MEMH: "487ca0073526b973220abd77c91d12dbc2420901443541ec5a79e36a780e1bf0",
    LOW_RESULT: "b4ee4f9cf4d55a4f722f1487ba4bc23948bc3f6a096178fa835d9ed18b50fe2a",
    HIGH_META: "5b44aa6a248a8768d59a85270a50b3ba805467377365e1b6e4ad8e58eafc7b34",
    HIGH_RESULT: "d0707b65f1c453636e3d6d050b036789f2366bcaa564f6fc32423aae3a128756",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}
ROWS_PER_TERCILE = 960
PHYSICAL_WORDS_PER_WINDOW = 4 * 48


class Failure(RuntimeError):
    pass


def need(value: bool, message: str) -> None:
    if not value:
        raise Failure(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            need(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          Failure("nonfinite JSON: " + token)))


def ratio(saved: int, ordinary: int) -> Fraction:
    return Fraction(saved, ordinary) if ordinary else Fraction(0, 1)


def canonical(row: dict) -> tuple:
    meta = row["metadata"]
    return (meta["sequence"], int(meta["sample_id"]),
            int(meta["layer_id"]), int(meta["token_start"]),
            int(meta["slot"]), row["population"])


def population() -> list[dict]:
    for path, digest in EXPECTED.items():
        need(path.is_file() and not path.is_symlink(), "missing identity " + str(path))
        need(sha256(path) == digest, "identity drift " + str(path))
    low_meta = strict_json(LOW_META)["rows"]
    low_result = strict_json(LOW_RESULT)["rows"]
    high_meta = strict_json(HIGH_META)["rows"]
    high_result = strict_json(HIGH_RESULT)["rows"]
    need(len(low_meta) == len(low_result) == 1920, "M2051 population")
    need(len(high_meta) == len(high_result) == 960, "M2067 population")
    rows = []
    for pop, metadata_rows, result_rows in (
            ("m2051_g_le_48", low_meta, low_result),
            ("m2067_fc2_continuation", high_meta, high_result)):
        for metadata, observed in zip(metadata_rows, result_rows):
            need(int(metadata["slot"]) == int(observed["workload_slot"]),
                 "metadata/result slot mismatch")
            if pop == "m2051_g_le_48":
                ordinary = int(metadata["base_misses"]) * 12 * 8
                tsbg = int(metadata["tsbg_misses"]) * 12 * 8
            else:
                ordinary = sum(int(chunk["ordinary_misses"])
                               for chunk in metadata["chunk_rows"]) * 12 * 8
                tsbg = sum(int(chunk["tsbg_misses"])
                           for chunk in metadata["chunk_rows"]) * 12 * 8
            need(0 <= tsbg <= ordinary, "request ordering")
            rows.append({
                "population": pop, "metadata": metadata, "observed": observed,
                "ordinary_accepted_bank_requests": ordinary,
                "tsbg_accepted_bank_requests": tsbg,
                "density": ratio(ordinary - tsbg, ordinary),
            })
    need(len(rows) == 2880, "complete direct-VCS population")
    rows.sort(key=lambda row: (row["density"], canonical(row)))
    return rows


def descriptor_sha(lines: list[str], slot: int) -> str:
    begin = slot * PHYSICAL_WORDS_PER_WINDOW
    selected = lines[begin:begin + PHYSICAL_WORDS_PER_WINDOW]
    need(len(selected) == PHYSICAL_WORDS_PER_WINDOW, "descriptor extent")
    return hashlib.sha256(("\n".join(selected) + "\n").encode("ascii")).hexdigest()


def select() -> dict:
    rows = population()
    lines = LOW_MEMH.read_text(encoding="ascii").splitlines()
    need(len(lines) == 1920 * PHYSICAL_WORDS_PER_WINDOW, "M2051 MEMH extent")
    labels = ("low", "median", "high")
    used_sequences: set[str] = set()
    selections = []
    bins = []
    for index, label in enumerate(labels):
        members = rows[index * ROWS_PER_TERCILE:(index + 1) * ROWS_PER_TERCILE]
        need(len(members) == ROWS_PER_TERCILE, "tercile extent")
        target = members[len(members) // 2]["density"]
        candidates = [row for row in members
                      if row["population"] == "m2051_g_le_48"
                      and row["ordinary_accepted_bank_requests"] > 0
                      and row["metadata"]["sequence"] not in used_sequences]
        need(candidates, "no eligible representative for " + label)
        candidates.sort(key=lambda row: (
            abs(row["density"] - target),
            -row["ordinary_accepted_bank_requests"], canonical(row)))
        chosen = candidates[0]
        used_sequences.add(chosen["metadata"]["sequence"])
        meta, observed = chosen["metadata"], chosen["observed"]
        selections.append({
            "stratum": label,
            "population_rank_begin_inclusive": index * ROWS_PER_TERCILE,
            "population_rank_end_exclusive": (index + 1) * ROWS_PER_TERCILE,
            "population_weight_numerator": ROWS_PER_TERCILE,
            "population_weight_denominator": 2880,
            "target_density_fraction": [target.numerator, target.denominator],
            "selected_density_fraction": [chosen["density"].numerator,
                                          chosen["density"].denominator],
            "source_fixture": str(LOW_MEMH.relative_to(ROOT)),
            "source_fixture_sha256": EXPECTED[LOW_MEMH],
            "global_slot": int(meta["slot"]),
            "sample_id": int(meta["sample_id"]),
            "sequence": meta["sequence"],
            "layer_id": int(meta["layer_id"]),
            "target": meta["target"],
            "token_role": meta["token_role"],
            "token_start": int(meta["token_start"]),
            "source_groups": int(meta["source_groups"]),
            "physical_source_groups": 48,
            "descriptor_word_count": PHYSICAL_WORDS_PER_WINDOW,
            "descriptor_text_sha256": descriptor_sha(lines, int(meta["slot"])),
            "rows": int(meta["live_rows"]),
            "issues": int(meta["issues"]),
            "products": int(meta["products"]),
            "commits": int(observed["commits"]),
            "ordinary": {
                "cycles": int(observed["base_cycles"]),
                "cache_misses": int(meta["base_misses"]),
                "cache_hits": int(meta["base_hits"]),
                "cache_evictions": int(meta["base_evictions"]),
                "bundles": int(observed["bundles_base"]),
                "accepted_bank_requests": int(observed["scalar_base"]),
            },
            "tsbg": {
                "cycles": int(observed["tsbg_cycles"]),
                "cache_misses": int(meta["tsbg_misses"]),
                "cache_hits": int(meta["tsbg_hits"]),
                "cache_evictions": int(meta["tsbg_evictions"]),
                "bundles": int(observed["bundles_tsbg"]),
                "accepted_bank_requests": int(observed["scalar_tsbg"]),
            },
        })
        bins.append({
            "stratum": label, "rows": ROWS_PER_TERCILE,
            "density_min_fraction": [members[0]["density"].numerator,
                                     members[0]["density"].denominator],
            "density_upper_median_fraction": [target.numerator, target.denominator],
            "density_max_fraction": [members[-1]["density"].numerator,
                                     members[-1]["density"].denominator],
            "zero_ordinary_request_rows": sum(
                row["ordinary_accepted_bank_requests"] == 0 for row in members),
        })
    need(len(used_sequences) == 3, "representatives are not cross-sequence")
    return {
        "schema": "m2217_ep34_tsbg_matched_power_window_selection_v1",
        "status": "FROZEN_PRE_POWER_SELECTION__NO_POWER_OR_PPA_USED",
        "population": {
            "rows": 2880, "m2051_rows": 1920, "m2067_rows": 960,
            "density": "(ordinary_accepted_bank_requests-tsbg_accepted_bank_requests)/ordinary_accepted_bank_requests; zero when ordinary is zero",
            "sort": "density_then_sequence_sample_layer_token_slot_population",
            "terciles": bins,
        },
        "representative_filter": {
            "standalone_m2018_physical_g48_island": True,
            "ordinary_accepted_bank_requests_positive": True,
            "distinct_sequences": True,
            "uses_cycles_power_energy_or_ppa": False,
        },
        "aggregate_weights": {"low": [1, 3], "median": [1, 3], "high": [1, 3]},
        "selections": selections,
        "identity": {str(path.relative_to(ROOT)): digest
                     for path, digest in EXPECTED.items()},
        "claim_boundary": {
            "selection_only": True, "rtl_or_eda_run": False,
            "power_or_energy_result": False, "full_network": False,
            "system_speedup": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", type=Path)
    args = parser.parse_args()
    result = select()
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.write:
        args.write.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Failure as exc:
        print("M2217_SELECTION_FAIL_CLOSED: " + str(exc),
              file=__import__("sys").stderr)
        raise SystemExit(2)
