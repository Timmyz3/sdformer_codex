#!/opt/anaconda3/bin/python
"""Build the fixed 960-workload M2067 G96/G192 continuation fixture.

CPU-only fixture generation.  It reuses the frozen M2051 extraction semantics
against the double-sealed M1707 capture, selects all forty samples, the eight
FC2 layers above physical G48, and first/middle/last B4 quartets.  No cycle,
RTL, EDA, accuracy, or paper result is produced.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
CAPTURE = HW / (
    "results/m1707_motion_ep34_s2_tsbg_deployment_complete_reduced_"
    "binary_capture_s40_r1_20260901"
)
M2051_BUILDER = HW / (
    "system_simulator/scripts/build_m2051_ep34_tsbg_full40_fixture.py"
)
M2051_META = HW / "tb_m2018/fixtures/m2051_ep34_tsbg_full40_s1920.json"
M2057 = HW / "results/m2057_m2053_ep34_tsbg_full40_missing3_vcs_r1_20260903"
M2065 = HW / (
    "reviews/m2065_m2064_ep34_fc2_exact_continuation_quick_gate_"
    "result_hammer_r1_20260903"
)
M2018 = HW / "rtl_m2018/m2018_c2_tsbg_b4_divfree_fair_scheduler_frontend.sv"
M803 = HW / "rtl_m803/m803_fc2_bundle_to_8bank_channel_split_cutthrough_adapter.sv"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
OUT = HW / "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.memh"
STATS = HW / "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960_stats.memh"
META = HW / "tb_m2018/fixtures/m2067_ep34_fc2_exact_continuation_s960.json"

EXPECTED = {
    M2051_BUILDER: "3a8642914ccad60df89dfdad1b78c375c6d4e4609435c5731357f294d9acf8cf",
    M2051_META: "3ac7048f0a97aeea0ac91627d303f4eea06b8a48bab816468825acfee180ccc5",
    M2018: "96fb355750d50a2f1944f9d27123eef1fc70525a8146b08856884fe09c4bec21",
    M803: "cd264021ee9639c575920e2f01e909273d132f64ee187fe8d25c6ff244c90156",
    DOC359: "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    M2057 / "SHA256SUMS": "f00ab87e69043ed1eaa15980728c3858001122e47e5ff621dcf238eb5aeba971",
    M2057 / "SHA256SUMS.seal.sha256": "3bd2d119e72792f75c636ca82856305151f08d02d15418b675139f504fb51df2",
    M2057 / "result.json": "b4ee4f9cf4d55a4f722f1487ba4bc23948bc3f6a096178fa835d9ed18b50fe2a",
    M2065 / "SHA256SUMS": "aacd2a34a409ba2b38887f7cc0922b1ba1b24d8ca845c51ac5c08980f4dc8ebf",
    M2065 / "SHA256SUMS.seal.sha256": "bb1165db65abb818c07a33f5174a61ac455a1d2afeb8cd4024e477b98747eeb8",
    M2065 / "review.json": "01152ad8d0c7539c4cd885cba8c434e5b98201d7b89c07a3f22c8b2cde1703b6",
}
TARGET_LAYER_IDS = (17, 19, 21, 23, 25, 27, 29, 31)
TARGET_SAMPLES = tuple(range(40))
TOKEN_ROLES = ("first", "middle", "last")
CONTEXTS = 4
PHYSICAL_GROUPS = 48
MAX_LOGICAL_GROUPS = 192
SOURCES = 16
SLICES = 6
LANES = 16
STAT_FIELDS = (
    "sample_id", "layer_id", "token_start", "source_groups",
    "output_tiles", "chunks", "token_role_id", "sequence_id",
    "expected_commits", "integer_checks", "nonzero_codes",
    "negative_codes",
)


def need(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_exact_inputs() -> None:
    for path, expected in EXPECTED.items():
        need(path.is_file() and not path.is_symlink(), f"missing/symlink {path}")
        need(sha256(path) == expected, f"identity drift {path}")
    for sealed in (M2057, M2065):
        manifest = sealed / "SHA256SUMS"
        outer = sealed / "SHA256SUMS.seal.sha256"
        need(outer.read_text().split() == [sha256(manifest), "SHA256SUMS"],
             f"outer seal content {sealed}")
        listed = set()
        for row in manifest.read_text().splitlines():
            digest, name = row.split(None, 1)
            rel = Path(name.lstrip("*"))
            need(not rel.is_absolute() and ".." not in rel.parts,
                 f"unsafe member {sealed}")
            member = sealed / rel
            need(member.is_file() and not member.is_symlink() and
                 sha256(member) == digest, f"sealed member drift {member}")
            listed.add(rel.as_posix())
        actual = {path.relative_to(sealed).as_posix()
                  for path in sealed.rglob("*")
                  if path.is_file() and path.name not in {
                      "SHA256SUMS", "SHA256SUMS.seal.sha256"}}
        need(actual == listed, f"non-exhaustive seal {sealed}")
    review = json.loads((M2065 / "review.json").read_text())
    need(review["status"].startswith("PASS_M2065_") and
         review["severity_counts"]["P0"] == 0 and
         review["severity_counts"]["P1"] == 0 and
         review["authorization"]["next_vcs_source_and_contract"] is True and
         review["authorization"]["vcs_launch"] is False,
         "M2065 source-only authorization drift")


def load_m2051_builder():
    spec = importlib.util.spec_from_file_location("m2067_m2051_builder",
                                                  M2051_BUILDER)
    need(spec is not None and spec.loader is not None, "M2051 import")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def describe_chunk(values: np.ndarray, group_base: int, builder) -> dict:
    need(values.shape == (CONTEXTS, PHYSICAL_GROUPS, SOURCES),
         "physical chunk geometry")
    active = values != 0
    live = active.any(axis=2)
    ordinary_accesses = [group_base + group
                         for context in range(CONTEXTS)
                         for group in range(PHYSICAL_GROUPS)
                         if live[context, group]]
    tsbg_accesses = [group_base + group
                     for group in range(PHYSICAL_GROUPS)
                     for context in range(CONTEXTS)
                     if live[context, group]]
    ordinary = builder.cache_counts(ordinary_accesses)
    tsbg = builder.cache_counts(tsbg_accesses)
    half_live = active.reshape(CONTEXTS, PHYSICAL_GROUPS, 2, 8).any(axis=3)
    return {
        "global_group_base": group_base,
        "first": group_base == 0,
        "intermediate": False,
        "final": False,
        "live_rows": int(live.sum()),
        "issues": int(half_live.sum()) * SLICES,
        "products": int(active.sum()) * SLICES * LANES,
        "ordinary_misses": ordinary[0],
        "ordinary_hits": ordinary[1],
        "ordinary_evictions": ordinary[2],
        "tsbg_misses": tsbg[0],
        "tsbg_hits": tsbg[1],
        "tsbg_evictions": tsbg[2],
    }


def pack_stats(row: dict) -> str:
    value = 0
    for index, field in enumerate(STAT_FIELDS):
        item = int(row[field])
        need(0 <= item < (1 << 32), f"stat overflow {field}")
        value |= item << (32 * index)
    return f"{value:0{len(STAT_FIELDS) * 8}x}\n"


def main() -> int:
    verify_exact_inputs()
    need(not any(path.exists() or path.is_symlink()
                 for path in (OUT, STATS, META)), "M2067 fixture exists")
    builder = load_m2051_builder()
    builder.verify_capture()
    capture_module = builder.load_module()
    layers_payload = json.loads((CAPTURE / "layers.json").read_text())
    samples_payload = json.loads((CAPTURE / "sample_order.json").read_text())
    layer_by_id = {int(row["layer_id"]): row
                   for row in layers_payload["layers"]}
    layers = [layer_by_id[layer_id] for layer_id in TARGET_LAYER_IDS]
    need(all(row["target"] == "FC2" for row in layers), "non-FC2 layer")
    need([int(row["weight_layout"]["source_group_count"])
          for row in layers] == [96] * 6 + [192] * 2, "G96/G192 inventory")
    sample_by_id = {int(row["global_sample_id"]): row
                    for row in samples_payload["samples"]}
    samples = [sample_by_id[index] for index in TARGET_SAMPLES]
    sequences = sorted({row["sequence"] for row in samples})
    need(len(sequences) == 4 and
         Counter(row["sequence"] for row in samples) ==
         Counter({sequence: 10 for sequence in sequences}),
         "sample/sequence inventory")
    sequence_id = {name: index for index, name in enumerate(sequences)}
    found = builder.extract(capture_module, layers)

    fixture_words: list[str] = []
    stats_words: list[str] = []
    rows: list[dict] = []
    for sample in samples:
        sample_id = int(sample["global_sample_id"])
        for layer in layers:
            layer_id = int(layer["layer_id"])
            groups = int(layer["weight_layout"]["source_group_count"])
            output_tiles = int(layer["weight_layout"]["output_tile_count"])
            for role_id, (role, token_start) in enumerate(zip(
                    TOKEN_ROLES,
                    builder.token_starts(int(layer["tokens_per_call"])))):
                raw = found[(sample_id, layer_id, token_start)]
                values = np.zeros((CONTEXTS, MAX_LOGICAL_GROUPS, SOURCES),
                                  dtype=np.int8)
                values[:, :groups, :] = raw.reshape(CONTEXTS, groups, SOURCES)
                need(set(int(item) for item in np.unique(values)).issubset(
                    {-1, 0, 1}), "source domain")
                chunks = []
                for base in range(0, groups, PHYSICAL_GROUPS):
                    chunk = values[:, base:base + PHYSICAL_GROUPS, :]
                    chunks.append(describe_chunk(chunk, base, builder))
                for index, chunk in enumerate(chunks):
                    chunk["first"] = index == 0
                    chunk["intermediate"] = 0 < index < len(chunks) - 1
                    chunk["final"] = index == len(chunks) - 1
                for context in range(CONTEXTS):
                    for group in range(MAX_LOGICAL_GROUPS):
                        source = values[context, group]
                        active = sum(1 << lane for lane, code in enumerate(source)
                                     if code != 0)
                        sign = sum(1 << lane for lane, code in enumerate(source)
                                   if code < 0)
                        need(sign & ~active == 0, "sign without activity")
                        fixture_words.append(f"{sign:04x}{active:04x}\n")
                row = {
                    "slot": len(rows), "sample_id": sample_id,
                    "sequence": sample["sequence"],
                    "sequence_id": sequence_id[sample["sequence"]],
                    "layer_id": layer_id, "token_role": role,
                    "token_role_id": role_id, "token_start": token_start,
                    "source_groups": groups, "physical_source_groups": 48,
                    "output_tiles": output_tiles, "chunks": len(chunks),
                    "global_group_bases": [chunk["global_group_base"]
                                           for chunk in chunks],
                    "expected_commits": output_tiles * CONTEXTS * SLICES,
                    "integer_checks": output_tiles * CONTEXTS * SLICES * LANES,
                    "nonzero_codes": int(np.count_nonzero(values)),
                    "negative_codes": int(np.count_nonzero(values < 0)),
                    "chunk_rows": chunks,
                }
                stats_words.append(pack_stats(row))
                rows.append(row)

    need(len(rows) == 960 and len(fixture_words) ==
         960 * CONTEXTS * MAX_LOGICAL_GROUPS, "fixture population")
    need(sum(row["integer_checks"] for row in rows) == 1_843_200,
         "integer-check population")
    need({tuple(row["global_group_bases"]) for row in rows} ==
         {(0, 48), (0, 48, 96, 144)}, "global group bases")
    OUT.write_text("".join(fixture_words), encoding="ascii")
    STATS.write_text("".join(stats_words), encoding="ascii")
    metadata = {
        "schema": "m2067_ep34_fc2_exact_continuation_fixture_r1_v1",
        "selection_rule": "all 40 M1707 samples; eight G>48 FC2 layers; first/middle/last aligned B4 quartet",
        "selection_uses_performance": False,
        "input_identity": {str(path.relative_to(ROOT)): digest
                           for path, digest in EXPECTED.items()},
        "geometry": {
            "workloads": 960, "samples": 40, "sequences": 4,
            "layers": 8, "g96_layers": 6, "g192_layers": 2,
            "quartets_per_layer_sample": 3, "contexts": CONTEXTS,
            "physical_source_groups": PHYSICAL_GROUPS,
            "max_logical_source_groups": MAX_LOGICAL_GROUPS,
            "sources_per_group": SOURCES,
            "integer_checks": 1_843_200,
        },
        "stat_fields_lsb_first_u32": list(STAT_FIELDS),
        "fixture_sha256": sha256(OUT), "stats_sha256": sha256(STATS),
        "rows": rows,
        "claim_boundary": {
            "real_ep34_activity_and_sign_descriptors": True,
            "directed_integer_weights": True,
            "hardware_weight_values": False,
            "cycle_result": False, "new_vcs": False, "new_eda": False,
            "system_speedup": False, "paper_admitted": False,
        },
    }
    META.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "PASS_M2067_FIXTURE_BUILD_SOURCE_ONLY",
        "workloads": len(rows), "fixture_sha256": sha256(OUT),
        "stats_sha256": sha256(STATS), "metadata_sha256": sha256(META),
        "integer_checks": sum(row["integer_checks"] for row in rows),
        "docs359_sha256": sha256(DOC359),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
