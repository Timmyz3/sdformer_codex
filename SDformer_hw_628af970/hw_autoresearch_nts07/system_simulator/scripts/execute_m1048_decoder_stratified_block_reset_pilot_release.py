#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1048/M1050 one-shot decoder stratified block-reset pilot driver.

The release is deliberately a protocol-calibration pilot.  Candidate and
baseline replay the same frozen A1_OSG work blocks.  The output contains raw
paired cycles, exact-miter identities, coverage, and CI envelopes, but cannot
be interpreted as an A1 speedup, a continuous-row cycle count, or a complete
decoder result.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
RESULTS = HW / "results"
CONTRACT = HW / "contracts/m1048_decoder_stratified_block_reset_pilot_release_contract_r1_20260829.json"
M1041_PATH = HERE / "analyze_m1041_decoder_stratified_block_reset_windows_source_r4.py"
M1041_SHA256 = "09a289835e55313f7dbe06a46064a18bf9ff0caa718cb4efd2ae34097dd4456f"
M1042_DIR = HW / "reviews/m1042_m1041_decoder_stratified_block_reset_windows_source_r4_hammer_r1_20260829"
M1049_DIR = HW / "reviews/m1049_m1048_decoder_stratified_block_reset_pilot_release_hammer_r1_20260829"
DOC359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOC359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SCHEMA = "m1048_decoder_stratified_block_reset_pilot_release_v1"
RESULT_SCHEMA = "m1050_decoder_stratified_block_reset_pilot_result_v1"
RAW_SCHEMA = "m1050_decoder_stratified_block_reset_pilot_raw_windows_v1"
POPULATION_ID = "M699_INTERLAKEN_01_A_S10"
SEQUENCE = "interlaken_01_a"
SAMPLE_ID = 0
TIMESTEP = 0
CONFIG = "A1_OSG"
LAYERS = ("D0", "D2", "D3")
MODULE_BY_LAYER = {"D0": 0, "D2": 2, "D3": 3}
STRATA = ("SOURCE_INIT_CENSUS", "COMPUTE_REGULAR",
          "DEPENDENCY_STRESS", "COMMIT_TAIL")
PILOT = 8
SELECTION_SEED = "M1009_STRATIFIED_WINDOW_R1_20260829"
CAP = 10000
RESULT_NAME = "m1050_m1048_decoder_stratified_block_reset_pilot_r1_20260829"
ATTEMPT_NAME = ".m1050_m1048_decoder_stratified_block_reset_pilot_attempt_consumed"


def require(value, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False).encode("utf-8")).hexdigest()


def strict_json(path: Path):
    def pairs(rows):
        output = {}
        for key, value in rows:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("non-finite JSON token: " + token)))


def atomic_json(path: Path, value: object) -> None:
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp." + str(os.getpid()))
    payload = json.dumps(value, indent=2, sort_keys=True,
                         ensure_ascii=False, allow_nan=False) + "\n"
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pinned(path: Path, expected: str, name: str):
    require(path.is_file() and not path.is_symlink(), name + " absent")
    require(sha256(path) == expected, name + " identity drift")
    spec = importlib.util.spec_from_file_location("m1048_" + name, path)
    require(spec is not None and spec.loader is not None,
            "cannot load " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1041 = load_pinned(M1041_PATH, M1041_SHA256, "m1041")
M785, M890 = M1041.M785, M1041.M890
CompressedTransaction = M1041.CompressedTransaction


def verify_flat_seal(directory: Path) -> Dict[str, str]:
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "sealed directory absent")
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(manifest.is_file() and not manifest.is_symlink() and
            outer.is_file() and not outer.is_symlink(), "seal absent")
    listed = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64,
                "malformed manifest")
        expected, name = fields
        require(name not in listed, "duplicate sealed member")
        target = directory / name
        require(target.is_file() and not target.is_symlink() and
                sha256(target) == expected, "sealed member drift: " + name)
        listed.add(name)
    actual = {str(path.relative_to(directory)) for path in directory.rglob("*")
              if path.is_file() and not path.is_symlink() and
              path.name not in ("SHA256SUMS", "SHA256SUMS.seal.sha256")}
    require(listed == actual, "sealed exact-set drift")
    manifest_sha = sha256(manifest)
    require(outer.read_text(encoding="utf-8") ==
            manifest_sha + "  SHA256SUMS\n", "outer seal drift")
    return {"manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha256(outer)}


def contract_value(path: Path = CONTRACT) -> Mapping[str, object]:
    value = strict_json(path)
    require(value["schema"] == SCHEMA and
            value["status"] == "RELEASE_SOURCE_ONLY__M1049_HAMMER_REQUIRED" and
            value["launch_now"] is False, "M1048 contract drift")
    require(value["workload"] == {
        "population_id": POPULATION_ID, "sequence": SEQUENCE,
        "sample_id": SAMPLE_ID, "timestep": TIMESTEP,
        "config": CONFIG, "layers": list(LAYERS)}, "workload drift")
    require(value["sampling"] == {
        "strata": list(STRATA), "source_census": 1,
        "pilot_per_noncensus_stratum": PILOT,
        "selection_seed": SELECTION_SEED,
        "window_expanded_request_cap": CAP,
        "selection_frozen_before_cycle_replay": True}, "sampling drift")
    require(value["pair_role"] ==
            "SELF_MATCHED_A1_OSG_PROTOCOL_CALIBRATION__NOT_SPEEDUP",
            "pair role drift")
    require(all(value["claim_boundary"][key] is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "local_speedup", "continuous_row_cycles",
                 "transaction_ratio_is_speedup", "d1_scheduled",
                 "eda_gpu_remote_used")), "claim boundary expanded")
    return value


def validate_source(contract_path: Path, runner: Path) -> Dict[str, object]:
    value = contract_value(contract_path)
    require(sha256(M1041_PATH) == value["source_identity"]["m1041"]["sha256"],
            "M1041 source drift")
    require(sha256(Path(__file__).resolve()) ==
            value["source_identity"]["driver"]["sha256"], "driver drift")
    require(Path(runner).is_file() and not Path(runner).is_symlink() and
            sha256(Path(runner)) == value["source_identity"]["runner"]["sha256"],
            "runner drift")
    require(sha256(DOC359) == DOC359_SHA256, "docs359 drift")
    m1042 = verify_flat_seal(M1042_DIR)
    require(sha256(M1042_DIR / "review.json") ==
            value["authority"]["m1042"]["review_sha256"] and
            m1042["manifest_sha256"] ==
            value["authority"]["m1042"]["manifest_sha256"] and
            m1042["outer_seal_file_sha256"] ==
            value["authority"]["m1042"]["outer_seal_file_sha256"],
            "M1042 authority drift")
    m1042_review = strict_json(M1042_DIR / "review.json")
    require(m1042_review["status"] ==
            "PASS_M1042_M1041_R4_INDEPENDENT_SOURCE_HAMMER__GO_EXECUTION_RELEASE_SOURCE_ONLY" and
            m1042_review["authorization"]["write_separate_execution_release_source"] is True and
            m1042_review["authorization"]["execute_real_windows"] is False,
            "M1042 scope drift")
    m785_contract = HW / value["source_identity"]["m785_contract"]["path"]
    require(sha256(m785_contract) ==
            value["source_identity"]["m785_contract"]["sha256"],
            "M785 contract drift")
    M785.validate_source_contract(REPO, m785_contract)
    payload = HW / value["payload_authority"]["directory"]
    sealed = M785.verify_sealed_directory(payload)
    require(sha256(payload / "manifest.json") ==
            value["payload_authority"]["manifest_sha256"] and
            sealed["outer_seal_file_sha256"] ==
            value["payload_authority"]["outer_seal_file_sha256"],
            "M699 payload drift")
    review_dir = HW / value["payload_authority"]["m705_review_directory"]
    review_seal = verify_flat_seal(review_dir)
    require(sha256(review_dir / "review.json") ==
            value["payload_authority"]["m705_review_sha256"] and
            review_seal["outer_seal_file_sha256"] ==
            value["payload_authority"]["m705_outer_seal_file_sha256"],
            "M705 review drift")
    return {
        "status": "PASS_M1048_RELEASE_SOURCE_VALIDATION__NO_EXECUTION",
        "contract_sha256": sha256(contract_path),
        "driver_sha256": sha256(Path(__file__).resolve()),
        "runner_sha256": sha256(Path(runner)),
        "launch_now": False, "real_payload_opened": False,
        "real_window_execution": False, "eda_gpu_remote_used": False,
    }


def validate_authority(expected_review_sha: str,
                       expected_manifest_sha: str,
                       expected_outer_sha: str) -> Dict[str, object]:
    sealed = verify_flat_seal(M1049_DIR)
    require(sha256(M1049_DIR / "review.json") == expected_review_sha and
            sealed["manifest_sha256"] == expected_manifest_sha and
            sealed["outer_seal_file_sha256"] == expected_outer_sha,
            "M1049 caller-pinned authority drift")
    value = strict_json(M1049_DIR / "review.json")
    require(value["status"] ==
            "PASS_M1049_M1048_DECODER_STRATIFIED_PILOT_RELEASE_HAMMER__GO_ONE_M1050_ATTEMPT" and
            value["authorization"]["one_m1050_attempt"] is True and
            value["authorization"]["execute_full_row"] is False and
            value["authorization"]["eda_gpu_remote"] is False,
            "M1049 authorization drift")
    return {"status": "PASS_M1049_CALLER_PINNED_AUTHORITY",
            "review_sha256": expected_review_sha,
            "manifest_sha256": expected_manifest_sha,
            "outer_seal_file_sha256": expected_outer_sha}


def safe_runtime_path(path: Path, role: str) -> Path:
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == RESULTS.resolve(),
            role + " parent drift")
    require(not path.is_symlink(), role + " symlink forbidden")
    if role == "attempt":
        require(path.name == ATTEMPT_NAME, "attempt namespace drift")
    elif role == "result":
        require(path.name == RESULT_NAME, "result namespace drift")
    elif role == "work":
        require(path.name.startswith("." + RESULT_NAME + ".work."),
                "work namespace drift")
    elif role == "quarantine":
        require(path.name.startswith(RESULT_NAME + ".failed_or_incomplete."),
                "quarantine namespace drift")
    else:
        raise RuntimeError("unknown runtime path role")
    return path


def consume_attempt(path: Path, runner: Path, contract_sha: str,
                    authority: Mapping[str, str]) -> Dict[str, object]:
    path = safe_runtime_path(path, "attempt")
    require(not path.exists(), "canonical M1050 attempt already consumed")
    os.mkdir(path, 0o700)
    value = {
        "schema": "m1050_decoder_stratified_pilot_attempt_v1",
        "status": "M1050_CANONICAL_ATTEMPT_CONSUMED_BEFORE_REAL_PAYLOAD",
        "attempt_consumed_unix_ns": time.time_ns(),
        "runner_path": str(Path(runner).resolve()),
        "runner_sha256": sha256(Path(runner)),
        "contract_sha256": contract_sha,
        "m1049_authority": dict(authority),
        "real_payload_opened_at_consumption": False,
        "real_window_execution_started_at_consumption": False,
        "paper_citable": False,
    }
    atomic_json(path / "attempt.json", value)
    return value


def _context():
    contract = contract_value()
    m785_path = HW / contract["source_identity"]["m785_contract"]["path"]
    m785_contract = M785.strict_json(m785_path)
    entry = m785_contract["inputs"]["secondary_m699"]
    payload_root = HW / entry["directory"]
    manifest = M785.strict_json(payload_root / "manifest.json")
    records = M785.normalized_population_records(manifest, POPULATION_ID)
    mapper_row = m785_contract["inputs"]["m672_mapper"]
    mapper = M785.load_pinned_module(HW / mapper_row["path"],
                                     mapper_row["sha256"], "m1048_mapper")
    m712, m722, storage = (m785_contract["inputs"][name] for name in
                           ("m712_oracle", "m722r2_oracle",
                            "m785_storage_oracle"))
    oracles = M785.load_pinned_oracles(
        HW / m712["path"], m712["sha256"],
        HW / m722["path"], m722["sha256"],
        HW / storage["path"], storage["sha256"])
    return payload_root, records, mapper, oracles


def select_record(records: Sequence[Mapping[str, object]], layer: str):
    matches = [row for row in records
               if row["sequence"] == SEQUENCE and
               int(row["sample_id"]) == SAMPLE_ID and
               int(row["module_index"]) == MODULE_BY_LAYER[layer]]
    require(len(matches) == 1, "M699 selected record is not unique: " + layer)
    row = matches[0]
    require(row["route"] == "EXACT_BINARY_BITPACK", "nonexact D0/D2/D3 route")
    return row


def _remapped_max_fanin(body: Sequence[CompressedTransaction]) -> int:
    produced = set()
    maximum = 0
    for ordinal, tx in enumerate(body):
        mapped = set()
        for dependency in tx.dependency_tokens:
            mapped.add(dependency if dependency in produced else "BOUNDARY_READY")
        if not mapped and ordinal == 0:
            mapped.add("BOUNDARY_READY")
        maximum = max(maximum, len(mapped))
        terminal = M890.terminal_token(tx)
        if terminal:
            produced.add(terminal)
    return maximum


def _metadata(layer: str, role: str, ordinal: int,
              body: Sequence[CompressedTransaction]) -> Dict[str, object]:
    require(body, "empty semantic block")
    expanded = sum(int(tx.count) for tx in body)
    require(expanded + 3 <= CAP, "semantic block exceeds reset window cap")
    source = role == "SOURCE"
    commit = sum(int(tx.count) for tx in body if tx.kind == "commit")
    compute = sum(int(tx.count) for tx in body if tx.kind == "compute")
    psum_external = sum(int(tx.count) for tx in body
                        if ":res" in tx.transaction_id and
                        tx.kind in ("external_read", "external_write"))
    refill_external = sum(int(tx.count) for tx in body
                          if "weight_refill_external" in tx.transaction_id)
    max_fanin = _remapped_max_fanin(body)
    if role == "DEPENDENCY":
        require(psum_external or refill_external or max_fanin >= 3,
                "dependency phase does not meet frozen stress definition")
    row = {
        "block_id": "M1048:{}:{}:{:09d}".format(layer, role, ordinal),
        "source_init": source,
        "commit_count": commit,
        "psum_external_move_count": psum_external,
        "weight_refill_external_count": refill_external,
        "max_dependency_fan_in": max_fanin,
        "compute_count": compute,
        "layer": layer,
        "sample_id": SAMPLE_ID,
        "timestep": TIMESTEP,
        "destination": "PHASE_SPLIT_BLOCK",
        "output_block": "PHASE_SPLIT_BLOCK",
        "subblock": ordinal,
        "population_id": POPULATION_ID,
        "config": CONFIG,
        "source_service_group_count": 1 if role == "COMPUTE" else 0,
        "dense_commit_address_count": commit,
        "compressed_transaction_count": len(body),
        "expanded_request_count": expanded,
    }
    M1041.validate_metadata_row(row)
    expected = {"SOURCE": "SOURCE_INIT_CENSUS",
                "COMPUTE": "COMPUTE_REGULAR",
                "DEPENDENCY": "DEPENDENCY_STRESS",
                "COMMIT": "COMMIT_TAIL"}[role]
    require(M1041.classify_stratum(row) == expected,
            "semantic phase classified into wrong stratum")
    return row


GROUP_RE = re.compile(r":g([0-9]+):")


def iter_semantic_blocks(transactions: Iterable[CompressedTransaction],
                         layer: str):
    iterator = iter(transactions)
    source = next(iterator, None)
    require(source is not None and "source_fetch" in source.transaction_id,
            "source-init transaction absent")
    counters = {"SOURCE": 0, "COMPUTE": 0, "DEPENDENCY": 0, "COMMIT": 0}
    yield _metadata(layer, "SOURCE", 0, [source]), [source]
    pending: List[CompressedTransaction] = []
    dependency: List[CompressedTransaction] = []
    compute: List[CompressedTransaction] = []
    current_group = None
    in_compute = False
    for tx in iterator:
        if tx.kind == "commit":
            require(current_group is None and not dependency and not compute,
                    "commit arrived inside source-service group")
            body = pending + [tx]
            pending = []
            counters["COMMIT"] += 1
            yield _metadata(layer, "COMMIT", counters["COMMIT"], body), body
            continue
        match = GROUP_RE.search(tx.transaction_id)
        if match:
            group = int(match.group(1))
            if current_group is None:
                current_group = group
                dependency = pending
                pending = []
            require(group == current_group, "interleaved service-group identity")
            if ":psum_read" in tx.transaction_id:
                require(not in_compute, "duplicate psum-read phase")
                in_compute = True
                compute = [tx]
            elif in_compute:
                compute.append(tx)
                if ":psum_write" in tx.transaction_id:
                    require(dependency, "empty dependency phase")
                    counters["DEPENDENCY"] += 1
                    yield _metadata(layer, "DEPENDENCY",
                                    counters["DEPENDENCY"], dependency), dependency
                    counters["COMPUTE"] += 1
                    yield _metadata(layer, "COMPUTE",
                                    counters["COMPUTE"], compute), compute
                    dependency, compute = [], []
                    current_group, in_compute = None, False
            else:
                dependency.append(tx)
        else:
            if current_group is None:
                pending.append(tx)
            else:
                require(not in_compute,
                        "unidentified transaction inside compute phase")
                dependency.append(tx)
    require(current_group is None and not dependency and not compute and
            not pending, "unclosed semantic block at end of row")


def _selection_key(block_id: str) -> str:
    return canonical_sha([SELECTION_SEED, block_id])


def select_streaming(blocks, layer: str):
    selected = {stratum: [] for stratum in STRATA}
    population = {stratum: 0 for stratum in STRATA}
    index_digest = hashlib.sha256()
    transaction_digest = hashlib.sha256()
    generated_transactions = assigned_transactions = 0
    identifiers = set()
    for metadata, body in blocks:
        stratum = M1041.classify_stratum(metadata)
        population[stratum] += 1
        block_id = metadata["block_id"]
        require(block_id not in identifiers, "duplicate block identity")
        identifiers.add(block_id)
        index_digest.update((json.dumps(metadata, sort_keys=True,
                                        separators=(",", ":")) + "\n").encode())
        tx_ids = [tx.transaction_id for tx in body]
        transaction_digest.update((json.dumps(
            [block_id, tx_ids], sort_keys=True, separators=(",", ":")) +
            "\n").encode())
        generated_transactions += len(body)
        assigned_transactions += len(body)
        limit = 1 if stratum == "SOURCE_INIT_CENSUS" else PILOT
        selected[stratum].append((_selection_key(block_id), block_id,
                                  dict(metadata), list(body)))
        selected[stratum].sort(key=lambda row: (row[0], row[1]))
        if len(selected[stratum]) > limit:
            selected[stratum].pop()
    require(generated_transactions == assigned_transactions and
            all(population[stratum] >=
                (1 if stratum == "SOURCE_INIT_CENSUS" else PILOT)
                for stratum in STRATA),
            "pilot stratum population/transaction conservation failure")
    return {
        "layer": layer,
        "population": population,
        "selected": selected,
        "block_population_index_sha256": index_digest.hexdigest(),
        "transaction_assignment_census_sha256": transaction_digest.hexdigest(),
        "generated_compressed_transactions": generated_transactions,
        "assigned_compressed_transactions": assigned_transactions,
        "selection_frozen_before_cycle_replay": True,
    }


def replay_layer(layer: str, record: Mapping[str, object], payload_root: Path,
                 mapper, oracles):
    stream = M785.iter_record_transactions(
        mapper, record, payload_root, POPULATION_ID, CONFIG, TIMESTEP, oracles)
    selection = select_streaming(iter_semantic_blocks(stream, layer), layer)
    selection_sha = canonical_sha({
        stratum: [[row[0], row[1], row[2]] for row in rows]
        for stratum, rows in selection["selected"].items()})
    windows = []
    exact_mismatch_count = 0
    raw_ci = []
    source_cycles = None
    for stratum in STRATA:
        candidate_cycles, baseline_cycles = [], []
        for _key, block_id, metadata, body in selection["selected"][stratum]:
            spec = M1041.WindowSpec(
                block_id, layer, stratum,
                selection["population"][stratum], SAMPLE_ID, TIMESTEP)
            pair = M1041.paired_replay(body, body, spec)
            require(pair["candidate_cycles"] == pair["baseline_cycles"],
                    "self-matched protocol calibration diverged")
            candidate_cycles.append(int(pair["candidate_cycles"]))
            baseline_cycles.append(int(pair["baseline_cycles"]))
            windows.append({
                "block_id": block_id,
                "window_identity_sha256": pair["window_identity_sha256"],
                "stratum": stratum,
                "metadata": metadata,
                "body_transaction_ids_sha256": canonical_sha(
                    [tx.transaction_id for tx in body]),
                "body_compressed_transaction_count": len(body),
                "body_expanded_request_count": sum(int(tx.count) for tx in body),
                "candidate_cycles": int(pair["candidate_cycles"]),
                "baseline_cycles": int(pair["baseline_cycles"]),
                "exact_mismatch_count": 0,
                "candidate_exact": pair["candidate"],
                "baseline_exact": pair["baseline"],
                "candidate_reset": pair["candidate_reset"],
                "baseline_reset": pair["baseline_reset"],
                "paired_reset_semantics_sha256":
                    pair["paired_reset_semantics_sha256"],
                "pair_role": "SELF_MATCHED_A1_OSG_PROTOCOL_CALIBRATION",
                "transaction_ratio_is_speedup": False,
            })
        if stratum == "SOURCE_INIT_CENSUS":
            require(len(candidate_cycles) == 1, "source census cardinality drift")
            source_cycles = (candidate_cycles[0], baseline_cycles[0])
        else:
            require(len(candidate_cycles) == PILOT,
                    "noncensus pilot cardinality drift")
            raw_ci.append({
                "stratum": stratum,
                "population_blocks": selection["population"][stratum],
                "candidate_cycles": candidate_cycles,
                "baseline_cycles": baseline_cycles,
            })
    require(source_cycles is not None, "source census missing")
    envelope = M1041.estimate_paired_totals(
        raw_ci, fixed_candidate=source_cycles[0],
        fixed_baseline=source_cycles[1])
    M1041.validate_publication_envelope(envelope)
    coverage = [{
        "stratum": stratum,
        "population_blocks": selection["population"][stratum],
        "sample_blocks": len(selection["selected"][stratum]),
        "finite_population_fraction":
            len(selection["selected"][stratum]) /
            selection["population"][stratum],
    } for stratum in STRATA]
    return {
        "layer": layer,
        "record_identity": {
            "population_id": POPULATION_ID, "sequence": record["sequence"],
            "sample_id": int(record["sample_id"]),
            "module_index": int(record["module_index"]),
            "timestep": TIMESTEP, "config": CONFIG,
            "route": record["route"],
            "relative_path": record["relative_path"],
            "packed_sha256": record["packed_sha256"],
        },
        "selection_identity_sha256": selection_sha,
        "selection_frozen_before_cycle_replay": True,
        "block_population_index_sha256":
            selection["block_population_index_sha256"],
        "transaction_assignment_census_sha256":
            selection["transaction_assignment_census_sha256"],
        "generated_compressed_transactions":
            selection["generated_compressed_transactions"],
        "assigned_compressed_transactions":
            selection["assigned_compressed_transactions"],
        "coverage": coverage,
        "source_census_cycles": {
            "candidate": source_cycles[0], "baseline": source_cycles[1]},
        "ci_raw_inputs": raw_ci,
        "ci_publication_envelope": envelope,
        "windows": windows,
        "exact_mismatch_count": exact_mismatch_count,
        "local_speedup_admitted": False,
        "continuous_row_cycles": False,
    }


def run_pilot(work: Path, attempt: Path,
              authority: Mapping[str, str]) -> Dict[str, object]:
    work = safe_runtime_path(work, "work")
    attempt = safe_runtime_path(attempt, "attempt")
    require(work.is_dir() and not work.is_symlink() and
            not any(work.iterdir()), "work directory not fresh/empty")
    require(attempt.is_dir() and not attempt.is_symlink() and
            {path.name for path in attempt.iterdir()} == {"attempt.json"},
            "canonical attempt receipt absent/drifted")
    attempt_value = strict_json(attempt / "attempt.json")
    require(attempt_value["status"] ==
            "M1050_CANONICAL_ATTEMPT_CONSUMED_BEFORE_REAL_PAYLOAD" and
            attempt_value["contract_sha256"] == sha256(CONTRACT) and
            attempt_value["m1049_authority"] == dict(authority) and
            attempt_value["real_payload_opened_at_consumption"] is False and
            attempt_value["real_window_execution_started_at_consumption"] is False,
            "canonical attempt authority drift")
    payload_root, records, mapper, oracles = _context()
    layers = []
    for layer in LAYERS:
        layers.append(replay_layer(
            layer, select_record(records, layer), payload_root, mapper, oracles))
    raw = {
        "schema": RAW_SCHEMA,
        "status": "PASS_M1050_RAW_STRATIFIED_BLOCK_RESET_WINDOWS__RESULT_HAMMER_REQUIRED",
        "workload": {"population_id": POPULATION_ID, "sequence": SEQUENCE,
                     "sample_id": SAMPLE_ID, "timestep": TIMESTEP,
                     "config": CONFIG, "layers": list(LAYERS)},
        "pair_role": "SELF_MATCHED_A1_OSG_PROTOCOL_CALIBRATION__NOT_SPEEDUP",
        "layers": layers,
        "exact_mismatch_count": sum(row["exact_mismatch_count"] for row in layers),
        "d1": {"status": "DIAGNOSTIC_ONLY_NO_GENERATOR_NO_SCHEDULER_CALL",
               "scheduled": False, "exact_binary": False,
               "decoder_numeric_equivalence": False},
        "claim_boundary": {
            "paper_citable": False, "decoder_complete": False,
            "table_a_row": False, "system_speedup": False,
            "local_speedup": False, "continuous_row_cycles": False,
            "transaction_ratio_is_speedup": False,
        },
    }
    require(raw["exact_mismatch_count"] == 0, "exact mismatch observed")
    atomic_json(work / "raw_windows.json", raw)
    result = {
        "schema": RESULT_SCHEMA,
        "status": "PASS_M1050_DECODER_STRATIFIED_BLOCK_RESET_PILOT__INDEPENDENT_RESULT_HAMMER_REQUIRED",
        "result_role": "DIAGNOSTIC_PROTOCOL_CALIBRATION_ONLY",
        "raw_windows_sha256": sha256(work / "raw_windows.json"),
        "layers": [{
            "layer": row["layer"],
            "selection_identity_sha256": row["selection_identity_sha256"],
            "block_population_index_sha256": row["block_population_index_sha256"],
            "transaction_assignment_census_sha256":
                row["transaction_assignment_census_sha256"],
            "generated_compressed_transactions":
                row["generated_compressed_transactions"],
            "assigned_compressed_transactions":
                row["assigned_compressed_transactions"],
            "coverage": row["coverage"],
            "source_census_cycles": row["source_census_cycles"],
            "ci_publication_envelope": row["ci_publication_envelope"],
            "window_count": len(row["windows"]),
            "exact_mismatch_count": row["exact_mismatch_count"],
        } for row in layers],
        "total_window_count": sum(len(row["windows"]) for row in layers),
        "exact_mismatch_count": 0,
        "d1_scheduled": False,
        "paper_citable": False,
        "decoder_complete": False,
        "table_a_row": False,
        "system_speedup": False,
        "local_speedup": False,
        "continuous_row_cycles": False,
        "eda_gpu_remote_used": False,
        "next_gate": "Independent receipt-blind M1051 result hammer",
    }
    atomic_json(work / "result.json", result)
    (work / "RUN_COMPLETE.txt").write_text(
        result["status"] + "\n", encoding="utf-8")
    return result


def assemble(work: Path) -> Dict[str, object]:
    work = safe_runtime_path(work, "work")
    require(work.is_dir() and not work.is_symlink(), "work absent")
    expected = {"raw_windows.json", "result.json", "RUN_COMPLETE.txt"}
    actual = {path.name for path in work.iterdir() if path.is_file()}
    require(actual == expected and not any(path.is_dir() for path in work.iterdir()),
            "work exact-set drift before seal")
    result = strict_json(work / "result.json")
    raw = strict_json(work / "raw_windows.json")
    require(result["schema"] == RESULT_SCHEMA and raw["schema"] == RAW_SCHEMA and
            result["raw_windows_sha256"] == sha256(work / "raw_windows.json") and
            result["exact_mismatch_count"] == raw["exact_mismatch_count"] == 0 and
            result["d1_scheduled"] is False and
            all(result[key] is False for key in
                ("paper_citable", "decoder_complete", "table_a_row",
                 "system_speedup", "local_speedup", "continuous_row_cycles",
                 "eda_gpu_remote_used")), "result boundary drift")
    lines = []
    for name in sorted(expected):
        lines.append(sha256(work / name) + "  " + name)
    (work / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest_sha = sha256(work / "SHA256SUMS")
    (work / "SHA256SUMS.seal.sha256").write_text(
        manifest_sha + "  SHA256SUMS\n", encoding="utf-8")
    return {"status": "PASS_M1050_WORK_SEALED",
            "manifest_sha256": manifest_sha,
            "outer_seal_file_sha256": sha256(work / "SHA256SUMS.seal.sha256")}


def publish(work: Path, result: Path) -> Dict[str, object]:
    work = safe_runtime_path(work, "work")
    result = safe_runtime_path(result, "result")
    require(work.is_dir() and not result.exists(), "publish namespace occupied")
    verify_flat_seal(work)
    os.replace(work, result)
    return {"status": "PASS_M1050_ATOMIC_RESULT_PUBLISHED",
            "result": str(result)}


def quarantine(work: Path, quarantine: Path, return_code: int) -> Dict[str, object]:
    work = safe_runtime_path(work, "work")
    quarantine = safe_runtime_path(quarantine, "quarantine")
    require(work.is_dir() and not quarantine.exists(), "quarantine path drift")
    os.replace(work, quarantine)
    atomic_json(quarantine / "FAILURE.json", {
        "schema": "m1050_decoder_stratified_pilot_failure_v1",
        "status": "FAILED_OR_INCOMPLETE_DO_NOT_CITE",
        "return_code": int(return_code), "paper_citable": False})
    return {"status": "PASS_M1050_FAILURE_QUARANTINED",
            "quarantine": str(quarantine)}


def self_test() -> Dict[str, object]:
    pop, cfg = "M1048_SYNTHETIC", CONFIG
    def tx(identifier, kind, deps=(), produces=True):
        return CompressedTransaction(
            transaction_id=identifier, population_id=pop, config=cfg,
            kind=kind, base_address=(1 << 60), address_stride_bytes=1,
            count=1, bank_pattern=(0,), width_bytes=1,
            dependency_tokens=tuple(deps),
            produces_token_prefix=(identifier + ":done" if produces else ""))
    source = tx(pop + ":A1_OSG:m0:t0:source_fetch", "external_read")
    source_done = M890.terminal_token(source)
    rows = [source]
    for group in range(12):
        desc = tx(pop + ":A1_OSG:m0:t0:g{}:osg_header".format(group),
                  "external_read", (source_done,))
        lane = tx(pop + ":A1_OSG:m0:t0:g{}:k1_descriptor0".format(group),
                  "external_read", (source_done,))
        weight = tx(pop + ":A1_OSG:m0:t0:g{}:k1_weight0".format(group),
                    "weight_read", (M890.terminal_token(desc),
                                    M890.terminal_token(lane),
                                    "external-ready"))
        read = tx(pop + ":A1_OSG:m0:t0:g{}:psum_read".format(group),
                  "psum_read", ("external-psum",))
        compute = tx(pop + ":A1_OSG:m0:t0:g{}:compute".format(group),
                     "compute", (M890.terminal_token(weight),
                                 M890.terminal_token(read)))
        write = tx(pop + ":A1_OSG:m0:t0:g{}:psum_write".format(group),
                   "psum_write", (M890.terminal_token(compute),))
        rows.extend((desc, lane, weight, read, compute, write))
    for commit in range(12):
        rows.append(tx(pop + ":A1_OSG:m0:t0:commit{}".format(commit),
                       "commit", ("external-final",)))
    blocks = list(iter_semantic_blocks(rows, "D0"))
    index = [metadata for metadata, _body in blocks]
    selected = select_streaming(iter(blocks), "D0")
    for stratum in STRATA:
        expected = M1041.deterministic_select(
            index, stratum, 1 if stratum == "SOURCE_INIT_CENSUS" else PILOT)
        require([row["block_id"] for row in expected] ==
                [row[2]["block_id"] for row in selected["selected"][stratum]],
                "streaming top-k selection diverges from M1041")
    require(len(blocks) == 37 and
            selected["generated_compressed_transactions"] == len(rows) and
            selected["assigned_compressed_transactions"] == len(rows),
            "synthetic transaction conservation drift")
    return {
        "status": "PASS_M1048_RELEASE_SMALL_SYNTHETIC_SELFTEST__NO_REAL_PAYLOAD",
        "semantic_blocks": len(blocks),
        "compressed_transactions": len(rows),
        "streaming_selection_matches_m1041": True,
        "real_payload_opened": False, "real_window_execution": False,
        "eda_gpu_remote_used": False,
    }


def main():
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--validate-source", action="store_true")
    modes.add_argument("--validate-authority", action="store_true")
    modes.add_argument("--consume-attempt", action="store_true")
    modes.add_argument("--run-pilot", action="store_true")
    modes.add_argument("--assemble", action="store_true")
    modes.add_argument("--publish", action="store_true")
    modes.add_argument("--quarantine", action="store_true")
    modes.add_argument("--self-test", action="store_true")
    parser.add_argument("--contract", type=Path, default=CONTRACT)
    parser.add_argument("--runner", type=Path)
    parser.add_argument("--expected-contract-sha")
    parser.add_argument("--expected-review-sha")
    parser.add_argument("--expected-manifest-sha")
    parser.add_argument("--expected-outer-sha")
    parser.add_argument("--attempt", type=Path)
    parser.add_argument("--work", type=Path)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--quarantine-path", type=Path)
    parser.add_argument("--return-code", type=int, default=1)
    args = parser.parse_args()
    if args.validate_source:
        require(args.runner is not None and args.expected_contract_sha,
                "source validation authority absent")
        require(sha256(args.contract) == args.expected_contract_sha,
                "caller-pinned contract SHA drift")
        output = validate_source(args.contract, args.runner)
    elif args.validate_authority:
        require(all((args.expected_review_sha, args.expected_manifest_sha,
                     args.expected_outer_sha)), "M1049 pins absent")
        output = validate_authority(args.expected_review_sha,
                                    args.expected_manifest_sha,
                                    args.expected_outer_sha)
    elif args.consume_attempt:
        require(args.attempt is not None and args.runner is not None and
                args.expected_contract_sha and
                all((args.expected_review_sha, args.expected_manifest_sha,
                     args.expected_outer_sha)), "attempt authority absent")
        validate_authority(args.expected_review_sha,
                           args.expected_manifest_sha,
                           args.expected_outer_sha)
        output = consume_attempt(args.attempt, args.runner,
            args.expected_contract_sha, {
                "review_sha256": args.expected_review_sha,
                "manifest_sha256": args.expected_manifest_sha,
                "outer_seal_file_sha256": args.expected_outer_sha})
    elif args.run_pilot:
        require(args.work is not None and args.attempt is not None and
                all((args.expected_review_sha, args.expected_manifest_sha,
                     args.expected_outer_sha)), "run authority absent")
        authority = validate_authority(args.expected_review_sha,
                                       args.expected_manifest_sha,
                                       args.expected_outer_sha)
        output = run_pilot(args.work, args.attempt, {
            "review_sha256": args.expected_review_sha,
            "manifest_sha256": args.expected_manifest_sha,
            "outer_seal_file_sha256": args.expected_outer_sha})
    elif args.assemble:
        require(args.work is not None and
                all((args.expected_review_sha, args.expected_manifest_sha,
                     args.expected_outer_sha)), "assemble authority absent")
        validate_authority(args.expected_review_sha,
                           args.expected_manifest_sha,
                           args.expected_outer_sha)
        output = assemble(args.work)
    elif args.publish:
        require(args.work is not None and args.result is not None,
                "publish paths absent")
        require(all((args.expected_review_sha, args.expected_manifest_sha,
                     args.expected_outer_sha)), "publish authority absent")
        validate_authority(args.expected_review_sha,
                           args.expected_manifest_sha,
                           args.expected_outer_sha)
        output = publish(args.work, args.result)
    elif args.quarantine:
        require(args.work is not None and args.quarantine_path is not None,
                "quarantine paths absent")
        output = quarantine(args.work, args.quarantine_path, args.return_code)
    else:
        output = self_test()
    print(json.dumps(output, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
