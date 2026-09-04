#!/usr/bin/env python3
"""Source-only ep34 decoder address-timed replay successor.

M1539 is the executable source successor authorized by M1536.  It admits only
the three non-product configurations from M1525 and binds the actual M1521
positive-plane population plus the M1527 actual-result hammer.  The weighted
``PRODUCT_CAPTURE_TYPED_K8`` branch is rejected in two independent places and
remains blocked by M1526.

The module contains a deterministic request scheduler and an exact K3/S2/P1/
OP1 contributor constructor suitable for a later one-shot production runner.
This file itself accepts only ``--describe``, ``--preflight`` and
``--synthetic-self-test``.  It cannot publish a production result.

Python syntax is deliberately compatible with CPython 3.6.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


HERE = Path(__file__).resolve().parent
HW = HERE.parent.parent
REPO = HW.parent
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
M1521_ROOT = HW / "results/m1521_ep34_decoder_positive_planes_s30_c120_r1_20260831"
M1521_MANIFEST = M1521_ROOT / "manifest.json"
M1527 = HW / "reviews/m1527_m1521_ep34_decoder_positive_plane_actual_result_hammer_r1_20260831"
M1525_SOURCE = HERE / "build_m1525_ep34_decoder_multibaseline_replay_successor_source.py"
M1525_CONTRACT = HW / "contracts/m1525_ep34_decoder_multibaseline_replay_successor_source_contract_r1_20260831.json"
M1536 = HW / "reviews/m1536_m1525_ep34_decoder_multibaseline_replay_source_independent_hammer_r1_20260831"
M1526_CONTRACT = HW / "contracts/m1526_ep34_decoder_int8_numeric_bridge_gate_source_contract_r1_20260831.json"

SCHEMA = "m1539_ep34_decoder_nonproduct_address_timed_replay_successor_source_r1_v1"
STATUS = "SOURCE_ONLY__THREE_NONPRODUCT_ADDRESS_TIMED_CONFIGS__NO_PRODUCTION"
CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
M1521_MANIFEST_SHA256 = "969b786bf66323174bc734630384ae03abab5b81a4fc59000b113e0b7a5d8304"
M1521_OUTER_FILE_SHA256 = "60a172e5cd041bcdd0ca38db87250090c48c66e655364b332868fb40a1b182f2"
M1527_REVIEW_SHA256 = "366068b725a16c42fc69adc29c463ce909b0f528f6a31c36eb25e6914366c714"
M1527_OUTER_FILE_SHA256 = "37841dfbd4f6d83d4004efdfa5e80e011396c9581fc8e2c0985ffec85274bb22"
M1525_SOURCE_SHA256 = "d52fa8b4d7a0f4395a4214f4209f449fcfb404fab7e90f12f179efe33405141a"
M1525_CONTRACT_SHA256 = "9b1a1d383b46aca7cdfa1b1085432848849f3e6f235fe594ceb4bf068a9671b9"
M1536_REVIEW_SHA256 = "dbd02a05c101a3a65464d08edff32bfc198feb393a211f4f698c52919b1de5cf"
M1536_OUTER_FILE_SHA256 = "51d60983e51f54867a95ad907ddcdc27a40f6e2819a1265814c89086e1391f97"
M1526_CONTRACT_SHA256 = "529151b5d4b682f8cde483678f853b8ea01f1364e48106b2ca8867d1de477a36"
M1525_RESOURCE_MANIFEST_SHA256 = "64661d825ee8ddbdccad9c3e09ca5e41c5ea9cfc75bcea394667dcfd91b4de10"

CONFIGS = (
    "DENSE_TYPED_K8",
    "BIT_EQUAL_SERVICE_K1X8",
    "BIT_TYPED_K8",
)
FORBIDDEN_CONFIG = "PRODUCT_CAPTURE_TYPED_K8"
MODULES = tuple("sttmultires_unet.decoders.{}.deconv.0".format(i)
                for i in range(4))
INPUT_SHAPES = (
    (10, 1, 1536, 15, 20),
    (10, 1, 770, 30, 40),
    (10, 1, 386, 60, 80),
    (10, 1, 194, 120, 160),
)
GEOMETRY = (
    (1536, 384, 15, 20, 30, 40),
    (770, 192, 30, 40, 60, 80),
    (386, 96, 60, 80, 120, 160),
    (194, 96, 120, 160, 240, 320),
)
SCALE_WORDS = (0x3F7FFD6B, 0x3F7FFFA0, 0x3F800000, 0x3F800000)
PSUM_VECTOR_BYTES = 96 * 3
OUTPUT_COMMIT_BYTES = 96 * 4
WEIGHT_SOURCE_TILE = 16
WEIGHT_TILE_BYTES = WEIGHT_SOURCE_TILE * 96
K1_DESCRIPTOR_BYTES = 16
K8_DESCRIPTOR_BASE_BYTES = 16
K8_DESCRIPTOR_PER_SOURCE_BYTES = 4

COMMON_RESOURCE = {
    "lanes": 96,
    "accumulator_bits": 24,
    "clock_ns": 3.0,
    "external_bytes_per_cycle": 192,
    "onchip_sram_bytes_macro_rounded": 245760,
    "partitions": {
        "weight_bytes": 13824,
        "psum_bytes": 221184,
        "descriptor_control_bytes": 8192,
        "reserved_unallocated_bytes": 2560,
    },
    "ports": {
        "weight": {"banks": 8, "mode": "1R1W", "row_bytes": 16,
                   "read_latency_cycles": 4,
                   "initiation_interval": 1, "outstanding_per_bank": 8},
        "psum": {"banks": 6, "mode": "1RW", "row_bytes": 48,
                 "read_latency_cycles": 2, "write_latency_cycles": 1,
                 "initiation_interval": 1, "outstanding_per_bank": 8},
        "external": {"banks": 1, "mode": "1RW", "row_bytes": 192,
                     "read_latency_cycles": 32, "write_latency_cycles": 3,
                     "initiation_interval": 1, "outstanding_per_bank": 16},
        "compute": {"contexts": 1, "row_bytes": 288,
                    "latency_cycles": 1, "initiation_interval": 1},
    },
}


class M1539Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1539Error(message)


def product(values):
    result = 1
    for value in values:
        result *= int(value)
    return result


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")).hexdigest()


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          M1539Error("nonfinite JSON token: " + token)))


def safe_member(name):
    member = PurePosixPath(name)
    require(member.parts and not member.is_absolute() and
            ".." not in member.parts and member.as_posix() == name,
            "unsafe member path: " + name)
    return member


def regular_exact(path, expected, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise M1539Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " is not regular")
    require(sha256(path) == expected, label + " SHA drift")


def verify_sealed_directory(path, expected_outer_file_sha256,
                            verify_all_members=True):
    path = Path(path)
    require(path.is_dir() and not path.is_symlink(), "bad sealed directory")
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    regular_exact(outer, expected_outer_file_sha256, "outer seal")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split("  ", 1)
        require(len(fields) == 2 and len(fields[0]) == 64 and
                fields[1] not in expected, "malformed/duplicate SHA256SUMS")
        expected[fields[1]] = fields[0]
    require(outer.read_text(encoding="utf-8").split() ==
            [sha256(manifest), "SHA256SUMS"], "outer seal content drift")
    if verify_all_members:
        for name, digest in expected.items():
            member = path.joinpath(*safe_member(name).parts)
            regular_exact(member, digest, "sealed member " + name)
        actual = set(p.relative_to(path).as_posix() for p in path.rglob("*")
                     if p.is_file() and p.name not in
                     ("SHA256SUMS", "SHA256SUMS.seal.sha256"))
        require(actual == set(expected), "sealed member coverage drift")
    return {"members": len(expected), "manifest_sha256": sha256(manifest),
            "outer_seal_file_sha256": expected_outer_file_sha256}


def validate_resource():
    require(sum(COMMON_RESOURCE["partitions"].values()) == 245760,
            "240 KiB resource conservation failure")
    require(COMMON_RESOURCE["lanes"] == 96 and
            COMMON_RESOURCE["accumulator_bits"] == 24 and
            COMMON_RESOURCE["external_bytes_per_cycle"] == 192 and
            COMMON_RESOURCE["clock_ns"] == 3.0,
            "common resource axis drift")
    require(221184 % PSUM_VECTOR_BYTES == 0 and
            13824 % WEIGHT_TILE_BYTES == 0 and
            13824 // WEIGHT_TILE_BYTES == 9,
            "physical partition geometry drift")
    digest = canonical_sha(COMMON_RESOURCE)
    require(digest == M1525_RESOURCE_MANIFEST_SHA256,
            "M1525 common-resource manifest digest drift")
    return digest


def validate_authorities(verify_payload_members=False):
    """Bind M1521/M1527 and M1525/M1536 while enforcing M1526 STOP.

    Full payload hashing is optional for a fast source preflight.  A future
    production runner must call this with ``verify_payload_members=True``.
    """
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    regular_exact(M1521_MANIFEST, M1521_MANIFEST_SHA256, "M1521 manifest")
    m1521_seal = verify_sealed_directory(
        M1521_ROOT, M1521_OUTER_FILE_SHA256, verify_payload_members)
    regular_exact(M1527 / "review.json", M1527_REVIEW_SHA256, "M1527 review")
    m1527_seal = verify_sealed_directory(M1527, M1527_OUTER_FILE_SHA256, True)
    m1527 = strict_json(M1527 / "review.json")
    require(m1527.get("status") ==
            "PASS_M1527_M1521_EP34_DECODER_POSITIVE_PLANE_ACTUAL_RESULT__ADDRESS_REPLAY_SUCCESSOR_ALLOWED",
            "M1527 replay authority drift")
    regular_exact(M1525_SOURCE, M1525_SOURCE_SHA256, "M1525 source")
    regular_exact(M1525_CONTRACT, M1525_CONTRACT_SHA256, "M1525 contract")
    regular_exact(M1536 / "review.json", M1536_REVIEW_SHA256, "M1536 review")
    m1536_seal = verify_sealed_directory(M1536, M1536_OUTER_FILE_SHA256, True)
    m1536 = strict_json(M1536 / "review.json")
    require(m1536.get("status") ==
            "PASS_M1536_M1525_SOURCE_FOR_THREE_NONPRODUCT_CONFIGS__PRODUCT_BLOCKED_BY_M1526",
            "M1536 authority drift")
    admitted = [row["configuration"] for row in
                m1536.get("configuration_admission", [])
                if row.get("successor_authoring") is True]
    require(admitted == list(CONFIGS) and FORBIDDEN_CONFIG not in admitted,
            "M1536 configuration admission drift")
    require(m1536.get("authorization", {}).get(
                "fresh_address_timed_successor_authoring_for_three_nonproduct_configs") is True and
            m1536.get("authorization", {}).get("production_execution") is False,
            "M1536 release boundary drift")
    regular_exact(M1526_CONTRACT, M1526_CONTRACT_SHA256, "M1526 contract")
    m1526 = strict_json(M1526_CONTRACT)
    require(m1526.get("status") ==
            "SOURCE_ONLY__FAIL_CLOSED_NO_AUTHORIZED_EP34_DECODER_INT8_RULE" and
            m1526.get("claim_boundary", {}).get("m1525_int8_replay_admitted") is False,
            "M1526 product STOP drift")
    manifest = strict_json(M1521_MANIFEST)
    validate_population_manifest(manifest)
    return {"m1521": m1521_seal, "m1527": m1527_seal,
            "m1536": m1536_seal,
            "resource_manifest_sha256": validate_resource(),
            "full_payload_verification": bool(verify_payload_members)}


def validate_population_manifest(manifest):
    require(type(manifest) is dict and
            manifest.get("capture", {}).get("checkpoint_sha256") == CHECKPOINT_SHA256,
            "ep34 checkpoint identity drift")
    population = manifest.get("population")
    rows = manifest.get("records")
    require(type(population) is dict and population.get("samples") == 30 and
            population.get("calls") == 120 and population.get("modules") == 4 and
            population.get("positive_plane_files") == 120 and
            population.get("negative_plane_files") == 0,
            "ep34 population drift")
    require(type(rows) is list and len(rows) == 120,
            "ep34 record population drift")
    paths = set()
    for ordinal, row in enumerate(rows):
        module = ordinal % 4
        sample = 10 + ordinal // 4
        path = "payloads/c{:03d}_s{:02d}_d{}.positive.le.bitpack".format(
            ordinal, sample, module)
        require(type(row) is dict and row.get("global_call_ordinal") == ordinal and
                row.get("global_sample_id") == sample and
                row.get("replay_sample_ordinal") == sample - 10 and
                row.get("module_ordinal") == module and
                row.get("module") == MODULES[module] and
                tuple(row.get("shape", ())) == INPUT_SHAPES[module] and
                row.get("elements") == product(INPUT_SHAPES[module]) and
                row.get("plane_bytes") == (product(INPUT_SHAPES[module]) + 7) // 8 and
                row.get("positive_output") == path and
                row.get("layer_scale_word_uint32") == SCALE_WORDS[module] and
                row.get("weight_folding") is False and
                row.get("normalized") is False and row.get("coerced") is False and
                row.get("negative_plane_output") is None and
                row.get("negative_plane_all_zero") is True,
                "ep34 call/order/numeric drift")
        digest = row.get("positive_output_sha256")
        require(type(digest) is str and len(digest) == 64 and
                all(c in "0123456789abcdef" for c in digest),
                "positive payload SHA drift")
        require(path not in paths, "duplicate payload path")
        paths.add(path)
    return {"calls": 120, "samples": 30,
            "population_projection_sha256": canonical_sha(rows)}


def validate_config(config):
    require(config in CONFIGS and config != FORBIDDEN_CONFIG,
            "configuration is not an admitted nonproduct branch")


def bank_unique_groups(contributors, channels):
    """Pack contributor tuples ``(tap, channel)`` at most once per bank."""
    queues = [[] for _ in range(8)]
    for tap, channel in contributors:
        tap = int(tap); channel = int(channel)
        require(0 <= tap < 9 and 0 <= channel < channels,
                "contributor identity out of range")
        queues[channel % 8].append((tap, channel))
    output = []
    for ordinal in range(max([len(row) for row in queues] or [0])):
        group = tuple(row[ordinal] for row in queues if ordinal < len(row))
        require(len(set(channel % 8 for _tap, channel in group)) == len(group),
                "bank collision in typed group")
        output.append(group)
    require(sum(len(row) for row in output) == len(contributors),
            "group construction loses contributors")
    return output


def destination_sources(output_y, output_x, height, width):
    """Return exact input/tap coordinates for K3/S2/P1/OP1."""
    result = []
    for ky in range(3):
        numerator_y = int(output_y) + 1 - ky
        if numerator_y % 2:
            continue
        input_y = numerator_y // 2
        if not 0 <= input_y < height:
            continue
        for kx in range(3):
            numerator_x = int(output_x) + 1 - kx
            if numerator_x % 2:
                continue
            input_x = numerator_x // 2
            if 0 <= input_x < width:
                result.append((input_y, input_x, ky * 3 + kx))
    return tuple(result)


def contributors_for_destination(bit_getter, config, channels, height, width,
                                 output_y, output_x):
    validate_config(config)
    output = []
    for input_y, input_x, tap in destination_sources(
            output_y, output_x, height, width):
        for channel in range(channels):
            if config == "DENSE_TYPED_K8" or bit_getter(channel, input_y, input_x):
                output.append((tap, channel))
    return output


def request(identifier, config, kind, addresses, banks, width_bytes,
            dependencies=(), produces="", earliest=0):
    validate_config(config)
    value = {"id": str(identifier), "config": config, "kind": kind,
             "addresses": tuple(int(v) for v in addresses),
             "banks": tuple(int(v) for v in banks),
             "width_bytes": int(width_bytes),
             "dependencies": tuple(str(v) for v in dependencies),
             "produces": str(produces), "earliest_issue_cycle": int(earliest)}
    require(value["id"] and value["kind"] in
            ("external_read", "external_write", "weight_read", "weight_write", "psum_read",
             "psum_write", "compute", "commit"), "bad request kind/id")
    require(value["addresses"] and
            len(value["addresses"]) == len(value["banks"]),
            "request address/bank arity drift")
    require(value["width_bytes"] > 0 and value["earliest_issue_cycle"] >= 0,
            "bad request width/time")
    return value


def port_for(kind):
    if kind == "weight_read":
        return "weight", "read"
    if kind == "weight_write":
        return "weight", "write"
    if kind == "psum_read":
        return "psum", "read"
    if kind == "psum_write":
        return "psum", "write"
    if kind == "external_read":
        return "external", "read"
    if kind in ("external_write", "commit"):
        return "external", "write"
    require(kind == "compute", "unmapped request kind")
    return "compute", "write"


def normalized_port(resource_name):
    """Return timing fields without changing the frozen M1525 resource JSON."""
    raw = COMMON_RESOURCE["ports"][resource_name]
    if resource_name == "compute":
        return {"banks": raw["contexts"], "mode": "1RW",
                "row_bytes": raw["row_bytes"],
                "read_latency_cycles": raw["latency_cycles"],
                "write_latency_cycles": raw["latency_cycles"],
                "initiation_interval": raw["initiation_interval"],
                "outstanding_per_bank": 1}
    return {"banks": raw["banks"], "mode": raw["mode"],
            "row_bytes": raw["row_bytes"],
            "read_latency_cycles": raw["read_latency_cycles"],
            "write_latency_cycles": raw.get("write_latency_cycles",
                                             raw["read_latency_cycles"]),
            "initiation_interval": raw["initiation_interval"],
            "outstanding_per_bank": raw["outstanding_per_bank"]}


class AddressTimedScheduler(object):
    """Deterministic fixed-latency bank/port scheduler for one config."""
    def __init__(self, config):
        validate_config(config)
        self.config = config
        self.tokens = {}
        self.next_port = {}
        self.outstanding = {}
        self.address_digest = hashlib.sha256()
        self.commit_digest = hashlib.sha256()
        self.last_cycle = -1
        self.requests = 0
        self.kind_counts = {}
        self.byte_counts = {}

    def schedule_one(self, row):
        require(row["config"] == self.config, "cross-config request splice")
        resource_name, operation = port_for(row["kind"])
        port = normalized_port(resource_name)
        require(all(0 <= bank < port["banks"] for bank in row["banks"]),
                "bank index out of range")
        missing = [token for token in row["dependencies"] if token not in self.tokens]
        require(not missing, "unresolved dependency token")
        dependency_ready = max([self.tokens[token] for token in row["dependencies"]]
                               or [row["earliest_issue_cycle"]])
        key_operation = "rw" if port["mode"] == "1RW" else operation
        port_ready = max([self.next_port.get((resource_name, bank, key_operation), 0)
                          for bank in row["banks"]] or [0])
        issue = max(row["earliest_issue_cycle"], dependency_ready, port_ready)
        changed = True
        while changed:
            changed = False
            for bank in row["banks"]:
                active = sorted(value for value in self.outstanding.get(
                    (resource_name, bank), []) if value > issue)
                if len(active) >= port["outstanding_per_bank"]:
                    proposed = active[len(active) - port["outstanding_per_bank"]]
                    if proposed > issue:
                        issue = proposed
                        changed = True
        service_bytes = (COMMON_RESOURCE["external_bytes_per_cycle"]
                         if resource_name == "external" else port["row_bytes"])
        beats = max(1, int(math.ceil(float(row["width_bytes"]) / service_bytes)))
        latency = port["read_latency_cycles"] if operation == "read" else port["write_latency_cycles"]
        returned = issue + latency + beats - 1
        for bank in row["banks"]:
            self.next_port[(resource_name, bank, key_operation)] = (
                issue + max(port["initiation_interval"], beats))
            key = (resource_name, bank)
            active = [value for value in self.outstanding.get(key, [])
                      if value > issue]
            active.append(returned)
            self.outstanding[key] = active
        if row["kind"] in ("psum_read", "psum_write"):
            require(all(0 <= address and
                        address + row["width_bytes"] <= 221184
                        for address in row["addresses"]),
                    "psum address exceeds 221184-byte partition")
        if row["kind"] in ("weight_read", "weight_write"):
            require(all(0 <= address and
                        address + row["width_bytes"] <= 13824 // 8
                        for address in row["addresses"]),
                    "weight bank-local address exceeds 1728-byte capacity")
        if row["produces"]:
            require(row["produces"] not in self.tokens, "duplicate token")
            self.tokens[row["produces"]] = returned
        for address, bank in zip(row["addresses"], row["banks"]):
            self.address_digest.update(json.dumps(
                [row["id"], row["kind"], address, bank],
                separators=(",", ":")).encode("utf-8"))
        if row["kind"] == "commit":
            for address in row["addresses"]:
                self.commit_digest.update(json.dumps(
                    [self.kind_counts.get("commit", 0), address,
                     row["width_bytes"]], separators=(",", ":")).encode("utf-8"))
        self.last_cycle = max(self.last_cycle, returned)
        self.requests += 1
        self.kind_counts[row["kind"]] = self.kind_counts.get(row["kind"], 0) + 1
        self.byte_counts[row["kind"]] = self.byte_counts.get(row["kind"], 0) + (
            row["width_bytes"] * len(row["banks"]))
        return {"id": row["id"], "issue_cycle": issue,
                "return_cycle": returned,
                "dependency_ready_cycle": dependency_ready}

    def schedule(self, rows):
        for row in rows:
            self.schedule_one(row)
        require(self.requests > 0, "empty schedule")
        return {"configuration": self.config,
                "resource_manifest_sha256": validate_resource(),
                "total_cycles": self.last_cycle + 1,
                "request_count": self.requests,
                "kind_counts": dict(self.kind_counts),
                "byte_counts": dict(self.byte_counts),
                "transaction_address_sha256": self.address_digest.hexdigest(),
                "commit_sequence_sha256": self.commit_digest.hexdigest()}


class WeightTileCache(object):
    """Nine 1536-byte read-only tiles over eight 1728-byte physical banks."""
    def __init__(self):
        self.capacity = 9
        self.key_to_slot = {}
        self.age = {}
        self.tick = 0

    def prepare(self, keys):
        unique = []
        for key in keys:
            key = tuple(int(v) for v in key)
            if key not in unique:
                unique.append(key)
        require(len(unique) <= 8, "one K8 group exceeds eight weight tiles")
        pinned = set(unique)
        misses = []
        for key in unique:
            self.tick += 1
            if key in self.key_to_slot:
                self.age[key] = self.tick
                continue
            if len(self.key_to_slot) < self.capacity:
                slot = min(set(range(self.capacity)) -
                           set(self.key_to_slot.values()))
            else:
                candidates = [item for item in self.key_to_slot
                              if item not in pinned]
                require(candidates, "weight cache has no unpinned victim")
                victim = min(candidates,
                             key=lambda item: (self.age[item], item))
                slot = self.key_to_slot.pop(victim)
                self.age.pop(victim)
            self.key_to_slot[key] = slot
            self.age[key] = self.tick
            misses.append((key, slot))
        return misses

    def slot(self, key):
        key = tuple(int(v) for v in key)
        require(key in self.key_to_slot, "weight tile is not resident")
        return self.key_to_slot[key]


def weight_bank_row(channel, cache_slot):
    """Bank-local address of one source's 96-byte output-block vector."""
    channel = int(channel); cache_slot = int(cache_slot)
    bank = channel % 8
    within_tile_source = channel % WEIGHT_SOURCE_TILE
    local_source = within_tile_source // 8
    address = cache_slot * (WEIGHT_TILE_BYTES // 8) + local_source * 96
    require(0 <= cache_slot < 9 and address + 96 <= 13824 // 8,
            "weight tile bank-local address out of range")
    return bank, address


def destination_transactions(config, module, timestep, destination,
                             output_block, contributors, previous_token="",
                             weight_cache=None):
    """Build exact request dependencies for one output-vector destination."""
    validate_config(config)
    cin, _cout, _hin, _win, _hout, _wout = GEOMETRY[module]
    groups = bank_unique_groups(contributors, cin)
    prefix = "{}:m{}:t{}:d{}:ob{}".format(
        config, module, timestep, destination, output_block)
    psum_slot = (destination * int(math.ceil(float(GEOMETRY[module][1]) / 96)) +
                 output_block) % (221184 // PSUM_VECTOR_BYTES)
    psum_base = psum_slot * PSUM_VECTOR_BYTES
    previous = previous_token
    if weight_cache is None:
        weight_cache = WeightTileCache()
    for ordinal, group in enumerate(groups):
        group_prefix = prefix + ":g{}".format(ordinal)
        descriptor_tokens = []
        if config in ("DENSE_TYPED_K8", "BIT_TYPED_K8"):
            token = group_prefix + ":typed_desc_done"
            size = K8_DESCRIPTOR_BASE_BYTES + K8_DESCRIPTOR_PER_SOURCE_BYTES * len(group)
            yield request(group_prefix + ":typed_desc", config, "external_read",
                          [(3 << 60) | (destination << 16) | ordinal], [0], size,
                          produces=token)
            descriptor_tokens.append(token)
        else:
            for lane, _item in enumerate(group):
                token = group_prefix + ":k1_desc{}_done".format(lane)
                yield request(group_prefix + ":k1_desc{}".format(lane), config,
                              "external_read",
                              [(3 << 60) | (destination << 16) |
                               (ordinal << 4) | lane], [0], K1_DESCRIPTOR_BYTES,
                              produces=token)
                descriptor_tokens.append(token)
        tile_keys = [(module, output_block, tap,
                      channel // WEIGHT_SOURCE_TILE)
                     for tap, channel in group]
        misses = weight_cache.prepare(tile_keys)
        refill_tokens = []
        for refill_ordinal, (key, slot) in enumerate(misses):
            external_token = group_prefix + ":refill{}_external_done".format(
                refill_ordinal)
            refill_address = ((2 << 60) | (module << 52) |
                              (output_block << 44) | (key[2] << 36) |
                              (key[3] << 16))
            yield request(group_prefix + ":refill{}".format(refill_ordinal),
                          config, "external_read", [refill_address], [0],
                          WEIGHT_TILE_BYTES, descriptor_tokens, external_token)
            token = group_prefix + ":refill{}_weight_done".format(
                refill_ordinal)
            yield request(group_prefix + ":refill{}_weight_write".format(
                              refill_ordinal), config, "weight_write",
                          [slot * (WEIGHT_TILE_BYTES // 8) for _bank in range(8)],
                          range(8), WEIGHT_TILE_BYTES // 8,
                          [external_token], token)
            refill_tokens.append(token)
        banks = []; offsets = []
        for (tap, channel), key in zip(group, tile_keys):
            bank, offset = weight_bank_row(channel, weight_cache.slot(key))
            banks.append(bank); offsets.append(offset)
        weight_tokens = []
        if config in ("DENSE_TYPED_K8", "BIT_TYPED_K8"):
            token = group_prefix + ":typed_weight_done"
            yield request(group_prefix + ":typed_weight", config, "weight_read",
                          offsets, banks, 96,
                          tuple(descriptor_tokens + refill_tokens), token)
            weight_tokens.append(token)
        else:
            for lane, (bank, offset) in enumerate(zip(banks, offsets)):
                token = group_prefix + ":k1_weight{}_done".format(lane)
                yield request(group_prefix + ":k1_weight{}".format(lane), config,
                              "weight_read", [offset], [bank], 96,
                              tuple(descriptor_tokens + refill_tokens), token)
                weight_tokens.append(token)
        read_token = group_prefix + ":psum_read_done"
        read_dependencies = tuple(([previous] if previous else []) + weight_tokens)
        yield request(group_prefix + ":psum_read", config, "psum_read",
                      [psum_base + bank * 48 for bank in range(6)], range(6), 48,
                      read_dependencies, read_token)
        compute_token = group_prefix + ":compute_done"
        yield request(group_prefix + ":compute", config, "compute", [0], [0],
                      PSUM_VECTOR_BYTES, [read_token], compute_token)
        write_token = group_prefix + ":psum_write_done"
        yield request(group_prefix + ":psum_write", config, "psum_write",
                      [psum_base + bank * 48 for bank in range(6)], range(6), 48,
                      [compute_token], write_token)
        previous = write_token
    return


def synthetic_config_transactions(config, bits, module=3, timestep=0):
    """Small exact synthetic kernel used only by author and hammer tests."""
    validate_config(config)
    cin = len(bits); height = len(bits[0]); width = len(bits[0][0])
    require((cin, height, width) == (GEOMETRY[module][0],
                                     GEOMETRY[module][2],
                                     GEOMETRY[module][3]) or
            (cin, height, width) == (8, 2, 2),
            "synthetic tensor geometry drift")
    output_height = height * 2; output_width = width * 2
    output_blocks = 1 if cin == 8 else int(math.ceil(float(GEOMETRY[module][1]) / 96))
    input_bytes = (cin * height * width + 7) // 8
    source_token = config + ":source_done"
    yield request(config + ":source", config, "external_read", [1 << 60], [0],
                  input_bytes, produces=source_token)
    # Common non-product protocol charge: one parent/control scratch
    # round-trip.  It is identical in all three branches and cannot be waived.
    control_read = config + ":control_read_done"
    yield request(config + ":control_read", config, "external_read",
                  [(5 << 60)], [0], 144, [source_token], control_read)
    control_write = config + ":control_write_done"
    yield request(config + ":control_write", config, "external_write",
                  [(5 << 60)], [0], 144, [control_read], control_write)
    getter = lambda channel, y, x: bool(bits[channel][y][x])
    weight_cache = WeightTileCache()
    for oy in range(output_height):
        for ox in range(output_width):
            destination = oy * output_width + ox
            contributors = contributors_for_destination(
                getter, config, cin, height, width, oy, ox)
            for output_block in range(output_blocks):
                last = ""
                rows = destination_transactions(
                    config, module if cin != 8 else 3, timestep, destination,
                    output_block, contributors, control_write, weight_cache)
                for row in rows:
                    if row["kind"] == "psum_write":
                        last = row["produces"]
                    yield row
                commit_address = ((4 << 60) | (module << 52) |
                                  (timestep << 44) |
                                  ((destination * output_blocks + output_block) *
                                   OUTPUT_COMMIT_BYTES))
                yield request("{}:commit:{}:{}".format(config, destination,
                                                       output_block),
                              config, "commit", [commit_address], [0],
                              OUTPUT_COMMIT_BYTES,
                              [last] if last else [control_write])


def compare_rows(rows):
    require(type(rows) is list and
            [row.get("configuration") for row in rows] == list(CONFIGS),
            "comparator row order/configuration drift")
    for key in ("resource_manifest_sha256", "commit_sequence_sha256",
                "checkpoint_sha256", "population_manifest_sha256"):
        require(len(set(row.get(key) for row in rows)) == 1,
                "comparator {} mismatch".format(key))
    require(all(row.get("checkpoint_sha256") == CHECKPOINT_SHA256 for row in rows),
            "comparator checkpoint drift")
    return True


def production_release(_token=None):
    raise M1539Error(
        "M1539 is source-only; production requires a sealed independent hammer "
        "and a distinct one-shot runner. PRODUCT_CAPTURE_TYPED_K8 remains blocked.")


def synthetic_self_test():
    bits = [[[0 for _x in range(2)] for _y in range(2)] for _c in range(8)]
    bits[0][0][0] = 1
    # Two channels at one spatial source force at least one two-lane group,
    # exposing K1x8's independent descriptor charge against one typed bundle.
    bits[1][0][0] = 1
    results = []
    for config in CONFIGS:
        scheduler = AddressTimedScheduler(config)
        results.append(scheduler.schedule(
            synthetic_config_transactions(config, bits)))
    require(results[0]["kind_counts"]["compute"] >
            results[2]["kind_counts"]["compute"],
            "dense denominator did not issue more groups than sparse K8")
    require(results[1]["kind_counts"]["compute"] ==
            results[2]["kind_counts"]["compute"],
            "equal-service and typed K8 product service drift")
    require(results[1]["byte_counts"]["external_read"] >
            results[2]["byte_counts"]["external_read"],
            "K1x8 descriptor traffic is not charged")
    require(len(set(row["commit_sequence_sha256"] for row in results)) == 1,
            "dense commit sequence differs across configurations")
    return {"status": "PASS_M1539_SYNTHETIC_SOURCE_TEST",
            "configurations": list(CONFIGS), "results": results,
            "production": False, "product_capture": False}


def describe():
    return {"schema": SCHEMA, "status": STATUS,
            "configurations": list(CONFIGS),
            "forbidden_configuration": FORBIDDEN_CONFIG,
            "common_resource": COMMON_RESOURCE,
            "resource_manifest_sha256": validate_resource(),
            "source_capabilities": {"identity_preflight": True,
                "synthetic_address_timed_schedule": True,
                "production_population_runner": False,
                "production_launch": False},
            "claim_boundary": {"source_only": True, "production": False,
                "transactions": False, "cycles": False, "traffic": False,
                "speedup": False, "system_speedup": False, "energy": False,
                "rtl": False, "eda": False, "ppa": False,
                "table_a": False, "paper_result": False}}


def main(argv=None):
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--describe", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--synthetic-self-test", action="store_true")
    parser.add_argument("--verify-payload-members", action="store_true")
    args = parser.parse_args(argv)
    if args.describe:
        require(not args.verify_payload_members,
                "payload verification is valid only with --preflight")
        value = describe()
    elif args.preflight:
        value = {"schema": SCHEMA,
                 "status": "PASS_M1539_SOURCE_PREFLIGHT__NO_PRODUCTION",
                 "authorities": validate_authorities(args.verify_payload_members),
                 "production": False}
    else:
        require(not args.verify_payload_members,
                "payload verification is valid only with --preflight")
        value = synthetic_self_test()
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
