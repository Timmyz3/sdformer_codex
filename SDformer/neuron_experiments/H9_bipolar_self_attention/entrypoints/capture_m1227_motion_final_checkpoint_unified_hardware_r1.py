#!/opt/conda/envs/sdformerflow/bin/python
"""M1227 checkpoint-parametric unified Motion hardware capture successor.

This checked-in source is inert and source-only.  A future release must bind a
fresh final-checkpoint selection result and a different-author source hammer.
The capture keeps the static 259-module/105-ATLIF topology, but its runtime
contract is the proven 247 live modules plus twelve H60 K-as-V dead sn_v
leaves.  Every completed sample is atomically persisted below staging for
failure forensics; only a complete 40-sample result can be published.
"""

import argparse
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
import uuid


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE_CONTRACT = HW / "contracts/m1227_motion_final_checkpoint_unified_capture_source_contract_r1_20260830.json"
SUBSTRATE = Path(__file__).with_name("capture_m1174_motion_checkpoint_parametric_unified_hardware.py")
SUBSTRATE_SHA256 = "b476fad6885be23aa63a6b5d8e690fb3e213421074270cbb25e8ec00c202080a"
PROFILE = Path(__file__).with_name("profile_nts11_hardware_p0.py")
BIT_WRITER = Path(__file__).with_name("h67_bit_trace.py")
M1224 = HW / "reviews/m1224_m1208_capture_contract_first_principles_audit_r1_20260830"
M1224_REVIEW_SHA256 = "56372da531b3c56b375d45372ff2aea9be1754df2ba4a3a8c0e50a62936505dc"
M1224_MANIFEST_SHA256 = "677bb08190b1f345db8f4e5535c73d22952cbb44324bfb4b465a728032705135"
M1224_OUTER_FILE_SHA256 = "9a858b44edb2bf36c4bad251eaa4501860b92c5d7ff83bcc6ba60a318a165b96"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
CANONICAL_LEASE = HW / "results/gpu_profile_lease.lock"
CANONICAL_RESULT = HW / "results/m1227_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830"
CANONICAL_ATTEMPT = HW / "results/.m1227_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830.attempt_consumed"
CANONICAL_LOG = HW / "results/.m1227_motion_final_checkpoint_unified_hardware_capture_s40_r1_20260830.production.log"
PINNED_LINK_REL = Path("data/Datasets/DSEC")
PINNED_DSEC_ROOT = Path("/root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC")
SOURCE_SCHEMA = "m1227_motion_final_checkpoint_unified_capture_source_contract_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1228_HAMMER_AND_RELEASE_REQUIRED__NO_GPU"
LAUNCH_SCHEMA = "m1227_motion_final_checkpoint_unified_capture_launch_r1_v1"
LAUNCH_STATUS = "FINAL_SELECTION_AND_M1228_BOUND__ONE_M1227_GPU_RUN_AUTHORIZED"
HAMMER_SCHEMA = "m1228_m1227_motion_final_checkpoint_unified_capture_source_hammer_r1_v1"
ATTEMPT_TOKEN = "M1227_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n"
PASS_TOKEN = "PASS_M1227_FINAL_CHECKPOINT_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED"
SEQUENCES = ("interlaken_01_a", "thun_01_b", "zurich_city_12_a")
C1_TARGETS = tuple(
    "sttmultires_unet.resblocks.{}.conv{}.0".format(block, conv)
    for block in range(2) for conv in range(1, 3)
)
DECODER_TARGETS = tuple(
    "sttmultires_unet.decoders.{}.deconv.0".format(index) for index in range(4)
)
ATTENTION_ALIASES = tuple(
    "S{}.B{}.attn".format(stage, block)
    for stage, blocks in enumerate((2, 2, 6, 2)) for block in range(blocks)
)
DEAD_SN_V = tuple(
    "sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.attn.sn_v.spiking_neuron".format(stage, block)
    for stage, blocks in enumerate((2, 2, 6, 2)) for block in range(blocks)
)
CATEGORIES = frozenset((
    "c1_conv3x3", "decoder_convtranspose", "atlif", "fc1", "fc2",
    "patch_embed", "batch_norm", "qkv", "attention",
))
EXPECTED_STATIC_COUNTS = {
    "c1_conv3x3": 4, "decoder_convtranspose": 4, "atlif": 105,
    "fc1": 12, "fc2": 12, "patch_embed": 8, "batch_norm": 78,
    "qkv": 24, "attention": 12,
}
EXPECTED_LIVE_COUNTS = dict(EXPECTED_STATIC_COUNTS, atlif=93)


class M1227Error(RuntimeError):
    pass


def require(value, message):
    if not value:
        raise M1227Error(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular(path, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise M1227Error("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "{} must be a non-symlink regular file: {}".format(label, path))


def directory(path, label):
    path = Path(path)
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise M1227Error("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISDIR(mode) and not path.is_symlink(),
            "{} must be a non-symlink directory: {}".format(label, path))


def strict_json(path):
    def reject(token):
        raise M1227Error("non-standard JSON token: " + token)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(Path(path).read_text(encoding="utf-8"),
                       object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def safe_repo_path(relative, missing_leaf=False):
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts and value.parts,
            "unsafe repository-relative path")
    candidate = ROOT / value
    cursor = ROOT
    limit = len(value.parts) - (1 if missing_leaf else 0)
    for part in value.parts[:limit]:
        cursor = cursor / part
        require(os.path.lexists(str(cursor)) and not cursor.is_symlink(),
                "missing/symlink repository component: " + str(cursor))
    return candidate


def _safe_member(root, name):
    value = Path(name)
    require(name == value.as_posix() and not value.is_absolute() and
            ".." not in value.parts and value.parts, "unsafe sealed member")
    cursor = root
    for part in value.parts:
        cursor = cursor / part
        require(os.path.lexists(str(cursor)) and not cursor.is_symlink(),
                "missing/symlink sealed component: " + str(cursor))
    regular(cursor, "sealed member")
    return cursor


def verify_double_seal(root, manifest_sha=None, outer_file_sha=None):
    root = Path(root)
    directory(root, "sealed root")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    regular(manifest, "manifest")
    regular(outer, "outer seal")
    if manifest_sha is not None:
        require(sha256(manifest) == manifest_sha, "manifest SHA mismatch")
    if outer_file_sha is not None:
        require(sha256(outer) == outer_file_sha, "outer-file SHA mismatch")
    require(outer.read_text(encoding="utf-8").split() == [sha256(manifest), "SHA256SUMS"],
            "outer seal content mismatch")
    rows = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64, "malformed manifest row")
        name = fields[1].lstrip("*")
        require(name not in rows, "duplicate sealed member")
        member = _safe_member(root, name)
        require(sha256(member) == fields[0], "sealed member SHA mismatch: " + name)
        rows[name] = fields[0]
    actual = set(
        path.relative_to(root).as_posix() for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in
        ("SHA256SUMS", "SHA256SUMS.seal.sha256")
    )
    require(actual == set(rows), "recursive sealed population mismatch")
    return rows


def write_double_seal(root):
    root = Path(root)
    excluded = ("SHA256SUMS", "SHA256SUMS.seal.sha256")
    members = sorted(
        path.relative_to(root) for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )
    manifest = root / "SHA256SUMS"
    manifest.write_text("".join(
        "{}  {}\n".format(sha256(root / item), item.as_posix()) for item in members
    ), encoding="utf-8")
    (root / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")
    verify_double_seal(root)


def load_substrate():
    """Lazy: called only after a production launch contract passes admission."""
    regular(SUBSTRATE, "sealed M1174 substrate")
    require(sha256(SUBSTRATE) == SUBSTRATE_SHA256, "M1174 substrate SHA drift")
    spec = importlib.util.spec_from_file_location("m1227_sealed_m1174", str(SUBSTRATE))
    require(spec is not None and spec.loader is not None, "cannot import M1174 substrate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def inventory_digest(names):
    return hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()


def frozen_non_atlif_inventory(policy):
    authorities = policy["inventory_authorities"]
    for entry in authorities.values():
        path = safe_repo_path(entry["path"])
        regular(path, "inventory authority")
        require(sha256(path) == entry["sha256"], "inventory authority SHA drift")
    with safe_repo_path(authorities["operator_runtime"]["path"]).open(
            newline="", encoding="utf-8") as stream:
        operators = list(csv.DictReader(stream))
    fc1 = sorted(set(row["name"] for row in operators if row["name"].endswith(".mlp.fc1")))
    fc2 = sorted(set(row["name"] for row in operators if row["name"].endswith(".mlp.fc2")))
    qkv = sorted(set(
        row["name"] for row in operators
        if row["name"].endswith(".attn.linear_q") or row["name"].endswith(".attn.linear_k")
    ))
    patch = sorted(set(row["name"] for row in operators if ".patch_embed." in row["name"]))
    bn = set()
    for line in safe_repo_path(authorities["dependency_events"]["path"]).read_text(
            encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("module_type") in ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d"):
            bn.add(row["name"])
    result = {
        "c1_conv3x3": list(C1_TARGETS),
        "decoder_convtranspose": list(DECODER_TARGETS),
        "fc1": fc1, "fc2": fc2, "qkv": qkv, "patch_embed": patch,
        "batch_norm": sorted(bn),
        "attention": sorted(set(name.rsplit(".", 1)[0] for name in qkv)),
    }
    for category, names in result.items():
        expected = policy["static_inventory"][category]
        require(len(names) == expected["modules"] and
                inventory_digest(names) == expected["names_sha256"],
                "frozen inventory drift: " + category)
    return result


def expected_live_inventory(static_inventory):
    dead = set(DEAD_SN_V)
    atlifs = set(static_inventory["atlif"])
    require(len(atlifs) == 105 and dead <= atlifs, "ATLIF static/dead inventory mismatch")
    result = dict((key, list(value)) for key, value in static_inventory.items())
    result["atlif"] = sorted(atlifs - dead)
    require(dict((key, len(value)) for key, value in result.items()) == EXPECTED_LIVE_COUNTS,
            "live inventory counts are not 247/ATLIF93")
    require(sum(len(value) for value in result.values()) == 247, "live inventory sum mismatch")
    return result


def audit_call_matrix(records, live_inventory, sample_ids):
    sample_ids = list(sample_ids)
    expected_name_category = {}
    for category, names in live_inventory.items():
        for name in names:
            expected_name_category[name] = category
    counts = dict((sample, {}) for sample in sample_ids)
    errors = []
    for row in records:
        sample = int(row.get("global_sample_id", -1))
        name = str(row.get("name", ""))
        category = str(row.get("category", ""))
        if sample not in counts:
            errors.append("unexpected_sample:{}".format(sample))
            continue
        if name in DEAD_SN_V:
            errors.append("dead_module_fired:{}:{}".format(sample, name))
            continue
        if expected_name_category.get(name) != category:
            errors.append("unexpected_name_or_category:{}:{}:{}".format(sample, category, name))
            continue
        counts[sample][name] = counts[sample].get(name, 0) + 1
    for sample in sample_ids:
        for name in expected_name_category:
            value = counts[sample].get(name, 0)
            if value != 1:
                errors.append("call_count:{}:{}:{}".format(sample, name, value))
    expected_rows = len(sample_ids) * len(expected_name_category)
    if len(records) != expected_rows:
        errors.append("record_count:{}:{}".format(len(records), expected_rows))
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "samples": len(sample_ids),
        "live_modules_per_sample": len(expected_name_category),
        "records": len(records),
        "expected_records": expected_rows,
        "dead_modules": len(DEAD_SN_V),
    }


def atomic_sample_snapshot(staging, sample_id, ordered_rows, profiler, call_audit):
    parent = Path(staging) / "forensic_samples"
    parent.mkdir(exist_ok=True)
    final = parent / "sample_{:02d}".format(sample_id)
    require(not os.path.lexists(str(final)), "sample snapshot already exists")
    temporary = parent / (".sample_{:02d}.{}.tmp".format(sample_id, uuid.uuid4().hex))
    temporary.mkdir()
    try:
        execution = [
            row for row in profiler.execution_records
            if int(row.get("sample_id", -1)) == int(sample_id)
        ]
        operator_rows = sorted(profiler.operator_records.values(), key=lambda row: row["name"])
        atlif_rows = sorted(
            [dict(value, name=name) for name, value in profiler.atlif_records.items()],
            key=lambda row: row["name"],
        )
        payloads = {
            "unified_ordered_sample.jsonl": "".join(
                json.dumps(row, sort_keys=True) + "\n" for row in ordered_rows),
            "execution_sample.json": json.dumps(execution, sort_keys=True) + "\n",
            "operator_runtime_cumulative.json": json.dumps(operator_rows, sort_keys=True) + "\n",
            "atlif_activity_cumulative.json": json.dumps(atlif_rows, sort_keys=True) + "\n",
        }
        for name, text in payloads.items():
            path = temporary / name
            path.write_text(text, encoding="utf-8")
            with path.open("rb") as stream:
                os.fsync(stream.fileno())
        manifest = {
            "schema": "m1227_atomic_sample_forensic_snapshot_r1_v1",
            "sample_id": int(sample_id),
            "status": "SAMPLE_COMPLETE__FORENSIC_ONLY__NOT_CANONICAL",
            "call_audit": call_audit,
            "counts": {
                "ordered": len(ordered_rows), "execution": len(execution),
                "operator_runtime_cumulative": len(operator_rows),
                "atlif_activity_cumulative": len(atlif_rows),
            },
            "files": dict((name, sha256(temporary / name)) for name in sorted(payloads)),
            "claim_boundary": {"forensic_only": True, "canonical": False, "paper_result": False},
        }
        manifest_path = temporary / "snapshot_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        with manifest_path.open("rb") as stream:
            os.fsync(stream.fileno())
        descriptor = os.open(str(temporary), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(str(temporary), str(final))
        descriptor = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        if temporary.exists():
            for path in sorted(temporary.rglob("*"), reverse=True):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
            temporary.rmdir()
        raise
    return final


def make_live_dead_writer(r1, static_non_atlif):
    class LiveDeadWriter(r1.UnifiedHookWriter):
        ACTIVE_PROFILER = None

        def _category(self, name, module):
            if name in C1_TARGETS:
                return "c1_conv3x3"
            if name in DECODER_TARGETS:
                return "decoder_convtranspose"
            if module.__class__.__name__ == "ATLIFTernaryPSN":
                return "atlif"
            for category in ("fc1", "fc2", "patch_embed", "batch_norm", "qkv", "attention"):
                if name in static_non_atlif[category]:
                    return category
            return None

        def attach(self, model):
            named = dict(model.named_modules())
            atlifs = sorted(
                name for name, module in named.items()
                if module.__class__.__name__ == "ATLIFTernaryPSN"
            )
            self.static_inventory = dict(static_non_atlif, atlif=atlifs)
            require(dict((key, len(value)) for key, value in self.static_inventory.items()) ==
                    EXPECTED_STATIC_COUNTS, "static inventory is not 259/ATLIF105")
            self.live_inventory = expected_live_inventory(self.static_inventory)
            super(LiveDeadWriter, self).attach(model)
            require(dict((key, sorted(value)) for key, value in self.module_inventory.items()) ==
                    dict((key, sorted(value)) for key, value in self.static_inventory.items()),
                    "attached inventory differs from exact static inventory")
            self._snapshot_cursor = 0
            self._m1227_closed = False

        def end(self):
            require(self.sample is not None, "sample not active")
            sample_id = int(self.sample["global_sample_id"])
            super(LiveDeadWriter, self).end()
            rows = self.records[self._snapshot_cursor:]
            audit = audit_call_matrix(rows, self.live_inventory, [sample_id])
            profiler = type(self).ACTIVE_PROFILER
            require(profiler is not None, "snapshot profiler is not attached")
            atomic_sample_snapshot(self.staging, sample_id, rows, profiler, audit)
            self._snapshot_cursor = len(self.records)
            require(audit["status"] == "PASS", "sample live/dead call matrix failed")

        def close(self):
            if getattr(self, "_m1227_closed", False):
                return
            try:
                if hasattr(self, "live_inventory"):
                    audit = audit_call_matrix(self.records, self.live_inventory, range(40))
                    require(audit["status"] == "PASS", "global live/dead call matrix failed")
            finally:
                super(LiveDeadWriter, self).close()
                self._m1227_closed = True

    return LiveDeadWriter


def make_snapshot_profiler(base, writer_type):
    class SnapshotProfiler(base):
        def __init__(self, *args, **kwargs):
            super(SnapshotProfiler, self).__init__(*args, **kwargs)
            writer_type.ACTIVE_PROFILER = self

    return SnapshotProfiler


def make_strict_attention_writer(base):
    class StrictAttention(base):
        def _assert_complete(self):
            audit_attention_population(self.records)
            import numpy as np
            for row in self.records:
                path = Path(row["file"])
                regular(path, "attention NPZ")
                require(sha256(path) == row["sha256"], "attention NPZ SHA mismatch")
                with np.load(path, allow_pickle=False) as payload:
                    require(set(("q_bits_packed", "k_bits_packed", "gate_q17")) <= set(payload.files),
                            "attention NPZ keys incomplete")
                    require(payload["q_bits_packed"].size > 0 and
                            payload["k_bits_packed"].size > 0 and
                            payload["gate_q17"].size > 0, "attention NPZ payload empty")

        def write_manifest(self):
            self.output_dir.mkdir(parents=True, exist_ok=True)
            stages = sorted(set(int(row["name"].split(".")[0][1:]) for row in self.records))
            payload = {
                "schema_version": 1, "sample_limit": self.sample_limit,
                "windows_per_call": self.windows_per_call,
                "first_block_only": self.first_block_only,
                "run_context": self.run_context, "records": self.records,
                "coverage": {"stages": stages, "stage_count": len(stages),
                             "record_count": len(self.records)},
            }
            (self.output_dir / "manifest.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        @property
        def manifest_path(self):
            self._assert_complete()
            return self.output_dir / "manifest.json"

    return StrictAttention


def audit_attention_population(records, samples=40):
    expected = set((sample, name) for sample in range(samples) for name in ATTENTION_ALIASES)
    observed = [(int(row["sample_id"]), str(row["name"])) for row in records]
    require(len(observed) == len(expected),
            "attention record count must be {}x12".format(samples))
    require(len(set(observed)) == len(observed), "duplicate attention record")
    require(set(observed) == expected, "attention Cartesian coverage mismatch")
    return {"status": "PASS", "samples": samples, "modules_per_sample": 12,
            "records": len(observed)}


def validate_payload_population(staging):
    files = sorted((Path(staging) / "payloads").iterdir())
    pattern = re.compile(r"s(\d{2})_o(\d{5})_([0-9a-f]{12})\.(fp32\.zlib|support_sign\.le\.bitpack)")
    rows = []
    for path in files:
        match = pattern.fullmatch(path.name)
        require(match is not None and path.is_file() and not path.is_symlink(),
                "malformed payload member")
        rows.append((int(match.group(1)), match.group(3), match.group(4)))
    expected_hashes = set(hashlib.sha256(name.encode()).hexdigest()[:12]
                          for name in C1_TARGETS + DECODER_TARGETS)
    expected = set(
        (sample, name_hash, suffix) for sample in range(40)
        for name_hash in expected_hashes
        for suffix in ("fp32.zlib", "support_sign.le.bitpack")
    )
    require(len(files) == 640 and set(rows) == expected, "payload population must be 40x8x2")
    return files


def validate_snapshot_population(staging):
    root = Path(staging) / "forensic_samples"
    directory(root, "forensic snapshot root")
    expected_files = set((
        "unified_ordered_sample.jsonl", "execution_sample.json",
        "operator_runtime_cumulative.json", "atlif_activity_cumulative.json",
    ))
    for sample in range(40):
        directory(root / "sample_{:02d}".format(sample), "atomic sample snapshot")
        sample_root = root / "sample_{:02d}".format(sample)
        manifest = strict_json(sample_root / "snapshot_manifest.json")
        require(manifest["sample_id"] == sample and
                manifest["call_audit"]["status"] == "PASS" and
                manifest["call_audit"]["records"] == 247,
                "sample forensic snapshot audit mismatch")
        require(set(manifest["files"]) == expected_files, "snapshot file population mismatch")
        for name, digest in manifest["files"].items():
            regular(sample_root / name, "snapshot member")
            require(sha256(sample_root / name) == digest, "snapshot member SHA mismatch")
    actual_dirs = sorted(path.name for path in root.iterdir() if path.is_dir())
    require(actual_dirs == ["sample_{:02d}".format(sample) for sample in range(40)],
            "snapshot directory population mismatch")


def final_validate_and_seal(staging, writer_type, selected_identity):
    staging = Path(staging)
    ordered_path = staging / "unified_ordered_records.jsonl"
    ordered = [json.loads(line) for line in ordered_path.read_text(encoding="utf-8").splitlines()]
    audit = audit_call_matrix(ordered, writer_type.ACTIVE_WRITER.live_inventory, range(40))
    require(audit["status"] == "PASS" and len(ordered) == 9880,
            "final ordered population is not 40x247")
    attention = strict_json(staging / "attention_qk/manifest.json")
    require(len(attention["records"]) == 480, "final attention population is not 480")
    validate_payload_population(staging)
    validate_snapshot_population(staging)
    execution = json.loads((staging / "execution_trace.json").read_text(encoding="utf-8"))
    operators = json.loads((staging / "operator_runtime.json").read_text(encoding="utf-8"))
    atlif = json.loads((staging / "atlif_activity.json").read_text(encoding="utf-8"))
    require(len(execution) == 7360, "execution population must be 40x184")
    require(len(operators) == 79 and all(int(row["calls"]) == 40 for row in operators),
            "operator runtime must contain 79 rows at 40 calls")
    require(len(atlif) == 93 and all(int(row["calls"]) == 40 for row in atlif),
            "ATLIF runtime must contain 93 live rows at 40 calls")
    require(not set(row["name"] for row in atlif) & set(DEAD_SN_V), "dead sn_v appeared in ATLIF runtime")
    manifest_path = staging / "manifest.json"
    manifest = strict_json(manifest_path)
    manifest.update({
        "schema": "m1227_motion_final_checkpoint_unified_hardware_capture_r1_v1",
        "status": "CAPTURE_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
        "m1227_runtime_contract": {
            "static_modules": 259, "static_atlif": 105,
            "live_modules_per_sample": 247, "live_atlif": 93,
            "dead_sn_v": list(DEAD_SN_V), "dead_calls_per_sample": 0,
            "ordered_records": 9880, "attention_records": 480, "payload_files": 640,
            "final_selection_identity": selected_identity,
        },
        "forensic_snapshots": {
            "samples": 40, "atomic_per_sample": True,
            "failure_forensic_only": True, "automatic_canonical_promotion": False,
        },
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    admission = {
        "schema": "m1227_final_capture_admission_r1_v1", "status": "PASS",
        "ordered": 9880, "attention": 480, "payload_files": 640,
        "execution": 7360, "operator_rows": 79, "atlif_live_rows": 93,
        "atlif_static": 105, "dead_sn_v": list(DEAD_SN_V),
        "claim_boundary": {"capture_only": True, "paper_result": False,
                           "cycles": False, "speedup": False, "energy": False, "ppa": False},
    }
    (staging / "m1227_admission.json").write_text(
        json.dumps(admission, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_double_seal(staging)


def validate_m1224():
    rows = verify_double_seal(M1224, M1224_MANIFEST_SHA256, M1224_OUTER_FILE_SHA256)
    require(rows.get("review.json") == M1224_REVIEW_SHA256, "M1224 review member SHA mismatch")
    review = strict_json(M1224 / "review.json")
    require(review.get("status") ==
            "PASS_AUDIT__EXACT40_STATIC_TOPOLOGY_ASSUMPTION_FALSE__PARTIAL_STAGING_ONLY",
            "M1224 semantic status mismatch")
    require(review["root_cause"]["arithmetic"]["runtime_live_unified_hook_modules_per_sample"] == 247 and
            review["root_cause"]["arithmetic"]["runtime_live_atlif_modules"] == 93 and
            review["root_cause"]["arithmetic"]["runtime_dead_atlif_modules"] == 12,
            "M1224 live/dead finding mismatch")
    return review


def validate_source_hammer(contract, policy):
    entry = contract["inputs"]["m1228_source_hammer"]
    path = safe_repo_path(entry["path"])
    require(path.parent.is_relative_to(HW / "reviews") if hasattr(path.parent, "is_relative_to")
            else str(path.parent).startswith(str(HW / "reviews")), "source hammer must be under reviews")
    rows = verify_double_seal(path, entry["manifest_sha256"], entry["outer_file_sha256"])
    require(rows.get("review.json") == entry["review_sha256"], "source hammer review SHA mismatch")
    review = strict_json(path / "review.json")
    require(review.get("schema") == HAMMER_SCHEMA and review.get("status") == "PASS",
            "source hammer semantic mismatch")
    require(review.get("source_sha256") == sha256(Path(__file__).resolve()) and
            review.get("contract_sha256") == sha256(SOURCE_CONTRACT) and
            review.get("test_sha256") == policy["test"]["sha256"] and
            review.get("authorization", {}).get("production_release") is True,
            "source hammer does not bind/authorize M1227")
    return review


def validate_final_selection(entry):
    root = safe_repo_path(entry["result_path"])
    rows = verify_double_seal(root, entry["manifest_sha256"], entry["outer_file_sha256"])
    member = entry["selection_member"]
    require(rows.get(member) == entry["selection_sha256"], "final selection member mismatch")
    selection = strict_json(root / member)
    require(selection.get("schema") == entry["selection_schema"], "final selection schema mismatch")
    selected = selection["selected"]
    require(type(selected["epoch"]) is int and selected["epoch"] >= 0,
            "final selection epoch must be a nonnegative integer")
    checkpoint = selected["checkpoint"]
    configuration = selection["configuration"]
    checkpoint_path = Path(checkpoint["absolute_path"])
    config_path = Path(configuration["absolute_path"])
    regular(checkpoint_path, "final selected checkpoint")
    regular(config_path, "final selected configuration")
    require(checkpoint_path.stat().st_size == checkpoint["size_bytes"] and
            checkpoint_path.stat().st_mtime_ns == checkpoint["mtime_ns"] and
            sha256(checkpoint_path) == checkpoint["sha256"], "final checkpoint identity drift")
    require(config_path.stat().st_size == configuration["size_bytes"] and
            config_path.stat().st_mtime_ns == configuration["mtime_ns"] and
            sha256(config_path) == configuration["sha256"], "final config identity drift")
    return {
        "selection": selection, "checkpoint_path": checkpoint_path, "config_path": config_path,
        "identity": {
            "epoch": selected["epoch"], "checkpoint_sha256": checkpoint["sha256"],
            "checkpoint_size_bytes": checkpoint["size_bytes"],
            "checkpoint_mtime_ns": checkpoint["mtime_ns"],
            "config_sha256": configuration["sha256"],
        },
    }


def resolve_sample(row):
    relative = Path(row["path"])
    require(not relative.is_absolute() and ".." not in relative.parts and
            relative.parts[:3] == PINNED_LINK_REL.parts and len(relative.parts) > 3,
            "sample must be below pinned DSEC link")
    link = ROOT / PINNED_LINK_REL
    cursor = ROOT
    for part in PINNED_LINK_REL.parts[:-1]:
        cursor = cursor / part
        require(os.path.lexists(str(cursor)) and not cursor.is_symlink(),
                "missing/symlink pre-link repository component: " + str(cursor))
    require(link.is_symlink() and os.readlink(str(link)) == str(PINNED_DSEC_ROOT),
            "pinned DSEC root link drift")
    require(link.resolve(strict=True) == PINNED_DSEC_ROOT and
            PINNED_DSEC_ROOT.resolve(strict=True) == PINNED_DSEC_ROOT,
            "pinned DSEC resolved root drift")
    cursor = link
    for part in relative.parts[len(PINNED_LINK_REL.parts):]:
        cursor = cursor / part
        require(os.path.lexists(str(cursor)) and not cursor.is_symlink(),
                "missing/symlink sample component")
    regular(cursor, "sample leaf")
    resolved = cursor.resolve(strict=True)
    require(str(resolved).startswith(str(PINNED_DSEC_ROOT) + os.sep), "sample escapes DSEC root")
    require(resolved.stat().st_size == row["bytes"] and sha256(resolved) == row["sha256"],
            "sample identity drift")
    return dict(row, resolved_path=str(resolved))


def validate_cohort(rows):
    require(len(rows) == 40 and [row["global_sample_id"] for row in rows] == list(range(40)),
            "cohort must be ordered samples 0..39")
    require([row["cohort"] for row in rows[:10]] == ["c1"] * 10 and
            [row["sequence"] for row in rows[:10]] == ["zurich_city_09_a"] * 10,
            "C1 cohort mismatch")
    require([row["cohort"] for row in rows[10:]] == ["decoder"] * 30 and
            [row["sequence"] for row in rows[10:]] ==
            [sequence for sequence in SEQUENCES for _ in range(10)],
            "decoder cohort mismatch")
    require(len(set(row["path"] for row in rows)) == 40 and
            len(set(row["sha256"] for row in rows)) == 40 and
            len(set(row["sample_key"] for row in rows)) == 40,
            "cohort identities must be unique")
    return [resolve_sample(row) for row in rows]


def validate_launch_contract(contract, contract_path):
    require(contract.get("schema") == LAUNCH_SCHEMA and contract.get("status") == LAUNCH_STATUS,
            "source-only or unhammered M1227 contract cannot launch")
    policy = strict_json(SOURCE_CONTRACT)
    require(policy.get("schema") == SOURCE_SCHEMA and policy.get("status") == SOURCE_STATUS,
            "canonical source policy mismatch")
    require(contract["contract_path"] == str(contract_path.relative_to(ROOT)),
            "launch contract path mismatch")
    require(contract["inputs"]["launcher"] == {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha256(Path(__file__).resolve()),
    }, "launch source identity mismatch")
    require(contract["inputs"]["source_contract"] == {
        "path": str(SOURCE_CONTRACT.relative_to(ROOT)), "sha256": sha256(SOURCE_CONTRACT),
    }, "source-contract identity mismatch")
    require(safe_repo_path(contract["one_shot"]["attempt_marker"], missing_leaf=True) == CANONICAL_ATTEMPT and
            safe_repo_path(contract["output"]["path"], missing_leaf=True) == CANONICAL_RESULT and
            safe_repo_path(contract["production_log"]["path"], missing_leaf=True) == CANONICAL_LOG and
            contract["gpu_ownership"]["lease_path"] == str(CANONICAL_LEASE.relative_to(ROOT)),
            "M1227 disjoint namespace mismatch")
    validate_m1224()
    validate_source_hammer(contract, policy)
    final = validate_final_selection(contract["inputs"]["final_selection_result"])
    samples = validate_cohort(contract["cohort"]["samples"])
    return dict(final, verified_samples=samples, policy=policy)


def run_capture(contract, binding, r1=None):
    if r1 is None:
        r1 = load_substrate()
    static_non_atlif = frozen_non_atlif_inventory(binding["policy"])
    writer_type = make_live_dead_writer(r1, static_non_atlif)
    writer_type.ACTIVE_WRITER = None
    original_init = writer_type.__init__

    def bind_writer(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        writer_type.ACTIVE_WRITER = self

    writer_type.__init__ = bind_writer
    r1.UnifiedHookWriter = writer_type
    r1.true = True
    original_load_source = r1.load_source
    original_selected_samples = r1.selected_samples
    original_seal = r1.write_double_seal

    def selected_samples(_contract):
        return binding["verified_samples"]

    def strict_load(name, path, expected_sha):
        module = original_load_source(name, path, expected_sha)
        if name == "m1174_profile":
            module.HardwareProfiler = make_snapshot_profiler(module.HardwareProfiler, writer_type)
        elif name == "m1174_bit_writer":
            module.AttentionBitTraceWriter = make_strict_attention_writer(module.AttentionBitTraceWriter)
        return module

    def final_seal(staging):
        final_validate_and_seal(staging, writer_type, binding["identity"])

    r1.load_source = strict_load
    r1.selected_samples = selected_samples
    r1.write_double_seal = final_seal
    substrate_contract = {
        "contract_path": contract["contract_path"],
        "inputs": {
            "profile": binding["policy"]["runtime_sources"]["profile"],
            "bit_writer": binding["policy"]["runtime_sources"]["bit_writer"],
        },
        "selected_identity": binding["identity"],
        "expected_topology": {"module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12}},
        "capture": {"attention_windows_per_call": contract["capture"]["attention_windows_per_call"]},
        "cohort": contract["cohort"],
        "output": contract["output"],
    }
    try:
        output = r1.run_capture(substrate_contract, binding)
        require(output == CANONICAL_RESULT, "substrate returned noncanonical result")
        return output
    finally:
        r1.load_source = original_load_source
        r1.selected_samples = original_selected_samples
        r1.write_double_seal = original_seal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    require(str(contract_path).startswith(str(ROOT) + os.sep), "launch contract must be in repository")
    contract = strict_json(contract_path)
    binding = validate_launch_contract(contract, contract_path)
    require(not os.path.lexists(str(CANONICAL_ATTEMPT)) and
            not os.path.lexists(str(CANONICAL_RESULT)) and
            not os.path.lexists(str(CANONICAL_LOG)), "fresh M1227 namespace required")
    r1 = load_substrate()
    with r1.exclusive_gpu_lease(CANONICAL_LEASE):
        descriptor = os.open(str(CANONICAL_ATTEMPT), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        try:
            os.write(descriptor, ATTEMPT_TOKEN.encode("ascii"))
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        output = run_capture(contract, binding, r1=r1)
    verify_double_seal(output)
    print(PASS_TOKEN + " " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
