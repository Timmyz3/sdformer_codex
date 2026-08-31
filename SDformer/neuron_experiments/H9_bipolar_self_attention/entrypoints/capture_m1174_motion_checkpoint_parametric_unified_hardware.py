#!/opt/conda/envs/sdformerflow/bin/python
"""Checkpoint-parametric, one-load Motion hardware capture entrypoint.

M1174 is intentionally authored before the final M1171 selection receipt is
admitted.  The checked-in source contract therefore cannot launch production.
A successor launch contract must bind both the sealed M1171 result and its
different-author result hammer, the exact selected checkpoint/configuration,
all forty input tensors, a fresh output namespace, and retirement of the
legacy M511 watcher.

The production path constructs and loads the model exactly once, then runs the
fixed C1 cohort followed by the three decoder cohorts.  One hook population
emits a globally ordered record stream covering C1, decoder, ATLIF, FC1/FC2,
patch embedding, BatchNorm and Q/K/V/attention scopes.  Exact C1 and decoder
inputs retain compressed FP32 bytes plus sign/support bitmaps.  This is capture
evidence only: it does not calculate cycles, speedup, energy, PPA or accuracy.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import uuid
import zlib
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
SOURCE_CONTRACT = HW / "contracts/m1174_motion_checkpoint_parametric_unified_capture_source_contract_r1_20260830.json"
PROFILE = Path(__file__).with_name("profile_nts11_hardware_p0.py")
BIT_WRITER = Path(__file__).with_name("h67_bit_trace.py")
LEASE = HW / "results/gpu_profile_lease.lock"
LEGACY_MARKERS = (
    "capture_m511_h67_convtranspose_binary_inputs.py",
    "m511_capture_watcher",
    "run_m511_h67",
)
C1_TARGETS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)
SEQUENCES = ("interlaken_01_a", "thun_01_b", "zurich_city_12_a")
CATEGORIES = frozenset({
    "c1_conv3x3", "decoder_convtranspose", "atlif", "fc1", "fc2",
    "patch_embed", "batch_norm", "qkv", "attention",
})


class CaptureError(RuntimeError):
    pass


def require(value: bool, message: str) -> None:
    if not value:
        raise CaptureError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    def reject(token: str) -> None:
        raise CaptureError("non-standard JSON token: " + token)

    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"),
                       object_pairs_hook=pairs, parse_constant=reject)
    require(isinstance(value, dict), "JSON root must be an object")
    return value


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise CaptureError("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            "{} must be a non-symlink regular file: {}".format(label, path))


def directory(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise CaptureError("missing {}: {}".format(label, path)) from exc
    require(stat.S_ISDIR(mode) and not path.is_symlink(),
            "{} must be a non-symlink directory: {}".format(label, path))


def repo_path(relative: str, *, missing_leaf: bool = False) -> Path:
    value = Path(relative)
    require(not value.is_absolute() and ".." not in value.parts,
            "unsafe repository-relative path")
    candidate = ROOT / value
    cursor = ROOT
    for index, part in enumerate(value.parts):
        cursor = cursor / part
        leaf = index == len(value.parts) - 1
        if os.path.lexists(cursor):
            require(not cursor.is_symlink(), "symlink component rejected: " + str(cursor))
        else:
            require(missing_leaf and leaf, "missing path component: " + str(cursor))
    return candidate


def verify_double_seal(path: Path, expected_payloads: set[str] | None = None) -> dict[str, str]:
    directory(path, "sealed directory")
    manifest = path / "SHA256SUMS"
    outer = path / "SHA256SUMS.seal.sha256"
    regular(manifest, "manifest")
    regular(outer, "outer seal")
    fields = outer.read_text(encoding="utf-8").split()
    require(fields == [sha256(manifest), "SHA256SUMS"], "outer seal mismatch")
    rows: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        parts = line.split(None, 1)
        require(len(parts) == 2 and len(parts[0]) == 64, "invalid manifest row")
        name = parts[1].lstrip("*")
        require(Path(name).name == name and name not in rows, "unsafe/duplicate member")
        member = path / name
        regular(member, "sealed member " + name)
        require(sha256(member) == parts[0], "member SHA mismatch: " + name)
        rows[name] = parts[0]
    actual = {item.name for item in path.iterdir() if item.is_file() and
              item.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    require(actual == set(rows), "sealed member population mismatch")
    if expected_payloads is not None:
        require(set(rows) == expected_payloads, "sealed payload population mismatch")
    return rows


def running_legacy_watchers(proc_root: Path = Path("/proc")) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
            stat_fields = (entry / "stat").read_text(encoding="utf-8").split()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\x00", b" ").decode("utf-8", "replace")
        matches = [marker for marker in LEGACY_MARKERS if marker in command]
        if matches:
            found.append({"pid": int(entry.name), "state": stat_fields[2],
                          "markers": matches, "cmdline_sha256": hashlib.sha256(raw).hexdigest()})
    return sorted(found, key=lambda row: row["pid"])


@contextlib.contextmanager
def exclusive_gpu_lease(path: Path) -> Iterator[int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise CaptureError("shared GPU profile lease is busy") from exc
        watchers = running_legacy_watchers()
        require(not watchers,
                "legacy M511 watcher remains present (including SIGSTOP state): " + repr(watchers))
        yield descriptor
        require(not running_legacy_watchers(),
                "legacy M511 watcher appeared while unified lease was held")
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def validate_launch_contract(contract: dict[str, Any], contract_path: Path) -> dict[str, Any]:
    require(contract.get("schema") == "m1174_motion_checkpoint_parametric_unified_capture_launch_v1",
            "M1174 source-only contract is not a production launch authority")
    require(contract.get("status") ==
            "HAMMERED_M1171_BOUND__HAMMERED_M1174_SOURCE__ONE_GPU_RUN_AUTHORIZED",
            "M1174 production status is not authorized")
    inputs = contract["inputs"]
    regular(Path(__file__).resolve(), "running M1174 source")
    require(inputs["launcher"]["sha256"] == sha256(Path(__file__).resolve()),
            "running M1174 source SHA drift")
    regular(DOCS359, "protected docs/359")
    require(sha256(DOCS359) == DOCS359_SHA256 == inputs["docs359"]["sha256"],
            "protected docs/359 drift")
    require(contract["contract_path"] == str(contract_path.relative_to(ROOT)),
            "contract path binding mismatch")

    binder_dir = repo_path(inputs["m1171_binder_result"]["path"])
    result_rows = verify_double_seal(binder_dir, {
        "RUN_COMPLETE.txt", "e0_e8_rebind_targets.json",
        "final_checkpoint_selection.json", "five_checkpoint_metrics.csv",
    })
    require(result_rows["final_checkpoint_selection.json"] ==
            inputs["m1171_binder_result"]["selection_sha256"],
            "M1171 selection SHA drift")
    hammer = repo_path(inputs["m1171_result_hammer"]["path"])
    hammer_rows = verify_double_seal(hammer)
    require(inputs["m1171_result_hammer"]["review_member"] in hammer_rows,
            "M1171 hammer review member absent")
    require(hammer_rows[inputs["m1171_result_hammer"]["review_member"]] ==
            inputs["m1171_result_hammer"]["review_sha256"],
            "M1171 hammer review SHA drift")

    selection = strict_json(binder_dir / "final_checkpoint_selection.json")
    require(selection.get("schema") == "m1167_motion_final_checkpoint_selection_rebind_binder_r3_v1",
            "M1171 selected receipt schema drift")
    selected = selection["selected"]
    checkpoint = selected["checkpoint"]
    expected = contract["selected_identity"]
    require(type(selected["epoch"]) is int and selected["epoch"] == expected["epoch"] == 29,
            "final selected epoch mismatch")
    require(checkpoint["sha256"] == expected["checkpoint_sha256"] and
            checkpoint["size_bytes"] == expected["checkpoint_size_bytes"] and
            checkpoint["mtime_ns"] == expected["checkpoint_mtime_ns"],
            "final checkpoint identity mismatch")
    require(selection["configuration"]["sha256"] == expected["config_sha256"],
            "final config identity mismatch")
    checkpoint_path = Path(checkpoint["absolute_path"])
    config_path = Path(selection["configuration"]["absolute_path"])
    regular(checkpoint_path, "selected checkpoint")
    regular(config_path, "selected config")
    require(checkpoint_path.stat().st_size == checkpoint["size_bytes"] and
            checkpoint_path.stat().st_mtime_ns == checkpoint["mtime_ns"] and
            sha256(checkpoint_path) == checkpoint["sha256"],
            "selected checkpoint changed after binder")
    require(sha256(config_path) == expected["config_sha256"],
            "selected config changed after binder")
    return {"selection": selection, "checkpoint_path": checkpoint_path,
            "config_path": config_path, "binder_rows": result_rows,
            "hammer_rows": hammer_rows}


def load_source(name: str, path: Path, expected_sha: str) -> Any:
    regular(path, name)
    require(sha256(path) == expected_sha, name + " source SHA drift")
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot import " + name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    require(sha256(path) == expected_sha, name + " changed during import")
    return module


def tensor_stats(torch: Any, tensor: Any) -> dict[str, Any]:
    value = tensor.detach()
    elements = int(value.numel())
    finite = torch.isfinite(value) if value.is_floating_point() else torch.ones_like(value, dtype=torch.bool)
    return {
        "shape": [int(item) for item in value.shape],
        "stride": [int(item) for item in value.stride()],
        "dtype": str(value.dtype),
        "elements": elements,
        "bytes": elements * int(value.element_size()),
        "active": int(value.ne(0).sum().item()),
        "positive": int(value.gt(0).sum().item()),
        "negative": int(value.lt(0).sum().item()),
        "nonfinite": int((~finite).sum().item()),
    }


class UnifiedHookWriter:
    def __init__(self, torch: Any, staging: Path, contract: dict[str, Any]):
        self.torch = torch
        self.staging = staging
        self.contract = contract
        self.handles: list[Any] = []
        self.records: list[dict[str, Any]] = []
        self.module_inventory: dict[str, list[str]] = {
            category: [] for category in sorted(CATEGORIES)
        }
        self.sample: dict[str, Any] | None = None
        self.order = 0

    def begin(self, sample: dict[str, Any]) -> None:
        require(self.sample is None, "nested sample")
        self.sample = sample

    def end(self) -> None:
        require(self.sample is not None, "sample not active")
        self.sample = None

    def _category(self, name: str, module: Any) -> str | None:
        low = name.lower()
        cls = module.__class__.__name__.lower()
        if name in C1_TARGETS:
            return "c1_conv3x3"
        if cls == "convtranspose2d":
            return "decoder_convtranspose"
        if cls == "atlifternarypsn":
            return "atlif"
        if "patch_embed" in low:
            return "patch_embed"
        if "batchnorm" in cls:
            return "batch_norm"
        if "fc1" in low:
            return "fc1"
        if "fc2" in low:
            return "fc2"
        leaf = low.split(".")[-1]
        if leaf in {"q", "k", "v", "qkv", "query", "key", "value"}:
            return "qkv"
        if "attn" in low or "attention" in cls:
            return "attention"
        return None

    def _payload(self, name: str, category: str, tensor: Any, index: int) -> dict[str, Any]:
        if category not in {"c1_conv3x3", "decoder_convtranspose"}:
            return {"retained": False, "reason": "ordered statistics only"}
        import numpy as np
        value = tensor.detach().to(device="cpu", dtype=self.torch.float32).contiguous().numpy()
        raw = value.tobytes(order="C")
        stem = "s{:02d}_o{:05d}_{}".format(self.sample["global_sample_id"], index,
                                            hashlib.sha256(name.encode()).hexdigest()[:12])
        raw_name = "payloads/{}.fp32.zlib".format(stem)
        support_name = "payloads/{}.support_sign.le.bitpack".format(stem)
        (self.staging / raw_name).write_bytes(zlib.compress(raw, level=6))
        positive = np.packbits((value > 0).reshape(-1), bitorder="little").tobytes()
        negative = np.packbits((value < 0).reshape(-1), bitorder="little").tobytes()
        (self.staging / support_name).write_bytes(positive + negative)
        return {
            "retained": True, "raw_fp32_sha256": hashlib.sha256(raw).hexdigest(),
            "compressed_fp32": raw_name, "compressed_sha256": sha256(self.staging / raw_name),
            "support_sign": support_name, "support_sign_sha256": sha256(self.staging / support_name),
            "positive_plane_bytes": len(positive), "negative_plane_bytes": len(negative),
        }

    def attach(self, model: Any) -> None:
        observed: set[str] = set()
        for name, module in model.named_modules():
            category = self._category(name, module)
            if category is None:
                continue
            observed.add(category)
            self.module_inventory[category].append(name)
            def hook(_module: Any, inputs: Any, output: Any, *, _name=name, _category=category) -> None:
                require(self.sample is not None, "hook fired outside sample")
                tensors = [item for item in inputs if self.torch.is_tensor(item)]
                require(tensors, "covered module has no tensor input: " + _name)
                index = self.order
                self.order += 1
                row = {
                    "global_order": index,
                    "global_sample_id": self.sample["global_sample_id"],
                    "cohort": self.sample["cohort"],
                    "sequence": self.sample["sequence"],
                    "sample_key": self.sample["sample_key"],
                    "source_sha256": self.sample["sha256"],
                    "category": _category, "name": _name,
                    "input": tensor_stats(self.torch, tensors[0]),
                }
                row["payload"] = self._payload(_name, _category, tensors[0], index)
                self.records.append(row)
            self.handles.append(module.register_forward_hook(hook))
        require(CATEGORIES <= observed,
                "required module categories absent: " + repr(sorted(CATEGORIES - observed)))

    def close(self) -> None:
        while self.handles:
            self.handles.pop().remove()


def selected_samples(contract: dict[str, Any]) -> list[dict[str, Any]]:
    rows = contract["cohort"]["samples"]
    require(len(rows) == 40 and [row["global_sample_id"] for row in rows] == list(range(40)),
            "unified cohort must contain ordered global sample ids 0..39")
    require([row["cohort"] for row in rows[:10]] == ["c1"] * 10,
            "C1 cohort must be first")
    for position, sequence in enumerate(SEQUENCES):
        subset = rows[10 + position * 10:20 + position * 10]
        require([row["sequence"] for row in subset] == [sequence] * 10,
                "decoder sequence order mismatch: " + sequence)
    observed: list[dict[str, Any]] = []
    for row in rows:
        path = repo_path(row["path"])
        regular(path, "cohort source")
        require(path.stat().st_size == row["bytes"] and sha256(path) == row["sha256"],
                "cohort source identity drift: " + row["path"])
        observed.append({**row, "resolved_path": str(path)})
    return observed


def write_double_seal(directory_path: Path) -> None:
    members = sorted(item.relative_to(directory_path) for item in directory_path.rglob("*")
                     if item.is_file() and item.name not in {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    manifest = directory_path / "SHA256SUMS"
    manifest.write_text("".join("{}  {}\n".format(sha256(directory_path / item), item.as_posix())
                                for item in members), encoding="utf-8")
    (directory_path / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(manifest)), encoding="utf-8")


def run_capture(contract: dict[str, Any], binding: dict[str, Any]) -> Path:
    # Heavy dependencies are imported only after the launch contract, binder,
    # hammer, watcher and shared-lease gates have all passed.
    profile = load_source("m1174_profile", PROFILE, contract["inputs"]["profile"]["sha256"])
    bit_module = load_source("m1174_bit_writer", BIT_WRITER,
                             contract["inputs"]["bit_writer"]["sha256"])
    torch = profile.torch
    import numpy as np

    samples = selected_samples(contract)
    output = repo_path(contract["output"]["path"], missing_leaf=True)
    require(not os.path.lexists(output), "fresh output namespace required")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    (staging / "payloads").mkdir()
    published = False
    profiler = None
    writer = None
    try:
        config, device = profile.load_config(binding["config_path"])
        require(str(device).startswith("cuda") and torch.cuda.is_available(),
                "M1174 production capture requires CUDA")
        model = profile.build_model(config, binding["checkpoint_path"], device)
        audit = profile.validate_h9_load_audit(model, config)
        require(audit is not None and int(audit.get("missing_count", 0)) == 0 and
                int(audit.get("unexpected_count", 0)) == 0, "checkpoint load is not exact")
        counts = profile.h9_module_counts(model)
        require(counts == contract["expected_topology"]["module_counts"],
                "selected topology count drift")
        bn_policy = config.get("test", {}).get("bn_policy", "running")
        bn_changed = profile.configure_batch_norm_evaluation(model, bn_policy)
        bit_writer = bit_module.AttentionBitTraceWriter(
            staging / "attention_qk", sample_limit=40,
            windows_per_call=contract["capture"]["attention_windows_per_call"],
            first_block_only=False)
        bit_writer.bind_run_context({
            "artifact_identity": contract["selected_identity"],
            "eval_protocol": {"bn_policy": bn_policy, "samples": 40},
            "module_counts": counts, "checkpoint_load_audit": audit,
            "source_sha256": {"profiler": sha256(PROFILE), "trace_writer": sha256(BIT_WRITER)},
        })
        profiler = profile.HardwareProfiler(model, ordered_trace=True,
                                             dual_line_trace=True,
                                             bit_trace_writer=bit_writer)
        profiler.attach()
        writer = UnifiedHookWriter(torch, staging, contract)
        writer.attach(model)
        with torch.no_grad():
            for row in samples:
                profile.functional.reset_net(model)
                profiler.begin_sample(row["global_sample_id"], sample_key=row["sample_key"],
                                      sequence_key=row["sequence"])
                writer.begin(row)
                array = np.load(row["resolved_path"], allow_pickle=False)
                require(array.shape == (10, 480, 640) and array.dtype == np.float32,
                        "raw input tensor identity drift")
                chunk = torch.from_numpy(array.copy()).unsqueeze(0)
                label = torch.zeros((1, 2, 480, 640), dtype=torch.float32)
                mask = torch.ones((1, 480, 640), dtype=torch.float32)
                x, _, _ = profile.preprocess_chunk(config, chunk, label, mask, None, device)
                model(x)
                torch.cuda.synchronize(device)
                writer.end()
        writer.close()
        profiler.close()
        categories = {row["category"] for row in writer.records}
        require(CATEGORIES <= categories, "runtime category population incomplete")
        ordered_path = staging / "unified_ordered_records.jsonl"
        ordered_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                        for row in writer.records), encoding="utf-8")
        summary = profiler.summary()
        (staging / "execution_trace.json").write_text(
            json.dumps(summary["execution_records"], sort_keys=True) + "\n", encoding="utf-8")
        (staging / "operator_runtime.json").write_text(
            json.dumps(summary["operator_rows"], sort_keys=True) + "\n", encoding="utf-8")
        (staging / "atlif_activity.json").write_text(
            json.dumps(summary["atlif_rows"], sort_keys=True) + "\n", encoding="utf-8")
        manifest = {
            "schema": "m1174_motion_checkpoint_parametric_unified_capture_v1",
            "status": "CAPTURE_COMPLETE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM",
            "identity": {"contract_sha256": sha256(repo_path(contract["contract_path"])),
                         "selection": binding["selection"], "checkpoint_load_audit": audit,
                         "module_counts": counts, "bn_policy": bn_policy,
                         "bn_modules_changed": bn_changed},
            "cohort": {"samples": samples, "population": 40,
                       "c1_samples": 10, "decoder_sequences": list(SEQUENCES),
                       "decoder_samples_per_sequence": 10},
            "ordered_population": {"records": len(writer.records),
                                   "categories": sorted(categories)},
            "module_inventory": writer.module_inventory,
            "qkv_architecture": {
                "module_names": writer.module_inventory["qkv"],
                "explicit_q": any(name.lower().split(".")[-1] in {"q", "query"}
                                  for name in writer.module_inventory["qkv"]),
                "explicit_k": any(name.lower().split(".")[-1] in {"k", "key"}
                                  for name in writer.module_inventory["qkv"]),
                "explicit_v": any(name.lower().split(".")[-1] in {"v", "value"}
                                  for name in writer.module_inventory["qkv"]),
                "fused_qkv": any(name.lower().split(".")[-1] == "qkv"
                                 for name in writer.module_inventory["qkv"]),
                "absence_is_explicit_if_false": true
            },
            "files": {"ordered": "unified_ordered_records.jsonl",
                      "execution": "execution_trace.json",
                      "operator": "operator_runtime.json",
                      "atlif": "atlif_activity.json",
                      "attention": str(bit_writer.manifest_path.relative_to(staging))},
            "claim_boundary": {"capture_only": True, "accuracy": False,
                               "cycles": False, "speedup": False,
                               "system_speedup": False, "energy": False,
                               "rtl": False, "ppa": False,
                               "fresh_result_hammer_required": True},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8")
        (staging / "RUN_COMPLETE.txt").write_text(
            "PASS_M1174_UNIFIED_CAPTURE__FRESH_RESULT_HAMMER_REQUIRED__NO_HARDWARE_CLAIM\n",
            encoding="utf-8")
        write_double_seal(staging)
        os.replace(staging, output)
        published = True
        return output
    except BaseException as error:
        if writer is not None:
            with contextlib.suppress(BaseException):
                writer.close()
        if profiler is not None:
            with contextlib.suppress(BaseException):
                profiler.close()
        if not published:
            (staging / "FAILED.json").write_text(json.dumps({
                "status": "FAIL_CLOSED_NO_CANONICAL_RESULT",
                "reason": "{}: {}".format(type(error).__name__, error),
            }, indent=2) + "\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    require(contract_path.is_relative_to(ROOT), "contract must be inside repository")
    contract = strict_json(contract_path)
    binding = validate_launch_contract(contract, contract_path)
    attempt = repo_path(contract["one_shot"]["attempt_marker"], missing_leaf=True)
    require(not os.path.lexists(attempt), "M1174 attempt already consumed")
    # The stopped legacy watcher is rejected inside the lease before any attempt
    # is consumed or heavy module is imported.
    with exclusive_gpu_lease(repo_path(contract["gpu_ownership"]["lease_path"])):
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        descriptor = os.open(attempt, flags, 0o400)
        os.write(descriptor, b"M1174_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n")
        os.fsync(descriptor)
        os.close(descriptor)
        output = run_capture(contract, binding)
    print("PASS M1174 " + str(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
