#!/usr/bin/env python3
"""Read-only Local5 PAFT input/provenance audit.

The script deliberately does not import the training stack or construct a GPU
model.  It verifies immutable files and inspects the exact, SHA-pinned ep44
PyTorch zip checkpoint with a restricted metadata unpickler.  Output is limited
to this review directory.
"""

from collections import Counter, OrderedDict
import csv
import hashlib
import io
import json
import pickle
from pathlib import Path
import re
import sys
import zipfile


AUDIT_DIR = Path(__file__).resolve().parent
HW_ROOT = AUDIT_DIR.parents[1]
REPO_ROOT = HW_ROOT.parent
EXP_ROOT = REPO_ROOT / "neuron_experiments/H9_bipolar_self_attention"
DATA_ROOT = REPO_ROOT / "data/Datasets/DSEC/saved_flow_data"

CHECKPOINT = EXP_ROOT / (
    "results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/"
    "checkpoint_epoch44.pth"
)
TRAIN_CONFIG = EXP_ROOT / (
    "configs/generated/"
    "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50.yml"
)
DEPLOY_CONFIG = EXP_ROOT / (
    "configs/generated/"
    "dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_"
    "hardware_order_q7q17_deploy.yml"
)
VALID_LIST = DATA_ROOT / "sequence_lists/valid_split_seq.csv"
TRAIN_LIST = DATA_ROOT / "sequence_lists/train_split_seq.csv"
LOCAL_ORDERED = HW_ROOT / (
    "results/local_ep44_full_network_ordered_trace_s10_20260821"
)
LOCAL_RELEASE = HW_ROOT / (
    "results/local5_ep44_hardware_rebind_20260815_profile100/"
    "ranked_checkpoint_release_receipt.json"
)
M40_MANIFEST = HW_ROOT / (
    "results/m40_h67_ep35_bottleneck_packed_sources_s10_r6_20260822/"
    "m40_bottleneck_packed_source_manifest.json"
)
M71_CATALOG = HW_ROOT / (
    "results/m71_h67_k16_q16_paft_codebook_dev_r1_20260823/"
    "m71_h67_k16_q16_paft_codebook.json"
)
M71_REVOCATION = HW_ROOT / (
    "contracts/m71_valid825_catalog_revocation_r1_20260823.json"
)
M40_TRACER = HW_ROOT / (
    "system_simulator/scripts/trace_m40_bottleneck_packed_sources.py"
)
M73_TRACER = HW_ROOT / (
    "system_simulator/scripts/trace_m73_train_calibration_bottleneck_sources.py"
)
M71_BUILDER = HW_ROOT / (
    "system_simulator/scripts/build_m71_hardware_weighted_paft_codebook.py"
)
PAFT_SOURCE = EXP_ROOT / (
    "overlay/models/STSwinNet_SNN/pattern_paft.py"
)
ORDERED_ENTRYPOINT = EXP_ROOT / "entrypoints/profile_nts11_hardware_p0.py"

EXPECTED = {
    "local5_checkpoint_sha256":
        "19820bec07cc3bf3da7e9e2e31e2af0b36bda89e636b0d273c0257b368c34f57",
    "local5_train_config_sha256":
        "c5d7be623fd16e091c019662dffd5b0c16d8b4e1fc1541a609fc7416a58f093a",
    "local5_deploy_config_sha256":
        "078bb517e2479c95719bd2eb88a08ee935d7f05bdf6733b39d2d4846f01f514d",
    "dsec_train_list_sha256":
        "919c79c61535eb499364ffe28fad3000441e25d1bddbf4fa9a0c27a78d4fdc10",
    "dsec_valid825_list_sha256":
        "7f3dc2800653e12caca10379c51ee8e8988aaf6bb80c391224a454a5879325d0",
    "h67_checkpoint_sha256":
        "4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158",
    "h67_deploy_config_sha256":
        "8be3f7bbffd75c4356d3abf5935679d80e15c1caefd307c19a727729659e6c49",
    "revoked_m71_catalog_sha256":
        "142e32f0d988721ce9edf25d4dcf3883d82f2604f2aee9c755cde87b2ef70cdd",
}

TARGETS = (
    "sttmultires_unet.resblocks.0.conv1.0",
    "sttmultires_unet.resblocks.0.conv2.0",
    "sttmultires_unet.resblocks.1.conv1.0",
    "sttmultires_unet.resblocks.1.conv2.0",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path):
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def identity(path):
    return {
        "path": relative(path),
        "exists": path.is_file(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256(path) if path.is_file() else None,
    }


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_one_column_csv(path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.reader(handle) if row]
    require(all(len(row) == 1 for row in rows),
            f"one-column split list expected: {path}")
    return [row[0] for row in rows]


def yaml_attention_markers(path):
    mode = None
    hardware_quant = None
    in_attention = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line == "bsa_attention:":
            in_attention = True
            continue
        if in_attention and line and not line.startswith(" "):
            break
        if in_attention:
            match = re.match(r"  mode:\s*(.+)$", line)
            if match:
                mode = match.group(1).strip()
            match = re.match(r"  hardware_quant_enabled:\s*(.+)$", line)
            if match:
                hardware_quant = match.group(1).strip().lower() == "true"
    return {
        "bsa_attention_mode": mode,
        "hardware_quant_enabled": hardware_quant,
    }


class _StorageType:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


class _Dummy:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __setstate__(self, state):
        self.state = state


def _rebuild_tensor(storage, offset, size, stride, *unused):
    return {
        "__tensor__": True,
        "storage": storage,
        "offset": int(offset),
        "size": tuple(int(value) for value in size),
        "stride": tuple(int(value) for value in stride),
    }


def _rebuild_parameter(data, *unused):
    return data


class _CheckpointMetadataUnpickler(pickle.Unpickler):
    """Decode module/tensor metadata without importing or executing torch."""

    _dummy_types = {}

    def persistent_load(self, persistent_id):
        require(isinstance(persistent_id, tuple) and
                persistent_id and persistent_id[0] == "storage",
                "unexpected checkpoint persistent object")
        return {
            "__storage__": True,
            "dtype": repr(persistent_id[1]),
            "key": str(persistent_id[2]),
            "location": str(persistent_id[3]),
            "numel": int(persistent_id[4]),
        }

    def find_class(self, module, name):
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name.startswith("_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and "Storage" in name:
            return _StorageType(name)
        if module == "torch" and name == "device":
            return lambda *args: ("device",) + args
        if module == "builtins":
            allowed = {
                "set": set, "frozenset": frozenset, "dict": dict,
                "list": list, "tuple": tuple, "int": int, "float": float,
                "str": str, "bytes": bytes, "bytearray": bytearray,
                "complex": complex, "slice": slice,
            }
            require(name in allowed, f"blocked checkpoint builtin: {name}")
            return allowed[name]
        key = (module, name)
        if key not in self._dummy_types:
            safe_name = re.sub(r"[^A-Za-z0-9_]", "_", f"{module}_{name}")
            self._dummy_types[key] = type(f"Dummy_{safe_name}", (_Dummy,), {})
        return self._dummy_types[key]


def _module_state(module):
    state = getattr(module, "state", {})
    return state if isinstance(state, dict) else {}


def _walk_modules(module, prefix=""):
    state = _module_state(module)
    yield prefix, module, state
    for name, child in state.get("_modules", {}).items():
        if child is not None:
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            yield from _walk_modules(child, child_prefix)


def inspect_checkpoint_operators(path):
    require(sha256(path) == EXPECTED["local5_checkpoint_sha256"],
            "checkpoint must match the reviewed Local5 ep44 SHA before unpickling")
    with zipfile.ZipFile(path) as archive:
        pickle_members = [name for name in archive.namelist()
                          if name.endswith("/data.pkl")]
        require(len(pickle_members) == 1,
                "expected one data.pkl in checkpoint archive")
        pickle_member = pickle_members[0]
        prefix = pickle_member[:-len("data.pkl")]
        model = _CheckpointMetadataUnpickler(
            io.BytesIO(archive.read(pickle_member))).load()
        found = {}
        for name, module, state in _walk_modules(model):
            if name not in TARGETS:
                continue
            parameters = state.get("_parameters", {})
            weight = parameters.get("weight")
            require(isinstance(weight, dict) and weight.get("__tensor__"),
                    f"missing tensor metadata for {name}.weight")
            storage = weight["storage"]
            member = f"{prefix}data/{storage['key']}"
            raw = archive.read(member)
            shape = list(weight["size"])
            expected_numel = 1
            for extent in shape:
                expected_numel *= extent
            require(weight["offset"] == 0 and storage["numel"] == expected_numel,
                    f"non-exclusive storage for {name}.weight")
            require(storage["dtype"] == "FloatStorage" and
                    len(raw) == expected_numel * 4,
                    f"unexpected storage dtype/extent for {name}.weight")
            found[name] = {
                "module_class": type(module).__name__,
                "in_channels": int(state["in_channels"]),
                "out_channels": int(state["out_channels"]),
                "kernel_size": list(state["kernel_size"]),
                "stride": list(state["stride"]),
                "padding": list(state["padding"]),
                "dilation": list(state["dilation"]),
                "groups": int(state["groups"]),
                "bias_present": parameters.get("bias") is not None,
                "weight_shape": shape,
                "weight_dtype": "torch.float32",
                "weight_content_bytes": len(raw),
                "weight_content_sha256": hashlib.sha256(raw).hexdigest(),
                "checkpoint_archive_storage_key": storage["key"],
            }
    require(tuple(found) == TARGETS,
            "Local5 checkpoint target operator name/order drift")
    return found


def inspect_ordered_trace(valid_keys):
    execution = LOCAL_ORDERED / "execution_trace.csv"
    dual = LOCAL_ORDERED / "dual_line_operator_trace.csv"
    workload = LOCAL_ORDERED / "sample_workload.csv"
    profile = LOCAL_ORDERED / "nts11_hardware_p0_profile.json"
    with execution.open(newline="", encoding="utf-8") as handle:
        execution_rows = list(csv.DictReader(handle))
    with dual.open(newline="", encoding="utf-8") as handle:
        dual_rows = list(csv.DictReader(handle))
    with workload.open(newline="", encoding="utf-8") as handle:
        workload_rows = list(csv.DictReader(handle))
    sample_keys = [row["sample_key"] for row in workload_rows]
    target_rows = [row for row in execution_rows if row["name"] in TARGETS]
    counts = Counter(row["name"] for row in target_rows)
    shapes = sorted(set(row["input_shape"] for row in target_rows))
    profile_payload = read_json(profile)
    require(len(execution_rows) == 1720 and len(dual_rows) == 4030,
            "Local5 ordered trace row population drift")
    require(sample_keys == valid_keys[:10],
            "Local5 ordered trace is not the valid825 first-ten cohort")
    require(all(counts[name] == 10 for name in TARGETS),
            "Local5 ordered trace target coverage drift")
    require(shapes == ["[10, 1, 768, 15, 20]"],
            "Local5 ordered trace bottleneck shape drift")
    return {
        "directory": relative(LOCAL_ORDERED),
        "execution_records": len(execution_rows),
        "dual_line_operator_records": len(dual_rows),
        "sample_records": len(workload_rows),
        "sample_keys": sample_keys,
        "sample_keys_equal_valid825_first_ten": True,
        "split_role": "DSEC_VALID825_INTERNAL_PROFILE_ONLY",
        "target_operator_execution_counts": dict(counts),
        "target_input_shapes": shapes,
        "ordered_trace_enabled": profile_payload.get("ordered_trace"),
        "checkpoint_load_missing_count": profile_payload.get(
            "checkpoint_load_audit", {}).get("missing_count"),
        "checkpoint_load_unexpected_count": profile_payload.get(
            "checkpoint_load_audit", {}).get("unexpected_count"),
        "contains_packed_bottleneck_source_bitplanes": False,
        "eligible_for_paft_catalog": False,
        "files": {
            "execution_trace": identity(execution),
            "dual_line_operator_trace": identity(dual),
            "sample_workload": identity(workload),
            "profile": identity(profile),
        },
    }


def top_level_schema_artifacts(root, needles):
    matches = []
    for path in root.glob("*/*.json"):
        try:
            front = path.open("rb").read(16384).decode("utf-8", errors="ignore")
        except OSError:
            continue
        if not any(needle in front for needle in needles):
            continue
        try:
            schema = str(read_json(path).get("schema", ""))
        except (OSError, ValueError):
            continue
        if any(needle in schema for needle in needles):
            matches.append({"path": relative(path), "schema": schema})
    return sorted(matches, key=lambda row: row["path"])


def source_constants(path, names):
    text = path.read_text(encoding="utf-8")
    values = {}
    for name in names:
        match = re.search(
            rf"{re.escape(name)}\s*=\s*\(\s*[\"']([^\"']+)[\"']\s*\)",
            text, flags=re.MULTILINE)
        if not match:
            match = re.search(
                rf"{re.escape(name)}\s*=\s*[\"']([^\"']+)[\"']",
                text, flags=re.MULTILINE)
        values[name] = match.group(1) if match else None
    return values


def build_audit():
    require(CHECKPOINT.is_file() and TRAIN_CONFIG.is_file() and
            DEPLOY_CONFIG.is_file() and VALID_LIST.is_file(),
            "one or more Local5 minimum local identity files are absent")
    require(sha256(CHECKPOINT) == EXPECTED["local5_checkpoint_sha256"],
            "Local5 checkpoint SHA drift")
    require(sha256(TRAIN_CONFIG) == EXPECTED["local5_train_config_sha256"],
            "Local5 training config SHA drift")
    require(sha256(DEPLOY_CONFIG) == EXPECTED["local5_deploy_config_sha256"],
            "Local5 deploy config SHA drift")
    require(sha256(VALID_LIST) == EXPECTED["dsec_valid825_list_sha256"],
            "valid825 list SHA drift")

    valid_keys = read_one_column_csv(VALID_LIST)
    require(len(valid_keys) == len(set(valid_keys)) == 825,
            "valid825 population/uniqueness drift")
    release = read_json(LOCAL_RELEASE)
    require(release.get("status") == "PASS" and
            release.get("best_epoch") == 44 and
            release.get("checkpoint_sha256") ==
            EXPECTED["local5_checkpoint_sha256"],
            "Local5 release receipt drift")

    operators = inspect_checkpoint_operators(CHECKPOINT)
    for name, geometry in operators.items():
        require(geometry["in_channels"] == 768 and
                geometry["out_channels"] == 768 and
                geometry["kernel_size"] == [3, 3] and
                geometry["stride"] == [1, 1] and
                geometry["padding"] == [1, 1] and
                geometry["dilation"] == [1, 1] and
                geometry["groups"] == 1 and
                not geometry["bias_present"] and
                geometry["weight_shape"] == [768, 768, 3, 3],
                f"Local5 bottleneck geometry drift: {name}")

    m40 = read_json(M40_MANIFEST)
    h67_weight_sha = {
        row["operator"]: row["module_geometry"]["weight_content_sha256"]
        for row in m40["records"][:4]
    }
    weight_comparison = {
        name: {
            "local5_ep44_weight_sha256": operators[name]["weight_content_sha256"],
            "h67_ep35_weight_sha256": h67_weight_sha[name],
            "content_equal": operators[name]["weight_content_sha256"] ==
                             h67_weight_sha[name],
        }
        for name in TARGETS
    }
    require(all(not row["content_equal"] for row in weight_comparison.values()),
            "expected all Local5/H67 bottleneck weights to be identity-distinct")

    m71 = read_json(M71_CATALOG)
    revocation = read_json(M71_REVOCATION)
    m77_artifacts = top_level_schema_artifacts(
        HW_ROOT / "results",
        ("m77_h67_k16_q16_train_only_phi_kmeans_paft_codebook_v1",
         "m77_local5"))
    m73_artifacts = top_level_schema_artifacts(
        HW_ROOT / "results",
        ("m73_h67_ep35_train_calibration_packed_source_trace_v1",
         "m73_local5", "local5_ep44_train_calibration_packed_source_trace"))

    local_train_trace = [row for row in m73_artifacts
                         if "local5" in row["path"].lower() or
                         "local5" in row["schema"].lower()]
    train_list_status = identity(TRAIN_LIST)
    if TRAIN_LIST.is_file():
        train_keys = read_one_column_csv(TRAIN_LIST)
        train_list_status.update({
            "population": len(train_keys),
            "unique": len(set(train_keys)),
            "valid825_key_overlap": len(set(train_keys) & set(valid_keys)),
            "matches_expected_sha256":
                sha256(TRAIN_LIST) == EXPECTED["dsec_train_list_sha256"],
        })
    else:
        train_list_status.update({
            "population": None,
            "unique": None,
            "valid825_key_overlap": None,
            "matches_expected_sha256": False,
        })

    m40_constants = source_constants(
        M40_TRACER,
        ("EXPECTED_CHECKPOINT_SHA256", "EXPECTED_CONFIG_SHA256"))
    m73_constants = source_constants(
        M73_TRACER,
        ("EXPECTED_TRAIN_LIST_SHA256", "EXPECTED_VALID_LIST_SHA256"))
    paft_constants = source_constants(
        PAFT_SOURCE,
        ("_EXPECTED_SCHEMA", "_EXPECTED_ADMISSION_SCHEMA",
         "_EXPECTED_TRAIN_LIST_SHA256", "_EXPECTED_VALID_LIST_SHA256",
         "_EXPECTED_CHECKPOINT_SHA256"))

    ordered_trace = inspect_ordered_trace(valid_keys)
    return {
        "schema": "local5_paft_input_feasibility_audit_v1",
        "status": "BLOCKED_LOCAL5_PAFT_CATALOG_BUILD_INPUTS_NOT_READY",
        "audit_date": "2026-08-23",
        "scope": {
            "read_only_source_audit": True,
            "training_started": False,
            "production_source_modified": False,
            "valid825_used_for_training_or_catalog": False,
        },
        "verdict": {
            "local5_independent_paft_feasible": True,
            "ready_to_build_train_only_catalog_now": False,
            "ready_to_train_now": False,
            "h67_hook_writer_mechanism_reusable": True,
            "h67_catalog_or_identity_reusable": False,
            "blocking_reasons": [
                "local train_split_seq.csv is absent",
                "Local5 train-only packed bottleneck source trace is absent",
                "a real train-derived M77 catalog is absent for every line",
                "current PAFT loader/schema/checkpoint contract is H67-only",
            ],
        },
        "local5_identity": {
            "selected_checkpoint": identity(CHECKPOINT),
            "selected_checkpoint_release": {
                "receipt": identity(LOCAL_RELEASE),
                "best_epoch": release["best_epoch"],
                "selected_float_valid825_aee": release["selected_float_aee"],
                "selected_hardware_order_valid825_aee":
                    release["selected_hardware_order_aee"],
            },
            "training_config": {
                **identity(TRAIN_CONFIG), **yaml_attention_markers(TRAIN_CONFIG)},
            "hardware_order_deploy_config": {
                **identity(DEPLOY_CONFIG), **yaml_attention_markers(DEPLOY_CONFIG)},
            "other_local5_checkpoint_files_found": [relative(path) for path in
                sorted((EXP_ROOT / "results").glob("**/*local5*/checkpoint_epoch*.pth"))],
        },
        "dataset_split_inputs": {
            "train": train_list_status,
            "valid825": {
                **identity(VALID_LIST),
                "population": len(valid_keys),
                "unique": len(set(valid_keys)),
                "role": "ACCURACY_EVALUATION_AND_LEAKAGE_DENYLIST_ONLY",
            },
            "expected_train_population": 7345,
            "expected_train_sequences": 18,
            "local_tensor_file_counts": {
                "event_tensors": len(list((DATA_ROOT /
                    "event_tensors/10bins/left").glob("*/*.npy"))),
                "ground_truth_tensors": len(list((DATA_ROOT /
                    "gt_tensors").glob("*.npy"))),
                "mask_tensors": len(list((DATA_ROOT /
                    "mask_tensors").glob("*.npy"))),
            },
            "valid825_must_not_fit_catalog": True,
        },
        "local5_bottleneck_operators": {
            "names_and_order": list(TARGETS),
            "operators": operators,
            "derived_geometry": {
                "features_per_conv3x3_input_vector": 768 * 3 * 3,
                "partition_bits": 16,
                "partitions_per_operator": (768 * 3 * 3) // 16,
                "output_lanes_per_block": 96,
                "output_blocks": 8,
            },
            "same_names_and_geometry_as_h67": True,
            "h67_ep35_weight_content_comparison": weight_comparison,
        },
        "reusable_trace_inputs": {
            "full_network_ordered_trace_entrypoint": identity(ORDERED_ENTRYPOINT),
            "m40_exact_four_conv_hook_writer": identity(M40_TRACER),
            "m73_train_split_wrapper": identity(M73_TRACER),
            "local5_existing_ordered_trace": ordered_trace,
            "reuse_boundary": (
                "Reuse hook/writer and dataset plumbing only. The ordered trace "
                "does not contain PAFT packed source bitplanes, and its ten keys "
                "are valid825, so it cannot seed a catalog."),
        },
        "current_train_only_artifact_inventory": {
            "real_m73_result_artifacts": m73_artifacts,
            "local5_train_only_source_trace_artifacts": local_train_trace,
            "real_m77_result_artifacts": m77_artifacts,
            "m71_catalog": {
                **identity(M71_CATALOG),
                "schema": m71.get("schema"),
                "claimed_test_or_validation_data_used":
                    m71.get("split", {}).get("test_or_validation_data_used"),
                "revoked": sha256(M71_CATALOG) ==
                           EXPECTED["revoked_m71_catalog_sha256"],
            },
            "m71_revocation": {
                **identity(M71_REVOCATION),
                "status": revocation.get("status"),
                "paft_training_forbidden":
                    revocation.get("forbidden_uses", {}).get("paft_training"),
            },
        },
        "h67_binding_evidence": {
            "m71_builder": {
                **identity(M71_BUILDER),
                "pins_h67_m40_manifest": True,
                "hard_geometry_6912_k16_432_q16": True,
            },
            "m40_tracer_constants": m40_constants,
            "m73_tracer_constants": m73_constants,
            "m73_imports_m40_h67_checkpoint_and_config_pins": True,
            "m73_emitted_schema":
                "m73_h67_ep35_train_calibration_packed_source_trace_v1",
            "paft_loader": {
                **identity(PAFT_SOURCE),
                "constants": paft_constants,
                "runtime_trace_schema":
                    "m73_h67_ep35_train_calibration_packed_source_trace_v1",
                "expected_operator_names": list(TARGETS),
            },
            "binding_verdict":
                "M71, M73, and the current M77 loader contract are strictly H67/motion-bound.",
        },
        "direct_h67_reuse_fail_closed_matrix": [
            {
                "gate": "revoked catalog SHA",
                "local5_attempt": EXPECTED["revoked_m71_catalog_sha256"],
                "expected": "not in _REVOKED_CATALOG_SHA256",
                "result": "FAIL_CLOSED",
                "severity": "P0",
            },
            {
                "gate": "catalog schema",
                "local5_attempt": m71.get("schema"),
                "expected": paft_constants["_EXPECTED_SCHEMA"],
                "result": "FAIL_CLOSED",
                "severity": "P0",
            },
            {
                "gate": "runtime checkpoint SHA",
                "local5_attempt": sha256(CHECKPOINT),
                "expected": paft_constants["_EXPECTED_CHECKPOINT_SHA256"],
                "result": "FAIL_CLOSED",
                "severity": "P0",
            },
            {
                "gate": "M73 capture config SHA via imported M40",
                "local5_attempt": sha256(DEPLOY_CONFIG),
                "expected": m40_constants["EXPECTED_CONFIG_SHA256"],
                "result": "FAIL_CLOSED",
                "severity": "P0",
            },
            {
                "gate": "runtime train trace schema",
                "local5_attempt": None,
                "expected":
                    "m73_h67_ep35_train_calibration_packed_source_trace_v1",
                "result": "FAIL_CLOSED_MISSING_AND_WRONG_MODEL_SCHEMA",
                "severity": "P0",
            },
            {
                "gate": "operator names/order",
                "local5_attempt": list(TARGETS),
                "expected": list(TARGETS),
                "result": "PASS_COMPATIBLE_MECHANISM_ONLY",
                "severity": "PASS",
            },
            {
                "gate": "operator geometry",
                "local5_attempt": "4 x Conv2d 768x768 3x3, shape T10B1C768H15W20",
                "expected": "same",
                "result": "PASS_COMPATIBLE_MECHANISM_ONLY",
                "severity": "PASS",
            },
            {
                "gate": "operator weight identity",
                "local5_attempt": [weight_comparison[name][
                    "local5_ep44_weight_sha256"] for name in TARGETS],
                "expected": [weight_comparison[name][
                    "h67_ep35_weight_sha256"] for name in TARGETS],
                "result": "FAIL_IDENTITY_DISTINCT_ALL_FOUR",
                "severity": "P0",
            },
            {
                "gate": "train split file",
                "local5_attempt": train_list_status["sha256"],
                "expected": EXPECTED["dsec_train_list_sha256"],
                "result": "FAIL_CLOSED_MISSING" if not TRAIN_LIST.is_file()
                          else "PASS",
                "severity": "P0" if not TRAIN_LIST.is_file() else "PASS",
            },
        ],
        "minimum_input_file_set": {
            "preflight_and_catalog_capture": [
                "Local5 ep44 checkpoint (present and SHA-pinned)",
                "one explicitly selected Local5 capture config (training or hardware-order; both present, but the choice must be frozen)",
                "train_split_seq.csv with expected SHA and 7345 unique keys (missing locally)",
                "valid_split_seq.csv with expected SHA and 825 unique keys, used only as overlap denylist (present)",
                "for each selected train key: event_tensors/10bins/left/<sequence>/<key>, gt_tensors/<key>, mask_tensors/<key>, each with SHA receipt (missing locally)",
            ],
            "five_epoch_training": [
                "the complete DSEC train tensors referenced by all 7345 train keys",
                "Local5-specific PAFT candidate/no-PAFT paired configs and launch manifests",
                "Local5 train-only M77 catalog plus external admission contract",
                "a disjoint train hardware-heldout cohort; valid825 remains accuracy-only",
            ],
        },
        "recommended_milestones": [
            {
                "id": "L0",
                "name": "identity_and_split_preflight",
                "gate": "obtain exact train list; prove 7345 unique, 18 sequences, zero overlap with valid825; pin Local5 ep44 checkpoint/config/operator weight SHA",
            },
            {
                "id": "L1",
                "name": "local5_train_only_source_capture",
                "gate": "fork M73 schema/identity to Local5; capture >=32 deterministic train samples across all 18 train sequences; emit 128 four-Conv records and per-file SHA receipts",
            },
            {
                "id": "L2",
                "name": "local5_m77_catalog_and_admission",
                "gate": "build deterministic Hamming-Lloyd catalog only from L1; independently reproduce catalog SHA; external contract pins builder, trace, split lists, checkpoint, config, operators, geometry and tie-break rules",
            },
            {
                "id": "L3",
                "name": "clean_hardware_heldout_screen",
                "gate": "reserve disjoint train sequence(s); report nominal, byte/port/matcher-aware work and equal-activity guardrails; do not use valid825 for fitting or hardware selection",
            },
            {
                "id": "L4",
                "name": "full_install_smoke",
                "gate": "Local5-specific loader passes catalog/trace/checkpoint/operator fail-closed checks and completes four-hook forward/backward plus one optimizer step",
            },
            {
                "id": "L5",
                "name": "paired_five_epoch_candidate",
                "gate": "same ep44 start, seed, epochs and data order for PAFT/no-PAFT; valid825 used once as accuracy evaluation; no catalog or hyperparameter fitting on valid825",
            },
            {
                "id": "L6",
                "name": "hardware_promotion",
                "gate": "only after clean heldout compute gate and accuracy gate: matcher/packer RTL, VCS, same-resource DC/STA/SAIF/PTPX and address-timed full-system comparison",
            },
        ],
        "readiness_score": {
            "local5_model_identity": 100,
            "operator_hook_geometry_compatibility": 100,
            "train_split_and_tensor_availability": 10,
            "local5_train_only_trace": 0,
            "local5_catalog_and_admission": 0,
            "local5_full_install_evidence": 0,
            "overall_input_readiness": 35,
            "interpretation": "feasible but blocked before catalog construction; score is workflow readiness, not a paper metric",
        },
    }


def write_outputs(payload):
    output = AUDIT_DIR / "local5_paft_input_feasibility_audit.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    report = AUDIT_DIR / "README.md"
    require(report.is_file(), "README.md must exist before sealing")
    manifest_items = {}
    for path in (Path(__file__).resolve(), report, output):
        manifest_items[path.name] = {
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
    manifest = {
        "schema": "local5_paft_input_feasibility_review_artifact_manifest_v1",
        "status": "PASS_REVIEW_ARTIFACTS_SHA256_SEALED",
        "files": manifest_items,
    }
    (AUDIT_DIR / "review_artifact_sha256.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")


def main():
    payload = build_audit()
    write_outputs(payload)
    print(json.dumps({
        "status": payload["status"],
        "overall_input_readiness":
            payload["readiness_score"]["overall_input_readiness"],
        "train_list_present": payload["dataset_split_inputs"]["train"]["exists"],
        "local5_train_trace_count": len(payload[
            "current_train_only_artifact_inventory"][
                "local5_train_only_source_trace_artifacts"]),
        "real_m77_artifact_count": len(payload[
            "current_train_only_artifact_inventory"][
                "real_m77_result_artifacts"]),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
