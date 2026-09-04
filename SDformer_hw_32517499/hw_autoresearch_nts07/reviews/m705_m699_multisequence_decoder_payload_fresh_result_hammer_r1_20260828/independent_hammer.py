#!/usr/bin/env python3
"""Read-only M705 fresh-result hammer for canonical M699 S3x10 payload.

This audit never imports torch, executes the model, touches a GPU, or invokes
RTL/EDA.  It reconstructs every admitted FP32 plane from the frozen bitpack in
order to verify the raw-content digest and recomputes all density summaries.
"""

import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import struct
import tempfile

import numpy as np


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[2]
HW = ROOT / "hw_autoresearch_nts07"
OUTPUT = HW / "system_handoff/outgoing/m699_h67_ep35_multisequence_decoder_payload_s3x10_r1_20260828"
ATTEMPT = HW / "results/.m699_h67_ep35_multisequence_decoder_payload_r1_attempt_consumed"
CONTRACT = HW / "contracts/m699_h67_ep35_multisequence_decoder_payload_contract_r1_20260828.json"
RUNNER = HW / "system_handoff/scripts/run_m699_h67_ep35_multisequence_decoder_payload_one_shot.sh"
STATIC_REVIEW = HW / "reviews/m700_m699_multisequence_decoder_capture_fresh_static_hammer_r1_20260828"
M686 = HW / "system_handoff/outgoing/m686r6_h67_ep35_layer_static_decoder_payload_s10_r1_20260828"
M692 = HW / "reviews/m692_m686r6_s10_payload_fresh_result_hammer_r1_20260828"

EXPECTED_MANIFEST_SHA = "e2d7c92a038c213b590603ff534a33f3579bf1224cc3f56c11629e1d4c813dc0"
EXPECTED_OUTPUT_OUTER_FILE_SHA = "eaf975a9a1a4829b2c0a2251e7ef297abd53b83b30e23630e5ce51db5c5de18c"
EXPECTED_CONTRACT_SHA = "43d3b024c1a78d8bc2422af3846c9a376a67bedbecb2ff7396a17bc51ec68fc7"
EXPECTED_RUNNER_SHA = "9c0e8052577fce7e306ee41bae1a9c27434d0779511a1e6f910bfa5bdf75b958"
EXPECTED_DOCS359_SHA = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"
EXPECTED_STATIC_REVIEW_SHA = "ead51ecff1f80f112acfaa7ab39d4a8c7a25d33507ae6e8d39f38a2fa1f7cd3f"
EXPECTED_STATIC_OUTER_FILE_SHA = "8d278f4ca81cfafcd5b4702b8fa51c4e15b74edf249e9df94865e29c24589ff9"
EXPECTED_M686_MANIFEST_SHA = "c06de650b50db92dd0c374b57f0ce3ea72cfb3dcd18a369aea7d552341e5bb33"
EXPECTED_M686_OUTER_FILE_SHA = "e468b03a60a0531c95555908cef5aaffbc9b7e8887a14f37b985186642354592"

SEQUENCES = ["interlaken_01_a", "thun_01_b", "zurich_city_12_a"]
NAMES = {
    0: "sttmultires_unet.decoders.0.deconv.0",
    1: "sttmultires_unet.decoders.1.deconv.0",
    2: "sttmultires_unet.decoders.2.deconv.0",
    3: "sttmultires_unet.decoders.3.deconv.0",
}
SHAPES = {
    0: [10, 1, 1536, 15, 20],
    1: [10, 1, 770, 30, 40],
    2: [10, 1, 386, 60, 80],
    3: [10, 1, 194, 120, 160],
}
PACKED_BYTES = {module: int(np.prod(shape, dtype=np.int64)) // 8
                for module, shape in SHAPES.items()}


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path):
    def reject(token):
        raise RuntimeError("non-standard JSON token: " + token)

    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def safe_member(name):
    member = PurePosixPath(name)
    require(not member.is_absolute() and member.parts and
            ".." not in member.parts and member.parts[0] not in ("", "."),
            "unsafe member: " + str(name))
    return member


def tree_inventory(directory):
    directory = Path(directory)
    require(directory.is_dir() and not directory.is_symlink(),
            "unsafe tree root: " + str(directory))
    files, directories = set(), {"."}
    for base, dirnames, filenames in os.walk(str(directory), followlinks=False):
        base_path = Path(base)
        for name in dirnames:
            path = base_path / name
            observed = os.lstat(str(path))
            require(stat.S_ISDIR(observed.st_mode) and
                    not stat.S_ISLNK(observed.st_mode),
                    "symlink/non-directory in tree: " + str(path))
            directories.add(path.relative_to(directory).as_posix())
        for name in filenames:
            path = base_path / name
            observed = os.lstat(str(path))
            require(stat.S_ISREG(observed.st_mode) and
                    not stat.S_ISLNK(observed.st_mode),
                    "symlink/non-file in tree: " + str(path))
            files.add(path.relative_to(directory).as_posix())
    return files, directories


def verify_double_seal(directory, complete=True):
    directory = Path(directory)
    files, _ = tree_inventory(directory)
    sums = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    require(sums.is_file() and outer.is_file(), "missing double seal")
    require(outer.read_text(encoding="utf-8").strip().split() ==
            [sha256(sums), "SHA256SUMS"], "outer seal mismatch")
    sealed = set()
    for raw in sums.read_text(encoding="utf-8").splitlines():
        fields = raw.split(None, 1)
        require(len(fields) == 2, "malformed SHA256SUMS line")
        expected, raw_name = fields
        name = safe_member(raw_name.strip()).as_posix()
        require(name not in sealed, "duplicate sealed member: " + name)
        path = directory / name
        observed = os.lstat(str(path))
        require(stat.S_ISREG(observed.st_mode) and
                not stat.S_ISLNK(observed.st_mode),
                "unsafe sealed member: " + name)
        require(sha256(path) == expected, "member digest mismatch: " + name)
        sealed.add(name)
    if complete:
        require(sealed == files - {"SHA256SUMS", "SHA256SUMS.seal.sha256"},
                "sealed population mismatch")
    return {"members": len(sealed), "manifest_sha256": sha256(sums),
            "outer_seal_file_sha256": sha256(outer)}


def reseal(directory):
    directory = Path(directory)
    files, _ = tree_inventory(directory)
    members = sorted(files - {"SHA256SUMS", "SHA256SUMS.seal.sha256"})
    (directory / "SHA256SUMS").write_text("".join(
        "{}  {}\n".format(sha256(directory / name), name)
        for name in members), encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        "{}  SHA256SUMS\n".format(sha256(directory / "SHA256SUMS")),
        encoding="utf-8")


def product(shape):
    value = 1
    for dim in shape:
        require(isinstance(dim, int) and not isinstance(dim, bool) and dim > 0,
                "invalid shape")
        value *= dim
    return value


def c_stride(shape):
    return [product(shape[index + 1:]) if index + 1 < len(shape) else 1
            for index in range(len(shape))]


def selected_indices(population):
    return [round(index * (population - 1) / 9) for index in range(10)]


def parse_key_values(path):
    result = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        key, value = raw.split("=", 1)
        require(key not in result, "duplicate receipt key: " + key)
        result[key] = value
    return result


def reconstruct_payload(path, elements, value_word):
    """Return popcount and SHA256 of reconstructed C-order FP32-LE values."""
    path = Path(path)
    raw_digest = hashlib.sha256()
    packed_digest = hashlib.sha256()
    ones = 0
    consumed_bits = 0
    lut = np.asarray([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    with path.open("rb") as handle:
        while True:
            block = handle.read(1 << 18)
            if not block:
                break
            packed_digest.update(block)
            packed = np.frombuffer(block, dtype=np.uint8)
            ones += int(lut[packed].sum(dtype=np.uint64))
            bits = np.unpackbits(packed, bitorder="little")
            remaining = elements - consumed_bits
            require(remaining > 0, "bitpack exceeds declared element range")
            take = min(int(bits.size), remaining)
            words = bits[:take].astype("<u4", copy=False) * np.uint32(value_word)
            raw_digest.update(words.astype("<u4", copy=False).tobytes(order="C"))
            consumed_bits += take
            if take < bits.size:
                require(not np.any(bits[take:]), "nonzero tail padding")
    require(consumed_bits == elements, "bitpack shorter than element range")
    return {"ones": ones, "zeros": elements - ones,
            "packed_sha256": packed_digest.hexdigest(),
            "raw_fp32_sha256": raw_digest.hexdigest()}


def validate_manifest_semantics(manifest, contract, verify_payload=False,
                                verify_sources=False):
    require(manifest["schema"] ==
            "m699_h67_ep35_multisequence_decoder_payload_v1" and
            manifest["status"] ==
            "PASS_CAPTURE_ONLY__FRESH_RESULT_HAMMER_REQUIRED",
            "manifest schema/status drift")
    require(manifest["population"] == {
        "hook_calls": 120, "modules": 4, "samples": 30,
        "samples_per_sequence": 10, "sequences": 3,
        "route_counts": {"COMMON_FP32_HASH_ONLY_FALLBACK": 0,
                         "EXACT_BINARY_BITPACK": 90,
                         "EXACT_SCALED_BINARY_BITPACK": 30},
        "per_sequence": {sequence: {
            "hook_calls": 40, "samples": 10,
            "routes": {"COMMON_FP32_HASH_ONLY_FALLBACK": 0,
                       "EXACT_BINARY_BITPACK": 30,
                       "EXACT_SCALED_BINARY_BITPACK": 10}}
                         for sequence in SEQUENCES}},
            "population ledger drift")
    boundary = manifest["claim_boundary"]
    require(boundary["payload_and_density_only"] is True and
            boundary["same_h67_ep35_checkpoint"] is True and
            boundary["checkpoint_exact_load_missing_unexpected_zero"] is True and
            boundary["no_threshold_round_or_coercion"] is True and
            boundary["original_fp32_content_hashed"] is True and
            all(boundary[key] is False for key in
                ("accuracy", "cycles", "speedup", "system_speedup", "rtl",
                 "vcs", "eda", "dc", "formality", "ptpx", "energy",
                 "ppa", "date_headline")), "claim boundary drift")

    selected = manifest["selection"]["selected_sources"]
    require(manifest["selection"]["sequences"] == SEQUENCES and
            manifest["selection"]["algorithm"] ==
            "round(i*(N-1)/9), i=0..9" and len(selected) == 30 and
            contract["selected_sources"] == [
                {key: row[key] for key in
                 ("global_sample_id", "sequence", "sequence_sample_id",
                  "source_population", "source_index", "path", "bytes",
                  "sha256")} for row in selected],
            "selection contract/manifest mismatch")
    for global_id, source in enumerate(selected):
        sequence_index = global_id // 10
        sequence_sample = global_id % 10
        sequence = SEQUENCES[sequence_index]
        require(source["global_sample_id"] == global_id and
                source["sequence"] == sequence and
                source["sequence_sample_id"] == sequence_sample and
                source["source_index"] ==
                selected_indices(source["source_population"])[sequence_sample] and
                source["shape"] == [10, 480, 640] and
                source["dtype"] == "float32" and
                source["bytes"] == 12288128,
                "selected source identity/order drift")
        path = ROOT / source["path"]
        require(str(path.resolve()) == source["resolved_path"],
                "source resolved path drift")
        if verify_sources:
            observed = os.lstat(str(path))
            require(stat.S_ISREG(observed.st_mode) and
                    not stat.S_ISLNK(observed.st_mode) and
                    path.stat().st_size == source["bytes"] and
                    sha256(path) == source["sha256"],
                    "selected source byte identity drift")
            array = np.load(str(path), mmap_mode="r", allow_pickle=False)
            require(list(array.shape) == source["shape"] and
                    array.dtype == np.dtype("float32"), "NPY header drift")

    records = manifest["records"]
    require(len(records) == 120 and
            [row["global_call_index"] for row in records] == list(range(120)),
            "record order/count drift")
    density_counts = {sequence: {module: {"ones": 0, "elements": 0}
                                 for module in range(4)}
                      for sequence in SEQUENCES}
    payload_checks = 0
    for global_call, row in enumerate(records):
        global_sample = global_call // 4
        module = global_call % 4
        source = selected[global_sample]
        sequence = source["sequence"]
        route = ("EXACT_SCALED_BINARY_BITPACK" if module == 1 else
                 "EXACT_BINARY_BITPACK")
        suffix = "theta" if module == 1 else "binary"
        relative = "calls/s{:02d}_d{}.{}.le.bitpack".format(
            global_sample, module, suffix)
        require(row["global_sample_id"] == global_sample and
                row["module_index"] == module and row["name"] == NAMES[module] and
                row["sequence"] == sequence and
                row["sequence_sample_id"] == source["sequence_sample_id"] and
                row["source_path"] == source["path"] and
                row["source_sha256"] == source["sha256"] and
                row["route"] == route and row["relative_path"] == relative and
                row["input_dtype"] == "torch.float32" and
                row["input_shape"] == SHAPES[module] and
                row["input_stride"] == c_stride(SHAPES[module]) and
                row["thresholded"] is False and row["rounded"] is False and
                row["coerced"] is False,
                "record lattice/route/source identity drift")
        elements = product(SHAPES[module])
        path = OUTPUT / safe_member(relative)
        if module == 1:
            stats = row["statistics"]["scaled_binary_audit"]
            raw = row["statistics"]["raw"]
            require(stats["comparison"] ==
                    "BIT_EXACT_X_EQ_0_OR_X_EQ_RUNTIME_SCALAR_THETA" and
                    stats["theta_gate_pass"] is True and
                    stats["other_finite_count"] == 0 and
                    stats["nonfinite_count"] == 0 and
                    stats["raw_payload_saved"] is False and
                    stats["thresholded"] is False and stats["rounded"] is False and
                    raw["route"] == "COMMON_FP32_DENSE_FALLBACK" and
                    raw["coerced_to_binary"] is False and
                    raw["raw_payload_saved"] is False and
                    raw["thresholded"] is False and raw["one_count"] == 0 and
                    raw["nonbinary_finite_count"] == stats["theta_count"] and
                    raw["zero_count"] == stats["zero_count"] and
                    raw["nonfinite_count"] == 0,
                    "D1 scaled-binary/nonexact boundary drift")
            ones = stats["theta_count"]
        else:
            stats = row["statistics"]
            require(stats["exact_binary_count"] == elements and
                    stats["nonbinary_finite_count"] == 0 and
                    stats["nonfinite_count"] == 0,
                    "binary exactness ledger drift")
            ones = stats["one_count"]
        require(stats["elements"] == elements and
                stats["packed_bytes"] == PACKED_BYTES[module] and
                stats["raw_content_bytes"] == elements * 4 and
                stats["zero_count"] == elements - ones and
                stats["packed_sha256"] == sha256(path) and
                stats["raw_content_sha256"] == row["raw_fp32_content_sha256"],
                "payload count/hash ledger drift")
        if verify_payload:
            theta_word = manifest["d1_runtime_threshold_identity"][
                "ieee754_uint32"] if module == 1 else 0x3F800000
            observed = reconstruct_payload(path, elements, theta_word)
            require(path.stat().st_size == PACKED_BYTES[module] and
                    observed["ones"] == ones and
                    observed["zeros"] == stats["zero_count"] and
                    observed["packed_sha256"] == stats["packed_sha256"] and
                    observed["raw_fp32_sha256"] ==
                    stats["raw_content_sha256"],
                    "independent bitpack/raw reconstruction mismatch")
            payload_checks += 1
        density_counts[sequence][module]["ones"] += ones
        density_counts[sequence][module]["elements"] += elements
    return density_counts, payload_checks


def main():
    manifest_path = OUTPUT / "manifest.json"
    require(sha256(manifest_path) == EXPECTED_MANIFEST_SHA,
            "externally frozen manifest root mismatch")
    require(sha256(OUTPUT / "SHA256SUMS.seal.sha256") ==
            EXPECTED_OUTPUT_OUTER_FILE_SHA,
            "externally frozen output root mismatch")
    require(sha256(CONTRACT) == EXPECTED_CONTRACT_SHA and
            sha256(RUNNER) == EXPECTED_RUNNER_SHA,
            "runner/contract root drift")

    manifest = strict_json(manifest_path)
    contract = strict_json(CONTRACT)
    expected_calls = {
        "calls/s{:02d}_d{}.{}.le.bitpack".format(
            sample, module, "theta" if module == 1 else "binary")
        for sample in range(30) for module in range(4)}
    files, directories = tree_inventory(OUTPUT)
    require(files == expected_calls | {"RUN_COMPLETE.txt", "manifest.json",
                                       "SHA256SUMS", "SHA256SUMS.seal.sha256"},
            "canonical file population drift")
    require(directories == {".", "calls"},
            "canonical directory population drift")
    seals = {"output": verify_double_seal(OUTPUT),
             "attempt": verify_double_seal(ATTEMPT),
             # Predecessor review members remain sealed even if a later local
             # py_compile left an unsealed __pycache__ beside them.
             "static_review": verify_double_seal(STATIC_REVIEW, False),
             "m686_predecessor": verify_double_seal(M686),
             "m692_predecessor_review": verify_double_seal(M692, False)}
    require((OUTPUT / "RUN_COMPLETE.txt").read_text(encoding="utf-8") ==
            "PASS_M699_CAPTURE_ONLY__FRESH_RESULT_HAMMER_REQUIRED\n",
            "completion sentinel drift")

    # Attempt and embedded runtime receipts.
    attempt_files, attempt_dirs = tree_inventory(ATTEMPT)
    require(attempt_files == {"ATTEMPT_CONSUMED.txt", "POSTCAPTURE_PASS.txt",
                              "identity.sha256", "SHA256SUMS",
                              "SHA256SUMS.seal.sha256"} and
            attempt_dirs == {"."}, "attempt population drift")
    attempt = parse_key_values(ATTEMPT / "ATTEMPT_CONSUMED.txt")
    require(attempt == {
        "status": "CONSUMED_IMMEDIATELY_BEFORE_M699_ONE_SHOT",
        "runner_sha256": EXPECTED_RUNNER_SHA,
        "contract_sha256": EXPECTED_CONTRACT_SHA,
        "review_sha256": EXPECTED_STATIC_REVIEW_SHA,
        "review_outer_seal_file_sha256": EXPECTED_STATIC_OUTER_FILE_SHA,
        "gpu_free_mib": attempt["gpu_free_mib"],
        "claim_boundary":
        "PAYLOAD_DENSITY_ONLY_NO_ACCURACY_CYCLES_SPEEDUP_SYSTEM_RTL_EDA_ENERGY_PPA_OR_HEADLINE"} and
            attempt["gpu_free_mib"].isdigit() and
            int(attempt["gpu_free_mib"]) >= 20000,
            "attempt receipt drift")
    require((ATTEMPT / "POSTCAPTURE_PASS.txt").read_text(encoding="utf-8") ==
            "PASS_CAPTURE_AND_RUNNER_REHASH\n", "postcapture receipt drift")
    for raw in (ATTEMPT / "identity.sha256").read_text(
            encoding="utf-8").splitlines():
        expected, name = raw.split(None, 1)
        path = Path(name.strip())
        require(path.is_file() and not path.is_symlink() and
                sha256(path) == expected, "attempt identity rehash drift")
    static_review = strict_json(STATIC_REVIEW / "review.json")
    require(sha256(STATIC_REVIEW / "review.json") == EXPECTED_STATIC_REVIEW_SHA and
            sha256(STATIC_REVIEW / "SHA256SUMS.seal.sha256") ==
            EXPECTED_STATIC_OUTER_FILE_SHA and
            static_review["status"] ==
            "GO_M699_GPU_ONE_SHOT__CAPTURE_ONLY__P0_0_P1_0" and
            static_review["execution_authorized"] is True and
            static_review["severity"]["p0"] == 0 and
            static_review["severity"]["p1"] == 0,
            "static review chain drift")
    require("m699_success=1\ntrap - EXIT\necho \"PASS M699 one-shot capture" in
            RUNNER.read_text(encoding="utf-8"),
            "runner exit-zero completion path drift")

    runtime = manifest["identity"]["runtime"]
    expected_env = dict(contract["runtime"]["expected_environment"])
    expected_env["M699_EXPECTED_CONTRACT_SHA256"] = EXPECTED_CONTRACT_SHA
    require(runtime["argv"] == contract["runtime"]["exact_python_argv"] and
            runtime["environment"] == expected_env and
            runtime["python"] == contract["runtime"]["python"] and
            runtime["hostname"] == contract["runtime"]["host_gpu"]["hostname"] and
            runtime["gpu"] == contract["runtime"]["host_gpu"]["gpu"] and
            manifest["deterministic_execution"] ==
            contract["runtime"]["deterministic_execution"] and
            manifest["cuda_synchronization"] == {
                "before_capture": 1, "per_sample_post_forward": 30,
                "final_pre_manifest": 1}, "embedded runtime receipt drift")

    # Exact-load and external identity chains, including all 30 source files.
    load = manifest["identity"]["checkpoint_load_audit"]
    require(load["missing_count"] == 0 and load["unexpected_count"] == 0 and
            load["overlay_missing_count"] == 0 and
            load["overlay_unexpected_count"] == 0,
            "checkpoint exact-load receipt drift")
    rehashed_external = {}
    for container in (manifest["identity"]["core_inputs"],
                      manifest["identity"]["frozen_m511_inputs"]):
        for key, entry in container.items():
            path = Path(entry["path"])
            if str(path) not in rehashed_external:
                require(path.is_file() and not path.is_symlink() and
                        sha256(path) == entry["sha256"],
                        "external identity drift: " + key)
                rehashed_external[str(path)] = entry["sha256"]
            else:
                require(rehashed_external[str(path)] == entry["sha256"],
                        "duplicate external identity disagreement")
    docs_path = HW / "docs/359_DATE终局冻结_20260813.md"
    require(rehashed_external[str(docs_path)] == EXPECTED_DOCS359_SHA,
            "docs359 identity drift")

    # Module geometry and weight identities are independently tied to the
    # already admitted, double-sealed M686 same-checkpoint payload.
    require(sha256(M686 / "manifest.json") == EXPECTED_M686_MANIFEST_SHA and
            sha256(M686 / "SHA256SUMS.seal.sha256") ==
            EXPECTED_M686_OUTER_FILE_SHA,
            "M686 predecessor external roots drift")
    m686 = strict_json(M686 / "manifest.json")
    m692 = strict_json(M692 / "review.json")
    require(m692["status"] == "GO_M672_STATIC_ADAPTER_INPUT_ONLY" and
            m692["go"] is True and
            manifest["module_identities"] == m686["module_identities"],
            "module/weight predecessor identity drift")
    m511 = strict_json(Path(manifest["identity"]["core_inputs"][
        "m511_contract"]["path"]))
    for module in m511["modules"]:
        index = module["module_index"]
        identity = manifest["module_identities"][module["name"]]
        require(identity["operator"] == module["operator"] and
                identity["in_channels"] == module["in_channels"] and
                identity["out_channels"] == module["out_channels"] and
                identity["kernel_size"] == module["kernel_size"] and
                identity["stride"] == module["stride"] and
                identity["padding"] == module["padding"] and
                identity["output_padding"] == module["output_padding"] and
                identity["dilation"] == module["dilation"] and
                identity["groups"] == module["groups"] and
                identity["weight"]["shape"] == module["weight_shape"] and
                identity["weight"]["content_bytes"] ==
                product(module["weight_shape"]) * 4 and
                module["input_shape"] == SHAPES[index],
                "module geometry/weight identity drift")

    theta = manifest["d1_runtime_threshold_identity"]
    theta_bytes = struct.pack("<f", theta["value"])
    require(theta["content_bytes"] == 4 and
            theta["ieee754_le_hex"] == theta_bytes.hex() and
            theta["ieee754_uint32"] == struct.unpack("<I", theta_bytes)[0] and
            theta["content_sha256"] == hashlib.sha256(theta_bytes).hexdigest() and
            theta["ieee754_uint32"] not in (0, 0x3F800000) and
            theta["threshold_mode"] == "official_atlif" and
            theta["source_semantics"] == "OfficialATLIFSurrogate returns out * thre",
            "D1 runtime theta identity/nonexact boundary drift")

    density_counts, payload_checks = validate_manifest_semantics(
        manifest, contract, verify_payload=True, verify_sources=True)
    density = {}
    module_spread = {}
    for sequence in SEQUENCES:
        density[sequence] = {}
        total_ones = 0
        total_elements = 0
        for module in range(4):
            counts = density_counts[sequence][module]
            value = counts["ones"] / counts["elements"]
            density[sequence][str(module)] = {
                "ones": counts["ones"], "elements": counts["elements"],
                "density": value}
            total_ones += counts["ones"]
            total_elements += counts["elements"]
        density[sequence]["all_modules_weighted"] = {
            "ones": total_ones, "elements": total_elements,
            "density": total_ones / total_elements}
    for module in range(4):
        values = [density[sequence][str(module)]["density"]
                  for sequence in SEQUENCES]
        module_spread[str(module)] = {
            "min_density": min(values), "max_density": max(values),
            "absolute_spread": max(values) - min(values),
            "relative_to_min": (max(values) - min(values)) / min(values)}
    overall_values = [density[sequence]["all_modules_weighted"]["density"]
                      for sequence in SEQUENCES]
    max_module_absolute_spread = max(
        entry["absolute_spread"] for entry in module_spread.values())

    # Fresh private attacks: member mutation/deletion, semantic reordering, and
    # D1-to-binary route theft.  Resealed semantic attacks must still fail the
    # frozen roots and the independent lattice validator.
    attacks = {}
    with tempfile.TemporaryDirectory(prefix="m705_private_") as temp:
        private = Path(temp) / "payload"
        shutil.copytree(str(OUTPUT), str(private))
        target = private / "calls/s00_d0.binary.le.bitpack"
        with target.open("r+b") as handle:
            first = handle.read(1)
            handle.seek(0)
            handle.write(bytes([first[0] ^ 1]))
        try:
            verify_double_seal(private)
            attacks["member_mutation_rejected"] = False
        except RuntimeError:
            attacks["member_mutation_rejected"] = True
        shutil.copy2(str(OUTPUT / "calls/s00_d0.binary.le.bitpack"), str(target))
        (private / "calls/s00_d1.theta.le.bitpack").unlink()
        try:
            verify_double_seal(private)
            attacks["member_deletion_rejected"] = False
        except (RuntimeError, FileNotFoundError):
            attacks["member_deletion_rejected"] = True

        shutil.rmtree(str(private))
        shutil.copytree(str(OUTPUT), str(private))
        altered = strict_json(private / "manifest.json")
        altered["records"][0], altered["records"][1] = (
            altered["records"][1], altered["records"][0])
        (private / "manifest.json").write_text(
            json.dumps(altered, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        reseal(private)
        verify_double_seal(private)
        try:
            validate_manifest_semantics(altered, contract)
            attacks["record_reorder_semantically_rejected"] = False
        except RuntimeError:
            attacks["record_reorder_semantically_rejected"] = True
        attacks["record_reorder_external_roots_rejected"] = (
            sha256(private / "manifest.json") != EXPECTED_MANIFEST_SHA and
            sha256(private / "SHA256SUMS.seal.sha256") !=
            EXPECTED_OUTPUT_OUTER_FILE_SHA)

        shutil.rmtree(str(private))
        shutil.copytree(str(OUTPUT), str(private))
        altered = strict_json(private / "manifest.json")
        altered["records"][1]["route"] = "EXACT_BINARY_BITPACK"
        (private / "manifest.json").write_text(
            json.dumps(altered, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        reseal(private)
        verify_double_seal(private)
        try:
            validate_manifest_semantics(altered, contract)
            attacks["d1_route_theft_semantically_rejected"] = False
        except RuntimeError:
            attacks["d1_route_theft_semantically_rejected"] = True
        attacks["d1_route_theft_external_roots_rejected"] = (
            sha256(private / "manifest.json") != EXPECTED_MANIFEST_SHA and
            sha256(private / "SHA256SUMS.seal.sha256") !=
            EXPECTED_OUTPUT_OUTER_FILE_SHA)
    require(all(attacks.values()), "tamper/reorder/route attack escaped")

    result = {
        "schema": "m705_m699_multisequence_decoder_payload_independent_hammer_v1",
        "status": "PASS_PAYLOAD_DENSITY_AND_OBSERVED_S3_STABILITY_ONLY",
        "frozen_roots": {"manifest_sha256": EXPECTED_MANIFEST_SHA,
                         "outer_seal_file_sha256":
                         EXPECTED_OUTPUT_OUTER_FILE_SHA},
        "seals": seals,
        "canonical_population": {"files": len(files),
                                  "directories": sorted(directories),
                                  "symlinks": 0},
        "attempt_and_runtime": {"attempt_consumed": True,
                                "postcapture_runner_rehash": True,
                                "embedded_exact_argv_env_gpu_receipt": True,
                                "exit_zero_completion_path": True},
        "checkpoint_load": {"missing_count": 0, "unexpected_count": 0,
                            "overlay_missing_count": 0,
                            "overlay_unexpected_count": 0},
        "identity": {"source_files_rehashed": 30,
                     "external_identity_files_rehashed":
                     len(rehashed_external),
                     "module_geometry_records": 4,
                     "weight_identities_equal_admitted_m686": True,
                     "docs359_sha256": EXPECTED_DOCS359_SHA},
        "payload": {"samples": 30, "hook_calls": 120,
                    "bitpacks_rehashed_popcounted_and_raw_reconstructed":
                    payload_checks,
                    "routes": {"EXACT_BINARY_BITPACK": 90,
                               "EXACT_SCALED_BINARY_BITPACK": 30,
                               "COMMON_FP32_HASH_ONLY_FALLBACK": 0},
                    "packed_bytes_total": sum(PACKED_BYTES.values()) * 30,
                    "all_tail_padding_zero": True,
                    "no_threshold_round_or_coercion": True},
        "d1_boundary": {"theta_value": theta["value"],
                        "theta_ieee754_uint32": theta["ieee754_uint32"],
                        "theta_is_not_exact_one": True,
                        "exact_scaled_binary_records": 30,
                        "exact_binary_records": 0,
                        "raw_dense_route_records": 30,
                        "folded_weight_or_decoder_equivalence_admitted": False},
        "density": density,
        "cross_sequence": {
            "definition": "for each module, max minus min aggregate bit density across the three selected S10 cohorts",
            "module_spread": module_spread,
            "max_module_absolute_spread": max_module_absolute_spread,
            "max_module_absolute_spread_percentage_points":
            max_module_absolute_spread * 100.0,
            "weighted_all_module_density_min": min(overall_values),
            "weighted_all_module_density_max": max(overall_values),
            "weighted_all_module_absolute_spread":
            max(overall_values) - min(overall_values),
            "observed_selected_s3_stability_only": True,
            "population_generalization": False},
        "attacks": attacks,
        "execution_boundary": {"gpu": False, "model": False, "rtl": False,
                               "eda": False, "cycles": False,
                               "speedup": False, "accuracy": False,
                               "ours": False, "system": False},
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
