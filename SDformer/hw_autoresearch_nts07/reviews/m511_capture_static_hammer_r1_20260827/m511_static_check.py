#!/usr/bin/env python3
"""Independent static-only M511 producer/contract audit.

This checker deliberately does not import torch, load a checkpoint, construct a
model, touch CUDA, or invoke the production capture.  It verifies immutable file
identities, recomputes the S10 payload population, and cross-checks the contract
against the sealed M510 result and the pinned topology sources.
"""

from __future__ import print_function

import functools
import hashlib
import json
import operator
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PRODUCER = REPO / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m511_h67_convtranspose_binary_inputs.py"
)
CONTRACT = REPO / (
    "hw_autoresearch_nts07/contracts/"
    "m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json"
)
EXPECTED_PRODUCER_SHA = (
    "201a40137f4a1d83f137eeb48b4b70fba9a72391ee603452df221c85f2d2cee8"
)
EXPECTED_CONTRACT_SHA = (
    "b3eb127df0ce72eca8891dd602128f9292ad287ad24f7805203c99c8934fb69d"
)


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
        value = {}
        for key, item in items:
            require(key not in value, "duplicate JSON key: " + key)
            value[key] = item
        return value

    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject,
    )


def prod(values):
    return functools.reduce(operator.mul, values, 1)


def main():
    require(sha256(PRODUCER) == EXPECTED_PRODUCER_SHA, "producer identity drift")
    require(sha256(CONTRACT) == EXPECTED_CONTRACT_SHA, "contract identity drift")
    contract = strict_json(CONTRACT)
    require(len(contract["samples"]) == 10, "sample population drift")
    require([row["sample_id"] for row in contract["samples"]] == list(range(10)),
            "sample ids are not exact 0..9")
    require(len(contract["modules"]) == 4, "module population drift")
    require([row["module_index"] for row in contract["modules"]] == list(range(4)),
            "module indices are not exact 0..3")

    input_checks = {}
    for name, entry in sorted(contract["inputs"].items()):
        path = REPO / entry["path"]
        observed = sha256(path)
        require(observed == entry["sha256"], "input drift: " + name)
        input_checks[name] = {
            "path": entry["path"],
            "sha256": observed,
            "size_bytes": path.stat().st_size,
        }

    expected_modules = [
        (0, "sttmultires_unet.decoders.0.deconv.0", 1536, 384,
         [10, 1, 1536, 15, 20], [10, 1, 384, 30, 40]),
        (1, "sttmultires_unet.decoders.1.deconv.0", 770, 192,
         [10, 1, 770, 30, 40], [10, 1, 192, 60, 80]),
        (2, "sttmultires_unet.decoders.2.deconv.0", 386, 96,
         [10, 1, 386, 60, 80], [10, 1, 96, 120, 160]),
        (3, "sttmultires_unet.decoders.3.deconv.0", 194, 96,
         [10, 1, 194, 120, 160], [10, 1, 96, 240, 320]),
    ]
    per_call = []
    for module, expected in zip(contract["modules"], expected_modules):
        index, name, cin, cout, input_shape, output_shape = expected
        require(module["module_index"] == index and module["name"] == name,
                "module name/index drift")
        require(module["operator"] == "ConvTranspose2d", "operator drift")
        require(module["in_channels"] == cin and module["out_channels"] == cout,
                "channel drift")
        require(module["kernel_size"] == [3, 3] and
                module["stride"] == [2, 2] and
                module["padding"] == [1, 1] and
                module["output_padding"] == [1, 1] and
                module["dilation"] == [1, 1] and module["groups"] == 1,
                "K3/S2/P1/OP1 property drift")
        require(module["weight_shape"] == [cin, cout, 3, 3],
                "ConvTranspose weight layout drift")
        require(module["input_shape"] == input_shape and
                module["output_shape"] == output_shape, "shape drift")
        elements = prod(input_shape)
        per_call.append({
            "module_index": index,
            "input_elements": elements,
            "packed_bytes": (elements + 7) // 8,
            "tail_used_bits": elements % 8 or 8,
        })

    elements_s10 = 10 * sum(row["input_elements"] for row in per_call)
    bytes_s10 = 10 * sum(row["packed_bytes"] for row in per_call)
    require(elements_s10 == 696240000, "independent S10 element total drift")
    require(bytes_s10 == 87030000, "independent S10 byte total drift")
    population = contract["expected_population"]
    require(population["input_elements"] == elements_s10 and
            population["packed_bytes"] == bytes_s10 and
            population["records"] == 40, "contract population mismatch")

    m510 = strict_json(REPO / contract["inputs"]["m510_result"]["path"])
    require(m510["status"] ==
            "PASS_CONFIRMED_OMITTED_CONVTRANSPOSE__TRACE_REQUIRED_BEFORE_RTL",
            "M510 status drift")
    require(len(m510["layers"]) == 4, "M510 layer population drift")
    for module, layer in zip(contract["modules"], m510["layers"]):
        require(module["module_index"] == layer["decoder"] and
                module["in_channels"] == layer["channels_in"] and
                module["out_channels"] == layer["channels_out"] and
                module["input_shape"] == layer["input_shape"] and
                module["output_shape"] == layer["output_shape"],
                "contract/M510 layer identity mismatch")

    spiking = (REPO / contract["inputs"]["spiking_modules"]["path"]).read_text(
        encoding="utf-8")
    snn = (REPO / contract["inputs"]["snn_models"]["path"]).read_text(
        encoding="utf-8")
    stswin = (REPO / contract["inputs"]["spiking_stswinnet"]["path"]).read_text(
        encoding="utf-8")
    producer = PRODUCER.read_text(encoding="utf-8")
    require("class MS_SpikingTransposeDecoderLayer" in spiking and
            "x = self.sn(x)\n        x = self.deconv(x)" in spiking,
            "MS sn-before-deconv source proof missing")
    require("layer.ConvTranspose2d(" in spiking and "stride=2" in spiking and
            "output_padding=1" in spiking, "transpose construction proof missing")
    require("if use_upsample_conv:" in snn and
            "self.UpsampleLayer = self.transpose_type" in snn,
            "transpose selection proof missing")
    require("transpose_type = MS_SpikingTransposeDecoderLayer" in stswin and
            "class MS_SpikingformerFlowNet_en4" in stswin,
            "H67 MS/en4 topology proof missing")

    # Adversarial control-flow findings.  These are intentionally static: the
    # review must not execute the production capture.
    require("os.replace(staging, output)" in producer and
            "verify_seal(output)" in producer and
            "postpublication_failed" in producer,
            "post-publication control-flow marker changed")
    require("sha256(contract_path)" in producer and
            '"contract"' not in contract["inputs"],
            "contract mutation audit premise changed")
    require("for line in (directory / \"SHA256SUMS\")" in producer and
            "rglob" not in producer.split("def verify_seal", 1)[1].split(
                "def main", 1)[0],
            "non-exhaustive seal-verifier premise changed")

    result = {
        "schema": "m511_capture_static_check_r1",
        "status": "PASS_STATIC_RECOMPUTATION_WITH_ADVERSARIAL_FINDINGS",
        "execution_boundary": {
            "production_capture_executed": False,
            "checkpoint_loaded": False,
            "model_constructed": False,
            "cuda_touched": False,
        },
        "identity": {
            "producer_sha256": EXPECTED_PRODUCER_SHA,
            "contract_sha256": EXPECTED_CONTRACT_SHA,
            "inputs": input_checks,
        },
        "topology": {
            "ms_sn_before_deconv": True,
            "module_names_channels_shapes_match": True,
            "k3_s2_p1_op1_group1_bias_null_contract": True,
            "convtranspose_weight_layout_cin_cout_k3_k3": True,
            "m510_crosscheck": True,
        },
        "population": {
            "samples": 10,
            "modules": 4,
            "records": 40,
            "input_elements": elements_s10,
            "packed_bytes": bytes_s10,
            "packed_mib": bytes_s10 / float(1 << 20),
            "per_call": per_call,
        },
        "findings": {
            "p0": [
                "Canonical PASS can survive a post-publication failure.",
                "The governing contract is not start/end identity checked.",
            ],
            "p1": [
                "verify_seal accepts files absent from SHA256SUMS.",
                "Sample names are checked, but raw data content is not bound to M51.",
                "Runtime does not assert the complete ConvTranspose2d module set equals the four targets.",
            ],
        },
    }
    out = Path(__file__).with_name("m511_static_check_r1.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print("PASS_STATIC_CHECK", out, sha256(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
