#!/usr/bin/env python3
"""Static-only adversarial check for the re-locked M511 capture.

No torch import, checkpoint load, model construction, CUDA call, or production
capture is permitted here.
"""

from __future__ import print_function

import functools
import hashlib
import json
import operator
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PRODUCER = REPO / ("neuron_experiments/H9_bipolar_self_attention/"
                   "entrypoints/capture_m511_h67_convtranspose_binary_inputs.py")
CONTRACT = REPO / ("hw_autoresearch_nts07/contracts/"
                   "m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json")
EXPECTED_PRODUCER_SHA = "73e26e731956fd38af949ecff8467479f5685832c9d953ca67237445e045d664"
EXPECTED_CONTRACT_SHA = "69f948f050f20dd54f314690a1bf2009316f9815d257298240bb44b651eca8dc"


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

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=pairs, parse_constant=reject)


def prod(items):
    return functools.reduce(operator.mul, items, 1)


def main():
    require(sha256(PRODUCER) == EXPECTED_PRODUCER_SHA, "producer drift")
    require(sha256(CONTRACT) == EXPECTED_CONTRACT_SHA, "contract drift")
    contract = strict_json(CONTRACT)
    require(contract["inputs"]["capture_script"]["sha256"] ==
            EXPECTED_PRODUCER_SHA, "contract/producer pin mismatch")

    checked_inputs = {}
    for name, entry in sorted(contract["inputs"].items()):
        path = REPO / entry["path"]
        observed = sha256(path)
        require(observed == entry["sha256"], "input drift: " + name)
        checked_inputs[name] = observed

    expected = [
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
    require([x["sample_id"] for x in contract["samples"]] == list(range(10)),
            "sample order drift")
    for row, item in zip(contract["modules"], expected):
        index, name, cin, cout, input_shape, output_shape = item
        require((row["module_index"], row["name"], row["in_channels"],
                 row["out_channels"], row["input_shape"], row["output_shape"]) ==
                (index, name, cin, cout, input_shape, output_shape),
                "module topology drift")
        require(row["kernel_size"] == [3, 3] and row["stride"] == [2, 2] and
                row["padding"] == [1, 1] and
                row["output_padding"] == [1, 1] and
                row["dilation"] == [1, 1] and row["groups"] == 1 and
                row["weight_shape"] == [cin, cout, 3, 3],
                "module property drift")
        elements = prod(input_shape)
        per_call.append({"module_index": index, "elements": elements,
                         "packed_bytes": (elements + 7) // 8,
                         "tail_used_bits": elements % 8 or 8})
    s10_elements = 10 * sum(x["elements"] for x in per_call)
    s10_bytes = 10 * sum(x["packed_bytes"] for x in per_call)
    require(s10_elements == 696240000 and s10_bytes == 87030000,
            "independent payload total drift")
    require(contract["expected_population"]["input_elements"] == s10_elements and
            contract["expected_population"]["packed_bytes"] == s10_bytes,
            "contract payload mismatch")

    m510 = strict_json(REPO / contract["inputs"]["m510_result"]["path"])
    require(len(m510["layers"]) == 4, "M510 layer count drift")
    for row, layer in zip(contract["modules"], m510["layers"]):
        require(row["module_index"] == layer["decoder"] and
                row["in_channels"] == layer["channels_in"] and
                row["out_channels"] == layer["channels_out"] and
                row["input_shape"] == layer["input_shape"] and
                row["output_shape"] == layer["output_shape"],
                "M510 cross-check drift")

    producer = PRODUCER.read_text(encoding="utf-8")
    spiking = (REPO / contract["inputs"]["spiking_modules"]["path"]).read_text(
        encoding="utf-8")
    snn = (REPO / contract["inputs"]["snn_models"]["path"]).read_text(
        encoding="utf-8")
    stswin = (REPO / contract["inputs"]["spiking_stswinnet"]["path"]).read_text(
        encoding="utf-8")
    require("x = self.sn(x)\n        x = self.deconv(x)" in spiking,
            "sn-before-deconv proof missing")
    require("self.UpsampleLayer = self.transpose_type" in snn and
            "transpose_type = MS_SpikingTransposeDecoderLayer" in stswin,
            "MS transpose selection proof missing")

    publish_pos = producer.index("os.replace(staging, output)")
    finally_pos = producer.index("    finally:", publish_pos)
    finally_body = producer[finally_pos:]
    controls = {
        "exact_actual_vs_sealed_members": "actual_names == sealed_names" in producer,
        "contract_start_end": "sha256(contract_path) == contract_start" in producer,
        "raw_source_start_end": "rehash_sample_sources(" in producer,
        "canonical_quarantine_in_except": "os.replace(output, quarantine)" in producer,
        "pid_only_quarantine_name":
            '".quarantine.failed.{}".format(os.getpid())' in producer,
        "handle_remove_in_uncaught_finally_after_publish":
            "handle.remove()" in finally_body,
        "m512_evidence_pinned": any(name.startswith("m512")
                                      for name in contract["inputs"]),
        "complete_runtime_convtranspose_set_asserted":
            "runtime_convtranspose" in producer or
            "all_convtranspose" in producer,
        "publish_precedes_finally": publish_pos < finally_pos,
    }

    result = {
        "schema": "m511_static_check_r2_v1",
        "status": "PASS_STATIC_RECOMPUTATION__CONTROL_FINDINGS_REPORTED",
        "execution_boundary": {
            "production_capture_executed": False,
            "checkpoint_loaded": False,
            "model_constructed": False,
            "cuda_touched": False,
        },
        "identity": {
            "producer_sha256": EXPECTED_PRODUCER_SHA,
            "contract_sha256": EXPECTED_CONTRACT_SHA,
            "checked_inputs": checked_inputs,
        },
        "topology_and_population": {
            "ms_sn_before_deconv": True,
            "k3_s2_p1_output_padding1": True,
            "m510_crosscheck": True,
            "samples": 10,
            "records": 40,
            "input_elements": s10_elements,
            "packed_bytes": s10_bytes,
            "packed_mib": s10_bytes / float(1 << 20),
            "per_call": per_call,
        },
        "controls": controls,
    }
    out = Path(__file__).with_name("m511_static_check_r2.json")
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print("PASS_STATIC_CHECK", out, sha256(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
