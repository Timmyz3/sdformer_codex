#!/usr/bin/env python3
"""Independent static-only M511 r3 identity/topology/control-flow check."""

from __future__ import print_function

import functools
import hashlib
import json
import operator
from pathlib import Path


REPO = Path(__file__).resolve().parents[3]
PRODUCER = REPO / ("neuron_experiments/H9_bipolar_self_attention/entrypoints/"
                   "capture_m511_h67_convtranspose_binary_inputs.py")
CONTRACT = REPO / ("hw_autoresearch_nts07/contracts/"
                   "m511_h67_ep35_convtranspose_binary_input_capture_contract_r1_20260827.json")
PRODUCER_SHA = "7a4f6f36f7336576d1c1959ce050ea43d83d0fbf9154a80131300ce8b7c6d4cd"
CONTRACT_SHA = "2b0d7bc98ca6d4275342ccb5446c4ce1943972511240647878bf389b378358f0"


def require(value, message):
    if not value:
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


def product(items):
    return functools.reduce(operator.mul, items, 1)


def main():
    require(sha256(PRODUCER) == PRODUCER_SHA, "producer drift")
    require(sha256(CONTRACT) == CONTRACT_SHA, "contract drift")
    contract = strict_json(CONTRACT)
    require(contract["inputs"]["capture_script"]["sha256"] == PRODUCER_SHA,
            "producer pin drift")
    require(set(name for name in contract["inputs"] if name.startswith("m512")) ==
            {"m512_review", "m512_review_manifest", "m512_review_seal"},
            "M512 provenance population drift")

    checked = {}
    for name, entry in sorted(contract["inputs"].items()):
        path = REPO / entry["path"]
        observed = sha256(path)
        require(observed == entry["sha256"], "input drift: " + name)
        checked[name] = observed
    m512 = strict_json(REPO / contract["inputs"]["m512_review"]["path"])
    require(m512["status"] ==
            "NO_GO__KILL_PHASE_BALANCED_MULTI_SOURCE_EPD_SCHEDULER_BEFORE_RTL",
            "M512 decision drift")

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
    require([row["sample_id"] for row in contract["samples"]] == list(range(10)),
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
        elements = product(input_shape)
        per_call.append({"module_index": index, "elements": elements,
                         "packed_bytes": (elements + 7) // 8,
                         "tail_used_bits": elements % 8 or 8})
    elements_s10 = 10 * sum(row["elements"] for row in per_call)
    bytes_s10 = 10 * sum(row["packed_bytes"] for row in per_call)
    require(elements_s10 == 696240000 and bytes_s10 == 87030000,
            "S10 payload recomputation drift")
    require(contract["expected_population"]["input_elements"] == elements_s10 and
            contract["expected_population"]["packed_bytes"] == bytes_s10,
            "contract payload drift")

    m510 = strict_json(REPO / contract["inputs"]["m510_result"]["path"])
    for row, layer in zip(contract["modules"], m510["layers"]):
        require(row["module_index"] == layer["decoder"] and
                row["in_channels"] == layer["channels_in"] and
                row["out_channels"] == layer["channels_out"] and
                row["input_shape"] == layer["input_shape"] and
                row["output_shape"] == layer["output_shape"],
                "M510 topology cross-check drift")

    producer = PRODUCER.read_text(encoding="utf-8")
    remove_pos = producer.index("while handles:")
    manifest_pos = producer.index("manifest = {", remove_pos)
    publish_pos = producer.index("os.replace(staging, output)", manifest_pos)
    published_pos = producer.index("published = True", publish_pos)
    verify_output_pos = producer.index("verify_seal(output)", published_pos)
    except_pos = producer.index("    except BaseException as error:", verify_output_pos)
    finally_pos = producer.index("    finally:", except_pos)
    require(remove_pos < manifest_pos < publish_pos < published_pos <
            verify_output_pos < except_pos < finally_pos,
            "success transaction ordering drift")
    require("handles.pop().remove()" in producer[remove_pos:manifest_pos],
            "prepublish hook clear missing")
    require("except BaseException:\n                # Any path reaching this fallback" in
            producer[finally_pos:], "best-effort failure finalizer drift")
    require("actual_names == sealed_names" in producer,
            "exact member seal check missing")
    require("sha256(contract_path) == contract_start" in producer and
            "rehash_sample_sources(" in producer,
            "start/end identity check missing")
    require("uuid.uuid4().hex" in producer[manifest_pos:publish_pos],
            "prepublish collision-resistant quarantine marker missing")
    recovery = producer[except_pos:finally_pos]
    require("os.replace(output, quarantine)" in recovery,
            "postpublish quarantine missing")
    require("uuid.uuid4" not in recovery and
            "quarantine.exists" not in recovery.split(
                "os.replace(output, quarantine)", 1)[0],
            "fallible quarantine preparation remains after publication")

    result = {
        "schema": "m511_static_check_r3_v1",
        "status": "PASS_STATIC_IDENTITY_TOPOLOGY_AND_TRANSACTION_RECOMPUTATION",
        "execution_boundary": {
            "production_capture_executed": False,
            "checkpoint_loaded": False,
            "model_constructed": False,
            "cuda_touched": False,
        },
        "identity": {
            "producer_sha256": PRODUCER_SHA,
            "contract_sha256": CONTRACT_SHA,
            "checked_input_count": len(checked),
            "checked_inputs": checked,
            "m512_kill_provenance": True,
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
        "transaction": {
            "hooks_cleared_before_manifest_and_publish": True,
            "success_finally_has_zero_handles_by_construction": True,
            "contract_and_all_pinned_inputs_rehashed": True,
            "raw_sequence_event_mask_flow_rehashed": True,
            "exact_actual_vs_sealed_member_set": True,
            "canonical_verified_after_atomic_publish": True,
            "postpublish_exception_quarantines_canonical": True,
            "quarantine_target_precomputed_before_publish": True,
            "postpublish_first_recovery_operation_is_atomic_rename": True,
        },
    }
    output = Path(__file__).with_name("m511_static_check_r3.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS_STATIC_CHECK", output, sha256(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
