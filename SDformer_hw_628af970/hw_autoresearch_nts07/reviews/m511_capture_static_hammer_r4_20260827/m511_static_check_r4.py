#!/usr/bin/env python3
"""Narrow static-only M511 r4 closure check; never runs production capture."""

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
PRODUCER_SHA = "e16a454d532acd15d96527cfddf43ebf9f95338a34ce9aeedbb10032cb26230a"
CONTRACT_SHA = "e556743dd18804a7aba5be5b18f33823bbcd5e5be85d7715edcc43a4c314c28e"


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
            "contract/producer pin mismatch")
    checked = {}
    for name, entry in sorted(contract["inputs"].items()):
        observed = sha256(REPO / entry["path"])
        require(observed == entry["sha256"], "input drift: " + name)
        checked[name] = observed

    expected_names = [
        "sttmultires_unet.decoders.0.deconv.0",
        "sttmultires_unet.decoders.1.deconv.0",
        "sttmultires_unet.decoders.2.deconv.0",
        "sttmultires_unet.decoders.3.deconv.0",
    ]
    require([row["name"] for row in contract["modules"]] == expected_names,
            "contract target order drift")
    producer = PRODUCER.read_text(encoding="utf-8")
    exact_set_snippet = (
        "runtime_convtranspose_names = [\n"
        "            name for name, module in model.named_modules()\n"
        "            if isinstance(module, torch.nn.ConvTranspose2d)\n"
        "        ]\n"
        "        require(runtime_convtranspose_names == [\n"
        "            item[\"name\"] for item in contract[\"modules\"]\n"
        "        ], \"M511 complete runtime ConvTranspose2d module set drift\")"
    )
    require(exact_set_snippet in producer,
            "complete ordered runtime ConvTranspose set assertion missing")
    set_pos = producer.index("runtime_convtranspose_names = [")
    hook_register_pos = producer.index("register_forward_hook", set_pos)
    capture_pos = producer.index("torch.cuda.synchronize(device)", hook_register_pos)
    require(set_pos < hook_register_pos < capture_pos,
            "complete set gate does not precede hook/capture")

    per_call = []
    for row in contract["modules"]:
        elements = product(row["input_shape"])
        per_call.append({"module_index": row["module_index"],
                         "elements": elements,
                         "packed_bytes": (elements + 7) // 8,
                         "tail_used_bits": elements % 8 or 8})
    elements_s10 = 10 * sum(row["elements"] for row in per_call)
    bytes_s10 = 10 * sum(row["packed_bytes"] for row in per_call)
    require(elements_s10 == 696240000 and bytes_s10 == 87030000,
            "payload total drift")
    require(contract["expected_population"]["input_elements"] == elements_s10 and
            contract["expected_population"]["packed_bytes"] == bytes_s10,
            "contract population drift")

    remove_pos = producer.index("while handles:")
    manifest_pos = producer.index("manifest = {", remove_pos)
    quarantine_pos = producer.index("quarantine = output.with_name", manifest_pos)
    publish_pos = producer.index("os.replace(staging, output)", quarantine_pos)
    except_pos = producer.index("    except BaseException as error:", publish_pos)
    finally_pos = producer.index("    finally:", except_pos)
    require(remove_pos < manifest_pos < quarantine_pos < publish_pos <
            except_pos < finally_pos, "r3 transaction order drift")
    require("handles.pop().remove()" in producer[remove_pos:manifest_pos],
            "prepublish hook clear drift")
    recovery = producer[except_pos:finally_pos]
    require("os.replace(output, quarantine)" in recovery and
            "uuid.uuid4" not in recovery and
            "quarantine.exists" not in recovery.split(
                "os.replace(output, quarantine)", 1)[0],
            "postpublish recovery drift")
    require("actual_names == sealed_names" in producer and
            "sha256(contract_path) == contract_start" in producer and
            "rehash_sample_sources(" in producer,
            "identity/seal closure drift")

    result = {
        "schema": "m511_static_check_r4_v1",
        "status": "PASS_STATIC_FINAL_CLOSURE",
        "execution_boundary": {
            "production_capture_executed": False,
            "checkpoint_loaded": False,
            "model_constructed": False,
            "cuda_touched": False,
            "vcs_dc_dse_executed": False,
        },
        "identity": {
            "producer_sha256": PRODUCER_SHA,
            "contract_sha256": CONTRACT_SHA,
            "checked_input_count": len(checked),
            "checked_inputs": checked,
        },
        "new_r4_gate": {
            "complete_runtime_convtranspose_set_and_order_exact": True,
            "expected_names": expected_names,
            "gate_precedes_hook_registration_and_capture": True,
        },
        "regression": {
            "payload_elements_s10": elements_s10,
            "payload_bytes_s10": bytes_s10,
            "hooks_removed_before_publish": True,
            "source_contract_inputs_rehashed": True,
            "exact_member_seal": True,
            "quarantine_precomputed": True,
            "postpublish_first_recovery_is_atomic_rename": True,
            "m512_provenance_pinned": all(
                name in contract["inputs"] for name in
                ("m512_review", "m512_review_manifest", "m512_review_seal")),
        },
    }
    output = Path(__file__).with_name("m511_static_check_r4.json")
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    print("PASS_STATIC_CHECK", output, sha256(output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
