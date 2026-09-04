#!/usr/bin/env python3
from copy import deepcopy
import importlib.util
import math
from pathlib import Path


HW = Path(__file__).resolve().parents[2]
SOURCE = HW / "system_simulator/scripts/build_m1525_ep34_decoder_multibaseline_replay_successor_source.py"
spec = importlib.util.spec_from_file_location("m1525", SOURCE)
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)


def planes():
    rows = []
    for ordinal in range(120):
        module = ordinal % 4; sample = 10 + ordinal // 4
        shape = M.SHAPES[module]
        rows.append({"global_call_ordinal": ordinal, "global_sample_id": sample,
            "replay_sample_ordinal": sample - 10, "module_ordinal": module,
            "module": M.MODULES[module], "shape": list(shape),
            "elements": M.product(shape), "plane_bytes": (M.product(shape)+7)//8,
            "positive_output": "payloads/c{:03d}_s{:02d}_d{}.positive.le.bitpack".format(ordinal,sample,module),
            "positive_output_sha256": "a"*64,
            "layer_scale_word_uint32": M.SCALE_WORDS[module],
            "numeric_encoding": "bit_times_layer_constant" if module in (0,1) else "exact_binary",
            "weight_folding": False, "normalized": False, "coerced": False,
            "negative_plane_output": None, "negative_plane_all_zero": True})
    return {"capture":{"checkpoint_sha256":M.CHECKPOINT_SHA256},
        "population":{"samples":30,"calls":120,"modules":4,
            "positive_plane_files":120,"negative_plane_files":0}, "records":rows}


def weights():
    rows=[]
    for i, shape in enumerate(M.WEIGHT_SHAPES):
        rows.append({"module_ordinal":i,"module":M.MODULES[i],"shape":list(shape),
            "dtype":"torch.float32","layout":"C_ORDER_CONTIGUOUS","byte_order":"little",
            "content_sha256":M.WEIGHT_SHA256[i],"content_bytes":M.product(shape)*4,"bias":None})
    return {"status":"PASS_M1514_SOURCE_ONLY_DECODER_WEIGHT_IDENTITY__NO_EXPORT",
        "checkpoint":{"sha256":M.CHECKPOINT_SHA256,"root_keys":["model_state_dict"]},
        "weights":rows}


def rejects(fn):
    try: fn()
    except M.M1525Error: return
    raise AssertionError("attack accepted")


def main():
    p=planes(); w=weights(); plan=M.build_replay_plan(p,w)
    assert [x["name"] for x in plan["configurations"]] == list(M.CONFIGS)
    assert plan["readiness"]["product_capture_ready"] is False
    assert plan["old_m1105dr2_reuse"]["allowed"] is False
    attacks=[]
    def attack(label, mutate):
        value=deepcopy(p); mutate(value); rejects(lambda:M.validate_positive_plane_manifest(value)); attacks.append(label)
    attack("old_checkpoint",lambda x:x["capture"].__setitem__("checkpoint_sha256",M.OLD_EP35_CHECKPOINT_SHA256))
    attack("old_d1",lambda x:x["records"][1].__setitem__("layer_scale_word_uint32",M.OLD_D1_WORD))
    attack("d0_binary_one",lambda x:x["records"][0].__setitem__("layer_scale_word_uint32",0x3f800000))
    attack("fold",lambda x:x["records"][0].__setitem__("weight_folding",True))
    attack("coerce",lambda x:x["records"][0].__setitem__("coerced",True))
    attack("path",lambda x:x["records"][0].__setitem__("positive_output","payloads/forged"))
    attack("reorder",lambda x:x["records"].__setitem__(slice(0,2),list(reversed(x["records"][:2]))))
    bad=deepcopy(w); bad["weights"][2]["content_sha256"]="0"*64
    rejects(lambda:M.validate_weight_identity(bad)); attacks.append("weight_sha")
    rejects(lambda:M.build_replay_plan(p,w,request_production=True)); attacks.append("production")
    rows=[{"configuration":name,"resource_manifest_sha256":"r","commit_address_hash":"c",
           "population_manifest_sha256":"p","checkpoint_sha256":M.CHECKPOINT_SHA256} for name in M.CONFIGS]
    M.validate_comparator_rows(rows)
    badrows=deepcopy(rows); badrows[-1]["resource_manifest_sha256"]="other"
    rejects(lambda:M.validate_comparator_rows(badrows)); attacks.append("resource_mismatch")
    assert len(attacks)==10
    print("PASS M1525 source tests attacks=10 configs=4 production=0 product_ready=0")


if __name__ == "__main__": main()
