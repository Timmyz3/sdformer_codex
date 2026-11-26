"""Independent E3 masks and NumPy signed-integer checks. No GPU or real GT."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reduce_e1_quality_v1 as b
E3 = HERE.parent/"e3_rne_probe"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    assert not a.output.exists()
    torch.set_num_threads(2)
    frozen = b.read(E3/"SOURCE_FREEZE_V1.json")
    assert all(b.sha(E3/n) == h for n,h in frozen["files"].items())
    spec = importlib.util.spec_from_file_location("_independent_e3_adapter", E3/"probe_contract.py")
    pc = importlib.util.module_from_spec(spec); spec.loader.exec_module(pc)
    core = pc.load_core(HERE.parent/"e0_ordinary/mixed_integer_core.py")
    c = b.read(E3/"PREPARED_CONTRACT_V1.json")
    ids = b.read(HERE.parent/"selections_v1.json")["e2_reserved_ids"]
    assert c["target_ids"] == ids
    try:
        pc.validate(c, core)
        raise AssertionError("Unbound endpoint accepted")
    except ValueError:
        pass
    counts = []
    for action in range(-1,6):
        bound = pc.BoundCore(core, c, action)
        assert bound.mode == ("Q4" if action == -1 else "M")
        subtotal = 0
        for name,g in c["geometries"].items():
            actual = bound.masks[core.group_name(name)].numpy()
            expected = np.zeros((g["H"],g["W"]), dtype=bool)
            if action >= 0 and name in core.CONVS:
                for index in range(action*151,(action+1)*151):
                    y,x = divmod(index,35)
                    expected[y*9:y*9+9,x*9:x*9+9] = True
            assert np.array_equal(actual,expected)
            subtotal += int(expected.sum())*10*4*((g["K"]+127)//128)*((g["Cout"]+15)//16)
        assert subtotal == (0 if action == -1 else 82192320)
        counts.append(subtotal)
    rng = np.random.default_rng(20261021)
    checks = 0
    for kind in ("linear", "conv"):
        source = torch.nn.Linear(18,3,bias=False) if kind=="linear" else torch.nn.Conv2d(2,3,3,padding=1,bias=False)
        weights = rng.normal(size=tuple(source.weight.shape)).astype(np.float32)
        weights[0] = 0
        with torch.no_grad():
            source.weight.copy_(torch.from_numpy(weights))
        if kind == "linear":
            xx = rng.integers(0,2,(10,1,10,55,18),dtype=np.int8)
            rows = xx.reshape(-1,18).astype(np.int64)
        else:
            xx = rng.integers(0,2,(10,1,2,10,55),dtype=np.int8)
            padded = np.pad(xx,((0,0),(0,0),(0,0),(1,1),(1,1)))
            rows = np.stack([padded[:,:,ch,dy:dy+10,dx:dx+55]
                             for ch in range(2) for dy in range(3) for dx in range(3)],axis=-1).reshape(-1,18).astype(np.int64)
        ww = weights.reshape(3,18)
        maximum = np.max(np.abs(ww),axis=1)
        s8, s4 = maximum/np.float32(127), maximum/np.float32(7)
        q8 = np.rint(ww/np.where(s8==0,np.float32(1),s8)[:,None]).clip(-127,127).astype(np.int64)
        q4 = (np.rint(ww/np.where(s4==0,np.float32(1),s4)[:,None]).clip(-8,7).astype(np.int64)<<4)
        for action in range(-1,6):
            chosen = np.zeros((10,55), dtype=bool)
            if action >= 0:
                chosen[:9,action*9:(action+1)*9] = True
            high = np.ones((10,55),dtype=bool); high[:9,:54] = False
            high |= chosen
            high = np.broadcast_to(high,(10,1,10,55)).reshape(-1)
            integer = np.empty((len(rows),3),dtype=np.int64)
            integer[high] = rows[high] @ q8.T
            integer[~high] = rows[~high] @ q4.T
            yy = np.empty((len(rows),3),dtype=np.float32)
            yy[high] = integer[high].astype(np.float32)*s8
            yy[~high] = integer[~high].astype(np.float32)*(s4/np.float32(16))
            expected = yy.reshape(10,1,10,55,3)
            if kind=="conv":
                expected = expected.transpose(0,1,4,2,3).copy()
            layer = core.STEAffine(source,kind,"Q4" if action==-1 else "M",
                selected=None if action==-1 else torch.from_numpy(chosen),ste=False,variant="rne4")
            layer.capture_integer = True
            with torch.no_grad():
                out = layer(torch.from_numpy(xx.astype(np.float32)))
            assert np.array_equal(layer.last_integer.numpy(), integer.astype(np.int32))
            assert np.array_equal(out.numpy().view(np.uint8), expected.view(np.uint8))
            assert layer.weight is source.weight
            checks += 1
    assert not torch.cuda.is_initialized()
    result = dict(schema="flow_e3_independent_cpu_review_v1",status="PASS",
                  source_freeze_sha256=b.sha(E3/"SOURCE_FREEZE_V1.json"),
                  source_sha256={n:b.sha(E3/n) for n in ("probe_contract.py","collect_probe.py")},
                  fixed_mask_checks=7*28, independent_NumPy_integer_cases=checks,
                  includes=["all7_actions","signed_weights","zero_weight_output_channel","conv_halo","true_tail","independent_scales"],
                  GPU_used=False,GT_read=False,full_model_forwards=0,endpoint_bound=False,
                  script_sha256=b.sha(__file__))
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
