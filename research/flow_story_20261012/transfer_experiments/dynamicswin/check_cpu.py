#!/usr/bin/env python3
"""Bounded CPU evidence; synthetic tensors only, no checkpoint or optimizer."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import ast
import copy
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
sys.dont_write_bytecode = True
import torch
from torch import nn
from torch.nn import functional as F
from adapter import Config, DynamicSwinMLPAdapter, FastBranch, LocalGlobalPredictor, ratio_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PIN = "1322e626d1e9eca18cc150569908fb4af29e15f7"


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def official_classes():
    source = HERE / "upstream/models/dyswin.py"
    tree = ast.parse(source.read_text())
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("fastMlp", "PredictorLG")]
    assert len(selected) == 2
    namespace = {"torch": torch, "nn": nn, "F": F}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace, {n.name: {"line_start": n.lineno, "line_end": n.end_lineno} for n in selected}


def native_factory():
    probe = ROOT / "research/flow_story_20261011/preflight/cpu_numerics/native_mlp_probe.py"
    spec = importlib.util.spec_from_file_location("flow_native_mlp_probe_for_dynamic_adapter", probe)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_mlp, probe


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    assert not torch.cuda.is_initialized()
    revision = subprocess.check_output(["git", "-C", str(HERE / "upstream"), "rev-parse", "HEAD"], text=True).strip()
    assert revision == PIN, revision
    upstream_files = ["README.md", "LICENSE", "models/dyswin.py", "losses.py", "main.py", "utils.py"]
    for name in upstream_files:
        # Compare working bytes with fixed-commit bytes, not only moving branch names.
        committed = subprocess.check_output(["git", "-C", str(HERE / "upstream"), "show", f"{PIN}:{name}"])
        assert (HERE / "upstream" / name).read_bytes() == committed, name
    official, lines = official_classes()
    provenance = {
        "repository": "https://github.com/raoyongming/DynamicViT", "commit": PIN,
        "paper": "https://arxiv.org/pdf/2207.01580v2",
        "files": {n: {"sha256": sha(HERE/"upstream"/n), "verified_against_git_object": True,
                      "url": f"https://github.com/raoyongming/DynamicViT/blob/{PIN}/{n}"} for n in upstream_files},
        "class_locations": lines,
        "origin": "official source inspected; fastMlp/PredictorLG class AST used only as CPU references",
        "local_module_origin": "mechanism adaptation; no attention/network code imported from DynamicViT",
    }
    (HERE / "source_provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    records = []

    # Exact topology and floating arithmetic against official class definitions.
    torch.manual_seed(1210)
    for c in (8, 96):
        official_p = official["PredictorLG"](c).double().eval()
        local_p = LocalGlobalPredictor(c).double().eval()
        local_p.load_state_dict(official_p.state_dict(), strict=True)
        x = torch.randn(1, 110, c, dtype=torch.float64)
        z = official_p.in_conv(x)
        expected = official_p.out_conv(torch.cat((z[..., :c//2], z[..., c//2:].mean(1, keepdim=True).expand(1, 110, c//2)), -1))
        actual = local_p(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        _, (slow_ids, _) = official_p(x, ratio=0.5)
        our_ids = torch.argsort(actual[..., 0], dim=1, descending=True, stable=True)[:, :55]
        assert torch.equal(slow_ids, our_ids)
        official_f = official["fastMlp"](c).double().eval()
        local_f = FastBranch(c).double().eval()
        local_f.load_state_dict(official_f.state_dict(), strict=True)
        torch.testing.assert_close(local_f(x), official_f(x), rtol=0, atol=0)
        records.append({"kind": "official_topology_and_math", "channels": c, "max_abs_error": 0.0, "passed": True})

    make_mlp, probe = native_factory()
    traces = []
    for c, dtype, tile in ((8, torch.float64, 9), (96, torch.float32, 9), (8, torch.float64, 1)):
        torch.manual_seed(771+c+tile)
        slow = make_mlp(c, dtype)
        adapter = DynamicSwinMLPAdapter(slow, Config(c, 4*c, tile=tile)).to(dtype=dtype).eval()
        u = torch.randn(10, 1, 10, 11, c, dtype=dtype)
        ny, nx = (10+tile-1)//tile, (11+tile-1)//tile
        yy, xx = torch.meshgrid(torch.arange(ny), torch.arange(nx), indexing="ij")
        masks = {"all_slow": torch.ones(ny,nx,dtype=torch.bool),
                 "all_fast": torch.zeros(ny,nx,dtype=torch.bool),
                 "checkerboard": (yy+xx)%2 == 0,
                 "last_edge_slow": (yy == ny-1)&(xx == nx-1)}
        with torch.inference_mode():
            full_slow, full_fast = slow(u), adapter.fast(u)
            for name, mask in masks.items():
                observed = {"fc1_rows": 0, "fc2_rows": 0, "fast_linear_rows": 0}
                handles = []
                for layer, key in ((slow.fc1,"fc1_rows"),(slow.fc2,"fc2_rows"),(adapter.fast.fc1[1],"fast_linear_rows")):
                    def hook(module, inputs, label=key):
                        observed[label] += inputs[0].numel()//inputs[0].shape[-1]
                    handles.append(layer.register_forward_pre_hook(hook))
                try:
                    actual = adapter(u, execution="sparse_eval", forced_keep=mask)
                finally:
                    for handle in handles: handle.remove()
                reference = adapter(u, execution="dense_reference", forced_keep=mask)
                error = (actual.branch-reference.branch).abs().max().item()
                tolerance = 2e-5 if dtype == torch.float32 else 1e-11
                torch.testing.assert_close(actual.branch, reference.branch, rtol=tolerance, atol=tolerance)
                torch.testing.assert_close(actual.add_residual(u), u+reference.branch, rtol=tolerance, atol=tolerance)
                if name == "all_slow":
                    torch.testing.assert_close(actual.branch, full_slow, rtol=0, atol=0)
                if name == "all_fast":
                    torch.testing.assert_close(actual.branch, full_fast, rtol=0, atol=0)
                expected_slow = 10*actual.trace["selected_slow_points"]
                expected_fast = 10*actual.trace["selected_fast_points"]
                assert observed == {"fc1_rows": expected_slow,"fc2_rows": expected_slow,"fast_linear_rows": expected_fast}
                assert sum(actual.trace["executed_spatial_points"].values()) == 110
                if tile == 9:
                    assert [z["valid_points"] for z in actual.trace["tile_spans"]] == [81,18,9,2]
                    if name == "last_edge_slow": assert expected_slow == 20
                traces.append({"case": name, "channels": c, "dtype": str(dtype), "trace": actual.trace, "observed_hooks": observed})
                records.append({"kind": "native_MLP_forward", "case": name, "channels": c, "tile": tile,
                                "dtype": str(dtype), "max_abs_error": error, "tolerance": tolerance, "passed": True})
            # A learned-score eval uses no forced mask and charges the selector.
            selected = adapter(u, execution="sparse_eval")
            assert sum(sum(r) for r in selected.trace["tile_keep"]) == (ny*nx)//2
            assert selected.trace["cost"]["selector"]["executed"]
            traces.append({"case": "untrained_predictor_eval", "channels": c, "trace": selected.trace})
            records.append({"kind": "predictor_eval_exact_tile_count", "channels": c, "tile": tile, "passed": True})

    # Recovery graph: frozen stats, but all intended parameters receive gradients.
    torch.manual_seed(123)
    slow = make_mlp(8, torch.float64)
    adapter = DynamicSwinMLPAdapter(slow, Config(8, 32)).double().train()
    u = torch.randn(10,1,18,21,8,dtype=torch.float64, requires_grad=True)
    bn_before = [(m, m.running_mean.clone(), m.running_var.clone(), m.num_batches_tracked.clone())
                 for m in slow.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    result = adapter(u, execution="dense_train")
    assert set(result.tile_gate.detach().unique().tolist()).issubset({0.,1.})
    assert set(result.tile_gate.detach().unique().tolist()) == {0.,1.}
    # This is a synthetic differentiability check, not optical-flow training.
    loss = result.add_residual(u).square().mean()+ratio_loss(result,0.3)
    loss.backward()
    gradient_groups = {}
    for name, module in (("predictor",adapter.predictor),("fast",adapter.fast),("native_slow",slow)):
        total = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
        assert total > 0
        assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
        gradient_groups[name] = total
    assert u.grad is not None and torch.isfinite(u.grad).all() and u.grad.abs().sum() > 0
    for m, mean, var, count in bn_before:
        assert not m.training and torch.equal(mean,m.running_mean) and torch.equal(var,m.running_var) and torch.equal(count,m.num_batches_tracked)
    assert result.trace["executed_spatial_points"] == {"slow": 18*21, "fast":18*21}
    records.append({"kind":"dense_ST_Gumbel_native_gradient", "gradient_absolute_sums":gradient_groups,
                    "input_gradient_absolute_sum":u.grad.abs().sum().item(),"BN_stats_unchanged":True,"optimizer_steps":0,"passed":True})
    traces.append({"case":"synthetic_gradient", "trace":result.trace})

    rejected=[]
    def rejects(label, fn):
        try: fn()
        except ValueError as exc: rejected.append({"case":label,"message":str(exc),"passed":True})
        else: raise AssertionError("Guard missing: "+label)
    adapter.eval()
    rejects("inference_with_grad_enabled",lambda:adapter(u, execution="sparse_eval"))
    rejects("wrong_T",lambda:adapter(u[:9]))
    rejects("wrong_B",lambda:adapter(u.expand(10,2,18,21,8)))
    rejects("bad_mask_shape",lambda:adapter(u,forced_keep=torch.ones(1,1,dtype=torch.bool)))
    rejects("native_cost_dimension_mismatch",lambda:DynamicSwinMLPAdapter(slow,Config(8,16)))
    bad=copy.deepcopy(adapter)
    for m in bad.slow.modules():
        if isinstance(m,nn.modules.batchnorm._BatchNorm):
            m.track_running_stats=False
            m.running_mean=m.running_var=None
    rejects("current_stat_BN",lambda:bad(u))
    records.extend({"kind":"rejection",**r} for r in rejected)
    assert not torch.cuda.is_initialized()
    native_source = ROOT/"SDformer/third_party/SDformerFlow/models/STSwinNet_SNN"
    output={"scope":"synthetic CPU forward/backward and source parity; not recovery, AEE, GPU, cycles or full integration",
            "python":platform.python_version(),"torch":torch.__version__,"device":"cpu","cuda_initialized":False,
            "threads":2,"upstream_commit":revision,"optimizer_steps":0,"checkpoint_loaded":False,
            "teacher_executed":False,"full_network_integrated":False,
            "source_sha256":{n:sha(HERE/n) for n in ("adapter.py","check_cpu.py","config.json")},
            "native_source_sha256":{n:sha(native_source/n) for n in ("Spiking_swin_transformer3D.py","Spiking_submodules.py","Spiking_modules.py")},
            "native_factory_sha256":sha(probe),"passed_cases":len(records),"records":records,
            "elapsed_seconds":time.monotonic()-start}
    (HERE/"cpu_results.json").write_text(json.dumps(output,indent=2)+"\n")
    (HERE/"cost_and_execution_examples.json").write_text(json.dumps(traces,indent=2)+"\n")
    print(json.dumps({"passed_cases":len(records),"elapsed_seconds":output["elapsed_seconds"],"source_pin":revision,"cuda_initialized":False}))


if __name__ == "__main__":
    main()
