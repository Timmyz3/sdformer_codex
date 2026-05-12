"""Experiment-local SDFormerFlow entrypoint.

This file keeps launch-time changes inside the experiment directory. It runs the
baseline SDFormerFlow script while placing this experiment's overlay before the
baseline on sys.path.
"""

from __future__ import annotations

import argparse
import types
import os
import runpy
import sys
from pathlib import Path


TRAIN_BLOCK = """    if config["model"]["spiking_neuron"]["neuron_type"] == "if":
        neurontype = getattr(neuron, "IFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "lif":
        neurontype = getattr(neuron, "LIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "plif":
        neurontype = getattr(neuron, "ParametricLIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "glif":
        neurontype = GatedLIFNode
    elif config["model"]["spiking_neuron"]["neuron_type"] == "psn":
        neurontype = PSN
    elif config["model"]["spiking_neuron"]["neuron_type"] == "SLTTlif":
        neurontype = SLTTLIFNode
    else:
        raise "neurontype not implemented!"
"""

PATCHED_BLOCK = """    if config["model"]["spiking_neuron"]["neuron_type"] == "if":
        neurontype = getattr(neuron, "IFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "lif":
        neurontype = getattr(neuron, "LIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "plif":
        neurontype = getattr(neuron, "ParametricLIFNode")
    elif config["model"]["spiking_neuron"]["neuron_type"] == "glif":
        neurontype = GatedLIFNode
    elif config["model"]["spiking_neuron"]["neuron_type"] == "psn":
        neurontype = PSN
    elif config["model"]["spiking_neuron"]["neuron_type"] == "SLTTlif":
        neurontype = SLTTLIFNode
    else:
        from models.STSwinNet_SNN.experimental_neurons.factory import resolve_backend_neuron_type
        neurontype = resolve_backend_neuron_type(config["model"]["spiking_neuron"]["neuron_type"])
"""

EXPERIMENTAL_TRAIN_IMPORT = (
    "from models.STSwinNet_SNN.Spiking_submodules import *\n"
    "from models.STSwinNet_SNN.experimental_neurons.training import freeze_experimental_parameters, regularize_activity, sanitize_threshold_grads, threshold_update\n"
)

LOSS_REGULARIZATION_ANCHOR = """                curr_loss.item()
                if (sample + 1) % 50 == 0:
                    try:
                        t_vals = []
                        for n, m in model.named_modules():
                            if hasattr(m, "thresh"):
                                t_vals.append(m.thresh.detach().mean().item())
                        if t_vals:
                            print(f"\n[Step {sample+1}] Current Mean Thresh: {sum(t_vals)/len(t_vals):.5f}")
                    except Exception as e:
                        pass
"""

LOSS_REGULARIZATION_PATCH = """                curr_loss = curr_loss + regularize_activity(model, config) / num_acc_steps
                curr_loss.item()
                if (sample + 1) % 50 == 0:
                    try:
                        t_vals = []
                        for n, m in model.named_modules():
                            if hasattr(m, "thresh"):
                                t_vals.append(m.thresh.detach().mean().item())
                        if t_vals:
                            print(f"\n[Step {sample+1}] Current Mean Thresh: {sum(t_vals)/len(t_vals):.5f}")
                    except Exception as e:
                        pass
"""

SCALER_STEP_ANCHOR = """                    scaler.step(optimizer)
                    scaler.update()
"""

SCALER_STEP_PATCH = """                    scaler.step(optimizer)
                    threshold_update(model, optimizer.param_groups[0]["lr"], config)
                    scaler.update()
"""

OPTIMIZER_STEP_ANCHOR = """                    optimizer.step()

                # zero grad
"""

OPTIMIZER_STEP_PATCH = """                    optimizer.step()
                    threshold_update(model, optimizer.param_groups[0]["lr"], config)

                # zero grad
"""

PIN_MEMORY_ANCHOR = """"pin_memory": True,
"""

PIN_MEMORY_PATCH = """"pin_memory": bool(config["loader"].get("pin_memory", True)),
"""

BACKWARD_ANCHOR = """            if config["loss"]["clip_grad"] is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
"""

BACKWARD_PATCH = """            sanitize_threshold_grads(model, config)
            if config["loss"]["clip_grad"] is not None:
                torch.nn.utils.clip_grad.clip_grad_norm_(model.parameters(), config["loss"]["clip_grad"])
"""

CONFIGURE_BACKEND_ANCHOR = """    configure_snn_backend(model, device, config, neurontype)

    #summary(model)
"""

CONFIGURE_BACKEND_PATCH = """    configure_snn_backend(model, device, config, neurontype)
    freeze_stats = freeze_experimental_parameters(model, config)
    if freeze_stats.get("mode") != "none":
        print(f"[experimental] freeze_stats={freeze_stats}")

    #summary(model)
"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _install_optional_mlflow_stub() -> None:
    disabled = os.getenv("SDFORMER_USE_MLFLOW", "1").lower() in {"0", "false", "no"}
    if not disabled:
        return
    try:
        __import__("mlflow")
    except ModuleNotFoundError:
        sys.modules["mlflow"] = types.ModuleType("mlflow")


def _absolutize_path_args(extra_args: list[str]) -> list[str]:
    path_flags = {"--save_path", "--resume", "--finetune", "--path_mlflow", "--prev_runid"}
    normalized = list(extra_args)
    index = 0
    while index < len(normalized):
        item = normalized[index]
        if item in path_flags and index + 1 < len(normalized):
            normalized[index + 1] = str(Path(normalized[index + 1]).resolve())
            index += 2
            continue
        matched = next((flag for flag in path_flags if item.startswith(f"{flag}=")), None)
        if matched is not None:
            value = item.split("=", 1)[1]
            normalized[index] = f"{matched}={Path(value).resolve()}"
        index += 1
    return normalized


def _run_baseline(entry_name: str, config: str, extra_args: list[str]) -> None:
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = _repo_root()
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = experiment_root / "overlay"
    baseline_entry = baseline_root / entry_name

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))
    sys.argv = [str(baseline_entry), "--config", str(Path(config).resolve()), *_absolutize_path_args(extra_args)]

    _install_optional_mlflow_stub()
    os.chdir(baseline_root)
    source = baseline_entry.read_text()
    factory = overlay_root / "models" / "STSwinNet_SNN" / "experimental_neurons" / "factory.py"
    if factory.exists():
        if TRAIN_BLOCK not in source:
            raise RuntimeError(f"Could not patch backend neuron block in {baseline_entry}")
        source = source.replace(TRAIN_BLOCK, PATCHED_BLOCK)
        source = source.replace("from models.STSwinNet_SNN.Spiking_submodules import *\n", EXPERIMENTAL_TRAIN_IMPORT)
        source = source.replace(PIN_MEMORY_ANCHOR, PIN_MEMORY_PATCH)
        source = source.replace(LOSS_REGULARIZATION_ANCHOR, LOSS_REGULARIZATION_PATCH)
        source = source.replace(CONFIGURE_BACKEND_ANCHOR, CONFIGURE_BACKEND_PATCH)
        source = source.replace(BACKWARD_ANCHOR, BACKWARD_PATCH)
        source = source.replace(SCALER_STEP_ANCHOR, SCALER_STEP_PATCH)
        source = source.replace(OPTIMIZER_STEP_ANCHOR, OPTIMIZER_STEP_PATCH)
        code = compile(source, str(baseline_entry), "exec")
        exec(code, {"__name__": "__main__", "__file__": str(baseline_entry)})
    else:
        runpy.run_path(str(baseline_entry), run_name="__main__")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args, extra_args = parser.parse_known_args()
    _run_baseline("train_flow_parallel_supervised_SNN.py", args.config, extra_args)


if __name__ == "__main__":
    main()
