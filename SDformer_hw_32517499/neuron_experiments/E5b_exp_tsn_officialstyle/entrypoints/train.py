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

PIN_MEMORY_ANCHOR = """"pin_memory": True,
"""

PIN_MEMORY_PATCH = """"pin_memory": bool(config["loader"].get("pin_memory", True)),
"""

OPTIMIZER_BLOCK = """    # optimizers
    if config["optimizer"]["name"] == 'AdamW':
        optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"],weight_decay=config["optimizer"]["wd"])

    else:
        optimizer = eval(config["optimizer"]["name"])(model.parameters(), lr=config["optimizer"]["lr"])
"""

OFFICIAL_TSN_OPTIMIZER_BLOCK = """    # optimizers
    optimizer_name = config["optimizer"]["name"]
    if bool(config["optimizer"].get("official_tsn_split_weights", False)):
        decay = []
        no_decay = []
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
                decay.append(module.weight)
                if module.bias is not None:
                    no_decay.append(module.bias)
            else:
                if hasattr(module, "weight") and isinstance(module.weight, torch.nn.Parameter):
                    no_decay.append(module.weight)
                if hasattr(module, "bias") and isinstance(module.bias, torch.nn.Parameter):
                    no_decay.append(module.bias)
                if hasattr(module, "fire_ratio") and isinstance(module.fire_ratio, torch.nn.Parameter):
                    no_decay.append(module.fire_ratio)
        seen = set()
        decay_unique = []
        no_decay_unique = []
        for param in decay:
            if id(param) not in seen:
                seen.add(id(param))
                decay_unique.append(param)
        for param in no_decay:
            if id(param) not in seen:
                seen.add(id(param))
                no_decay_unique.append(param)
        param_groups = [
            {"params": decay_unique},
            {"params": no_decay_unique, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.SGD(
            params=param_groups,
            lr=config["optimizer"]["lr"],
            momentum=config["optimizer"].get("momentum", 0.9),
            weight_decay=config["optimizer"].get("wd", 1e-4),
        )
        print(f"[E5b] official TSN split_weights: decay={len(decay_unique)} no_decay={len(no_decay_unique)} "
              f"lr={config['optimizer']['lr']} momentum={config['optimizer'].get('momentum', 0.9)} "
              f"wd={config['optimizer'].get('wd', 1e-4)}")
    elif optimizer_name == 'AdamW':
        optimizer = eval(optimizer_name)(model.parameters(), lr=config["optimizer"]["lr"],weight_decay=config["optimizer"]["wd"])

    else:
        optimizer = eval(optimizer_name)(model.parameters(), lr=config["optimizer"]["lr"])
"""

SCHEDULER_BLOCK = """    if config["optimizer"]["scheduler"] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["optimizer"]["milestones"], gamma=0.5)
    else:
        scheduler = None
"""

OFFICIAL_TSN_SCHEDULER_BLOCK = """    if config["optimizer"]["scheduler"] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config["loader"]["n_epochs"],
            eta_min=config["optimizer"].get("eta_min", 0.0),
        )
    elif config["optimizer"]["scheduler"] == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=config["optimizer"]["milestones"], gamma=0.5)
    else:
        scheduler = None
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
    path_flags = {"--save_path", "--resume", "--finetune", "--path_mlflow"}
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
        source = source.replace(PIN_MEMORY_ANCHOR, PIN_MEMORY_PATCH)
        if OPTIMIZER_BLOCK not in source:
            raise RuntimeError(f"Could not patch optimizer block in {baseline_entry}")
        if SCHEDULER_BLOCK not in source:
            raise RuntimeError(f"Could not patch scheduler block in {baseline_entry}")
        source = source.replace(OPTIMIZER_BLOCK, OFFICIAL_TSN_OPTIMIZER_BLOCK)
        source = source.replace(SCHEDULER_BLOCK, OFFICIAL_TSN_SCHEDULER_BLOCK)
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
