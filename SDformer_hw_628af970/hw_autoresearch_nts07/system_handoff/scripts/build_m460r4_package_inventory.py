#!/usr/bin/env python3
"""Build and verify the read-only M460R4 Python/package/build inventory.

This entrypoint imports the frozen runtime closure but never constructs a
model, initializes CUDA, reads a checkpoint, or launches a capture.  It is
Python-3.6 syntax compatible; the sealed remote execution uses Python 3.10.
"""

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys


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
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON key: " + key)
            result[key] = value
        return result

    def reject(token):
        raise RuntimeError("non-standard JSON number: " + token)

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle, object_pairs_hook=pairs,
                         parse_constant=reject)


def check_output(argv):
    return subprocess.check_output(argv, universal_newlines=True).strip()


def distribution_for(name):
    try:
        from importlib import metadata
        return metadata.distribution(name)
    except ImportError:
        try:
            import importlib_metadata
            return importlib_metadata.distribution(name)
        except ImportError:
            import pkg_resources
            return pkg_resources.get_distribution(name)


def distribution_paths(distribution):
    base = getattr(distribution, "_path", None)
    if base is None:
        location = Path(distribution.location)
        candidates = sorted(location.glob(
            "{}-*.dist-info".format(distribution.project_name.replace("-", "_"))))
        require(len(candidates) == 1,
                "cannot resolve unique dist-info for " + distribution.project_name)
        base = candidates[0]
    base = Path(base).resolve()
    return base / "METADATA", base / "RECORD"


def load_file_module(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    require(spec is not None and spec.loader is not None,
            "cannot import file " + str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def package_record(expected):
    module = importlib.import_module(expected["module"])
    module_path = Path(module.__file__).resolve()
    distribution = distribution_for(expected["distribution"])
    metadata_path, record_path = distribution_paths(distribution)
    version = getattr(distribution, "version", None)
    if version is None:
        version = distribution.version
    return {
        "module": expected["module"],
        "distribution": expected["distribution"],
        "module_version": getattr(module, "__version__", None),
        "distribution_version": str(version),
        "module_file": str(module_path),
        "module_file_sha256": sha256(module_path),
        "metadata_file": str(metadata_path),
        "metadata_sha256": sha256(metadata_path),
        "record_file": str(record_path),
        "record_sha256": sha256(record_path),
    }


def runtime_import_records(code_repo, expected_records):
    code_repo = Path(code_repo).resolve()
    profile_path = (code_repo / "neuron_experiments/H9_bipolar_self_attention/"
                    "entrypoints/profile_nts11_hardware_p0.py")
    require(profile_path.is_file(), "frozen profile absent from clean code repo")
    load_file_module(profile_path, "m460r4_profile_inventory_probe")

    models = importlib.import_module("models")
    stsnn = importlib.import_module("models.STSwinNet_SNN")
    overlay_models = (code_repo / "neuron_experiments/"
                      "H9_bipolar_self_attention/overlay/models")
    overlay_stsnn = overlay_models / "STSwinNet_SNN"
    if str(overlay_models) not in list(models.__path__):
        models.__path__.append(str(overlay_models))
    if str(overlay_stsnn) not in list(stsnn.__path__):
        stsnn.__path__.append(str(overlay_stsnn))

    records = []
    for expected in expected_records:
        module = importlib.import_module(expected["module"])
        origin = getattr(module, "__file__", None)
        if origin is None:
            resolved = None
            observed_sha = None
        else:
            resolved = str(Path(origin).resolve())
            observed_sha = sha256(resolved)
        records.append({
            "module": expected["module"],
            "origin": resolved,
            "sha256": observed_sha,
        })
    return records


def expected_runtime_origin(code_repo, expected):
    if expected["origin"] is None:
        return None
    if expected["root"] == "code_repo":
        return str((Path(code_repo).resolve() / expected["origin"]).resolve())
    require(expected["root"] == "site_packages",
            "unknown runtime import root: " + expected["root"])
    return str(Path(expected["origin"]).resolve())


def validate_inventory(inventory, freeze):
    require(inventory["schema"] == "m460r4_live_package_build_inventory_v1",
            "live inventory schema drift")
    require(inventory["python"] == freeze["python"],
            "Python executable/build/conda-history identity drift")
    require(inventory["packages"] == freeze["packages"],
            "critical Python package identity drift")
    require(inventory["build"] == freeze["build"],
            "CUDA/cuDNN/driver binary build identity drift")
    require(inventory["isolation"]["python_isolated"] is True,
            "Python -I isolation absent")
    require(inventory["isolation"]["PYTHONNOUSERSITE"] == "1",
            "PYTHONNOUSERSITE is not frozen to 1")
    require(inventory["isolation"]["PYTHONPATH"] is None,
            "PYTHONPATH must be unset")
    require(inventory["cuda_initialized"] is False,
            "inventory unexpectedly initialized CUDA")

    observed = inventory["runtime_imports"]
    expected = freeze["runtime_imports"]
    require(len(observed) == len(expected), "runtime import population drift")
    for actual, target in zip(observed, expected):
        require(actual["module"] == target["module"],
                "runtime import order/name drift")
        require(actual["origin"] == expected_runtime_origin(
            inventory["code_repo"], target),
            "runtime import origin drift: " + target["module"])
        require(actual["sha256"] == target["sha256"],
                "runtime import file SHA drift: " + target["module"])

    forbidden = freeze["isolation"]["forbidden_sys_path_substrings"]
    for item in inventory["final_sys_path"]:
        for fragment in forbidden:
            require(fragment not in item,
                    "forbidden original/usersite sys.path entry: " + item)
    return True


def collect_inventory(code_repo, freeze):
    code_repo = Path(code_repo).resolve()
    initial_sys_path = [str(item) for item in sys.path]
    require(sys.flags.isolated == 1,
            "M460R4 inventory must be launched with Python -I")
    require(os.environ.get("PYTHONNOUSERSITE") == "1",
            "M460R4 requires PYTHONNOUSERSITE=1")
    require("PYTHONPATH" not in os.environ,
            "M460R4 requires PYTHONPATH to be unset")
    for item in initial_sys_path:
        require(item in freeze["isolation"]["initial_sys_path_allowed_prefixes"],
                "unexpected initial isolated sys.path entry: " + item)

    executable = Path(sys.executable).resolve()
    history = Path(freeze["python"]["conda_history"]["path"]).resolve()
    python_record = {
        "executable": sys.executable,
        "realpath": str(executable),
        "sha256": sha256(executable),
        "version": sys.version,
        "version_info": list(sys.version_info[:5]),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "conda_history": {
            "path": str(history),
            "bytes": int(history.stat().st_size),
            "sha256": sha256(history),
        },
    }
    packages = [package_record(record) for record in freeze["packages"]]
    import torch
    require(torch.cuda.is_initialized() is False,
            "CUDA initialized before build inventory")

    binary_files = []
    for record in freeze["build"]["binary_files"]:
        path = Path(record["path"]).resolve()
        binary_files.append({
            "role": record["role"],
            "path": str(path),
            "bytes": int(path.stat().st_size),
            "sha256": sha256(path),
        })
    nvcc_lines = check_output(
        [freeze["build"]["nvcc_path"], "--version"]).splitlines()
    driver_line = check_output([
        "nvidia-smi", "--query-gpu=driver_version,name",
        "--format=csv,noheader"]).splitlines()
    require(len(driver_line) == 1 and ", " in driver_line[0],
            "M460R4 requires exactly one frozen A800")
    driver, gpu_name = driver_line[0].split(", ", 1)
    build = {
        "torch_cuda": torch.version.cuda,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "torch_debug": getattr(torch.version, "debug", None),
        "cudnn_version": torch.backends.cudnn.version(),
        "nvcc_path": freeze["build"]["nvcc_path"],
        "nvcc_build": nvcc_lines[-1],
        "nvidia_driver": driver,
        "gpu_name": gpu_name,
        "binary_files": binary_files,
    }
    runtime_imports = runtime_import_records(
        code_repo, freeze["runtime_imports"])
    require(torch.cuda.is_initialized() is False,
            "read-only import inventory initialized CUDA")
    result = {
        "schema": "m460r4_live_package_build_inventory_v1",
        "status": "PASS_M460R4_EXACT_PACKAGE_BUILD_AND_IMPORT_INVENTORY",
        "code_repo": str(code_repo),
        "python": python_record,
        "isolation": {
            "python_isolated": bool(sys.flags.isolated),
            "PYTHONNOUSERSITE": os.environ.get("PYTHONNOUSERSITE"),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "initial_sys_path": initial_sys_path,
        },
        "packages": packages,
        "build": build,
        "runtime_imports": runtime_imports,
        "final_sys_path": [str(item) for item in sys.path],
        "cuda_initialized": bool(torch.cuda.is_initialized()),
        "model_constructed": False,
        "checkpoint_read": False,
        "capture_launched": False,
        "training": False,
        "system_speedup": False,
        "headline": False,
    }
    validate_inventory(result, freeze)
    return result


def write_new_json(path, value):
    path = Path(path)
    require(not path.exists(), "refusing inventory overwrite: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-repo", required=True, type=Path)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    freeze = strict_json(args.freeze)
    require(freeze.get("schema") ==
            "m460r4_remote_package_build_environment_freeze_v1",
            "M460R4 environment freeze schema drift")
    result = collect_inventory(args.code_repo, freeze)
    write_new_json(args.output, result)
    print(json.dumps({
        "status": result["status"],
        "output": str(args.output),
        "packages": len(result["packages"]),
        "runtime_imports": len(result["runtime_imports"]),
        "cuda_initialized": result["cuda_initialized"],
        "capture_launched": False,
    }, sort_keys=True))


if __name__ == "__main__":
    raise SystemExit(main())
