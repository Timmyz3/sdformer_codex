#!/usr/bin/env python3
"""Read-only M1324 source forensic for the consumed M1320 attempt."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M1227 = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1227_motion_final_checkpoint_unified_hardware_r1.py")
M1174 = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1174_motion_checkpoint_parametric_unified_hardware.py")
M1249 = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1249_motion_final_checkpoint_unified_hardware_one_shot_release_r1.py")
M1319 = ROOT / (
    "neuron_experiments/H9_bipolar_self_attention/entrypoints/"
    "capture_m1319_motion_ep34_identity_compatibility_successor_r1.py")
M1313 = HW / (
    "contracts/m1313_motion_ep34_final_unified_capture_production_launch_"
    "r1_20260831.json")
M1182 = HW / (
    "contracts/m1182_m1180_motion_ep29_unified_capture_launch_release_r1_20260830.json")
M1210 = HW / (
    "contracts/m1210_m1208_motion_ep29_unified_capture_launch_release_r1_20260830.json")
FAILURE_ROOT = HW / "results/m1320_remote_failed_attempt_forensic_r1_20260831"
FAILURE_ARTIFACTS = {
    FAILURE_ROOT / "launcher.log":
        "432a8c131bfc11d38099f114b11f9e6e507c83fd9ff8cf1fdc68dfacdec182ab",
    FAILURE_ROOT / "attempt_consumed":
        "9be7c7f0db51d15310fcd43698b502e49fec5f5d7710b91ab0b345481fd6b737",
    FAILURE_ROOT / "temp.log":
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}

EXACT = {
    M1227: "11826d81c257bb0a14def4ab620be6c3971e4eea4175d6701e88de055140116b",
    M1174: "b476fad6885be23aa63a6b5d8e690fb3e213421074270cbb25e8ec00c202080a",
    M1249: "5fbcc4d287f3ffd3b1c9994efa24245e5e3828927cdac925c1a35d8a88a19219",
    M1319: "84a43559c408fcdb0f02a6cbbf76fc2d062d1749224b2302bffd79af609698f2",
    M1313: "eeb0a8380e51610652ec6cdf1c2bb58c22395c9d72608e98f6a88a18f5c6bbda",
    M1182: "46450015bcdb3b8c0a32ccd7aaba68a78abf923705a133147202283e7bc7220f",
    M1210: "5aeeaf9cab836f32e025f0c329ef1fe90caa4ee3acae691514f4793c1d143829",
}


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), str(path) + " must hold object")
    return value


def function_node(path: Path, name: str) -> ast.FunctionDef:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    require(len(nodes) == 1, "missing or duplicate function " + name)
    return nodes[0]


def subscript_chain(node: ast.AST):
    parts = []
    while isinstance(node, ast.Subscript):
        key = node.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            parts.append(key.value)
        else:
            return None
        node = node.value
    if isinstance(node, ast.Name):
        return tuple([node.id] + list(reversed(parts)))
    return None


def chains(path: Path, function: str, root: str) -> set[tuple[str, ...]]:
    result = set()
    for node in ast.walk(function_node(path, function)):
        if not isinstance(node, ast.Subscript):
            continue
        chain = subscript_chain(node)
        if chain and chain[0] == root:
            result.add(chain)
    return result


def main() -> int:
    checks = []
    for path, expected in EXACT.items():
        require(sha256(path) == expected, "frozen SHA drift: " + str(path))
    checks.append("seven frozen source/contracts exact")

    for path, expected in FAILURE_ARTIFACTS.items():
        require(path.is_file() and not path.is_symlink() and sha256(path) == expected,
                "failure artifact drift: " + str(path))
    traceback = (FAILURE_ROOT / "launcher.log").read_text(encoding="utf-8")
    require('line 778, in run_capture' in traceback and "KeyError: 'capture'" in traceback,
            "exact traceback boundary mismatch")
    require((FAILURE_ROOT / "attempt_consumed").read_text(encoding="ascii") ==
            "M1249_ATTEMPT_CONSUMED__AUTOMATIC_RETRY_FALSE\n",
            "attempt token mismatch")
    require((FAILURE_ROOT / "temp.log").stat().st_size == 0,
            "temporary log is not exact empty artifact")
    checks.append("three local failure artifacts exact, including consumed attempt")

    launch = object_json(M1313)
    require(set(launch) == {"schema", "status", "contract_path", "release_identity",
                            "inputs", "cohort", "one_shot", "output", "production_log"},
            "M1313 top-level shape drift")
    require("capture" not in launch, "M1313 unexpectedly contains capture")
    checks.append("M1313 exact top-level shape omits capture")

    m1227_contract = chains(M1227, "run_capture", "contract")
    require(("contract", "capture", "attention_windows_per_call") in m1227_contract,
            "M1227 capture access absent")
    require({chain[1] for chain in m1227_contract} ==
            {"contract_path", "capture", "cohort", "output"},
            "M1227 runtime contract dependency set drift")
    checks.append("M1227 needs contract_path/capture/cohort/output only")

    source = M1227.read_text(encoding="utf-8")
    require(source.count('contract["capture"]["attention_windows_per_call"]') == 1,
            "M1227 direct capture lookup count drift")
    checks.append("KeyError site uniquely identified at runtime projection")

    m1249_source = M1249.read_text(encoding="utf-8")
    require('"capture"' not in m1249_source[m1249_source.index("TOP_KEYS = {"):
                                             m1249_source.index("class M1249Error")],
            "M1249 TOP_KEYS unexpectedly includes capture")
    require("return contract, dict(binding" in M1319.read_text(encoding="utf-8"),
            "M1319 no longer returns original contract")
    checks.append("M1249 admits and M1319 returns no capture projection")

    values = [object_json(path)["r1_compatible_binding"]["capture"]
              for path in (M1182, M1210)]
    require(values == [{"attention_windows_per_call": 100}] * 2,
            "frozen runtime capture authorities disagree")
    checks.append("two frozen executable bindings independently pin windows=100")

    m1174_contract = chains(M1174, "run_capture", "contract")
    required_substrate = {
        ("contract", "contract_path"),
        ("contract", "inputs", "profile", "sha256"),
        ("contract", "inputs", "bit_writer", "sha256"),
        ("contract", "expected_topology", "module_counts"),
        ("contract", "capture", "attention_windows_per_call"),
        ("contract", "selected_identity"),
        ("contract", "output", "path"),
    }
    require(required_substrate <= m1174_contract, "M1174 runtime dependency drift")
    checks.append("M1227 substrate projection covers all M1174 run dependencies")

    proposed = {
        "contract_path": "hw_autoresearch_nts07/contracts/"
                         "m1324_motion_ep34_capture_runtime_projection_launch_r1_20260831.json",
        "capture": {"attention_windows_per_call": 100},
        "cohort": launch["cohort"],
        "output": {"path": "hw_autoresearch_nts07/results/"
                           "m1324_motion_ep34_unified_hardware_capture_s40_r1_20260831"},
    }
    require(proposed["cohort"] == launch["cohort"] and proposed["output"] != launch["output"],
            "projection must retain cohort and replace output")
    checks.append("minimal additive runtime projection is pure and disjoint")

    namespaces = {
        "result": proposed["output"]["path"],
        "attempt": "hw_autoresearch_nts07/results/"
                   ".m1324_motion_ep34_unified_hardware_capture_s40_r1_20260831."
                   "attempt_consumed",
        "log": "hw_autoresearch_nts07/results/"
               ".m1324_motion_ep34_unified_hardware_capture_s40_r1_20260831."
               "production.log",
    }
    old_names = {launch["output"]["path"], launch["one_shot"]["attempt_marker"],
                 launch["production_log"]["path"]}
    require(not old_names & set(namespaces.values()), "M1324 namespace intersects M1249")
    checks.append("new result/attempt/log are pairwise disjoint from consumed M1249")

    require(len(checks) == 10, "check count drift")
    print(json.dumps({
        "status": "PASS_M1324_PRE_GPU_FAILURE_ROOT_CAUSE_AND_MINIMUM_PROJECTION",
        "checks": checks,
        "root_cause": "M1313 omitted capture while M1227 directly required it",
        "minimum_field_delta": {"capture": {"attention_windows_per_call": 100}},
        "proposed_runtime_contract": proposed,
        "proposed_namespaces": namespaces,
        "execution": {"remote": False, "gpu": False, "capture": False},
    }, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
