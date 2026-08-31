#!/opt/conda/envs/sdformerflow/bin/python
"""Source-only successor that binds Motion ep34 capture to the sealed live-105 list.

M1347 proved that M1343 used a synthetic-name digest and that its own contract
projection rejected the sealed test metadata.  This additive source keeps the
M1343 capture substrate, but makes the M1347 read-only CPU inventory the exact
ATLIF-name authority.  It owns no production CLI, attempt token, GPU work, model
forward, capture execution, or remote write.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Iterator


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
M1343_SOURCE = Path(__file__).with_name(
    "capture_m1343_motion_ep34_live105_inventory_successor_r1.py")
M1343_SOURCE_SHA256 = "2302cef461eebc3b9ba17a8425e7196770e6bc5dfdfb480e42a56bca437c8dd3"
M1347_ROOT = HW / (
    "reviews/m1347_m1343_motion_ep34_live105_inventory_successor_source_"
    "blind_hammer_r1_20260831")
M1347_REVIEW = M1347_ROOT / "review.json"
M1347_MANIFEST = M1347_ROOT / "SHA256SUMS"
M1347_OUTER = M1347_ROOT / "SHA256SUMS.seal.sha256"
M1347_INVENTORY = M1347_ROOT / "remote_cpu_inventory.json"
M1347_REVIEW_SHA256 = "ac7d6f50c38478e1efce8bc81a8d9da7308052064fd064872ec8ab05dbe34c94"
M1347_MANIFEST_SHA256 = "f3fd79448fb7944698caa24784eb11c4fbc82cb202d98baf1d06cb7f9a4d8ec4"
M1347_OUTER_SHA256 = "07ef3b96f878a61ec1bc7631ffc5685d845c14103afc85b25a3ed0a79b7dde11"
M1347_INVENTORY_SHA256 = "1dbc50271cb4b604a8961d58c6adaea72d74d025586ccebaeb6818279b6c9c84"

CHECKPOINT_SHA256 = "4bbaf7fc9fa48e6efd46898e40a05ca6f5c606d4497551394caf2885b394ca48"
CONFIG_SHA256 = "630e735c8fe1d643b524ecd82ecf69d514df548d36380144cef442541daa4d39"
PROFILE_SOURCE_SHA256 = "04f692c5bda6d1f88cdc932ce48f012767f22a2bb1ca161378971232f99c0684"
ATLIF_OVERLAY_SOURCE_SHA256 = "d9ee7e172f941a53ad1c031b0d5cdbbf7819f521c807e5bc54001a80c41b57f3"
EXPECTED_ATLIF_NAMES_SHA256 = "6a616f164625e3516bd2410f82d5f577c547c43a15b3bb2a5c4065add8a94cb7"
EXPECTED_ATLIF_COUNT = 105
EXPECTED_LIVE_MODULES = 259
EXPECTED_ORDERED_RECORDS = 10360
EXPECTED_RETAINED = 320
EXPECTED_ATTENTION = 480
EXPECTED_PAYLOAD = 640

SOURCE_CONTRACT = HW / (
    "contracts/m1349_motion_ep34_live105_inventory_successor_source_contract_r1_20260831.json")
TEST = HW / "tests/test_m1349_motion_ep34_live105_inventory_successor.py"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
DOCS359_SHA256 = "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4"

CANONICAL_RESULT = HW / "results/m1349_motion_ep34_live105_unified_hardware_capture_s40_r1_20260831"
CANONICAL_ATTEMPT = HW / "results/.m1349_motion_ep34_live105_unified_hardware_capture_s40_r1_20260831.attempt_consumed"
CANONICAL_LOG = HW / "results/.m1349_motion_ep34_live105_unified_hardware_capture_s40_r1_20260831.production.log"
SOURCE_SCHEMA = "m1349_motion_ep34_live105_inventory_successor_source_r1_v1"
SOURCE_STATUS = "SOURCE_ONLY__M1347_P0_CLOSED__DIFFERENT_AUTHOR_HAMMER_REQUIRED__NO_GPU"
PASS_TOKEN = "PASS_M1349_LIVE105_SOURCE_SELF_CHECK__NO_ATTEMPT_NO_GPU_NO_CAPTURE"


class M1349Error(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise M1349Error(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def regular_exact(path: Path, expected: str, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as error:
        raise M1349Error("missing " + label) from error
    require(stat.S_ISREG(mode) and not path.is_symlink(),
            label + " must be a regular non-symlink")
    require(sha256(path) == expected, label + " SHA mismatch")


def strict_json(path: Path) -> dict[str, Any]:
    def pairs(items):
        output = {}
        for key, value in items:
            require(key not in output, "duplicate JSON key: " + key)
            output[key] = value
        return output
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           M1349Error("nonfinite JSON token: " + token)))
    require(type(value) is dict, "JSON root must be object")
    return value


def _load_m1343():
    regular_exact(M1343_SOURCE, M1343_SOURCE_SHA256, "sealed M1343 source")
    spec = importlib.util.spec_from_file_location("m1349_sealed_m1343", M1343_SOURCE)
    require(spec is not None and spec.loader is not None, "cannot load sealed M1343")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M1343 = _load_m1343()
M1327 = M1343.M1327
M1249 = M1343.M1249
R1 = M1343.R1


def terminal_lf_digest(names: list[str]) -> str:
    return hashlib.sha256(("\n".join(names) + "\n").encode("utf-8")).hexdigest()


def validate_authority_payload(remote: Any) -> tuple[str, ...]:
    require(type(remote) is dict and
            remote.get("schema") == "m1347_remote_readonly_cpu_inventory_r1_v1" and
            remote.get("status") == "DIAGNOSTIC_ONLY__NO_GPU_NO_CAPTURE_NO_ATTEMPT",
            "M1347 inventory schema/status mismatch")
    require(remote.get("identity") == {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "profile_source_sha256": PROFILE_SOURCE_SHA256,
        "atlif_overlay_source_sha256": ATLIF_OVERLAY_SOURCE_SHA256,
    }, "checkpoint/config/profile/overlay binding mismatch")
    require(remote.get("load_audit") == {"missing": 0, "unexpected": 0},
            "checkpoint load audit mismatch")
    inventory = remote.get("inventory")
    require(type(inventory) is dict and inventory.get("atlif_count") == 105 and
            inventory.get("sn_v_count") == 0 and
            inventory.get("atlif_names_policy") ==
            "sorted model.named_modules names joined by LF with terminal LF",
            "live-105 inventory metadata mismatch")
    names = remote.get("atlif_names")
    require(type(names) is list and len(names) == EXPECTED_ATLIF_COUNT and
            all(type(name) is str and name for name in names),
            "sealed ATLIF list must contain 105 strings")
    require(names == sorted(names), "sealed ATLIF list is not sorted")
    require(len(set(names)) == EXPECTED_ATLIF_COUNT, "sealed ATLIF list is not unique")
    require(not any(".sn_v" in name for name in names), "sealed ATLIF list contains sn_v")
    digest = terminal_lf_digest(names)
    require(digest == EXPECTED_ATLIF_NAMES_SHA256 ==
            inventory.get("atlif_names_sha256"), "sealed ATLIF terminal-LF digest mismatch")
    require(inventory.get("first_name") == names[0] and
            inventory.get("last_name") == names[-1], "sealed ATLIF endpoints mismatch")
    repeat = remote.get("repeatability")
    require(type(repeat) is dict and type(repeat.get("rebuilds")) is int and
            repeat["rebuilds"] >= 2 and repeat.get("same_digest") is True,
            "at least two identical rebuilds required")
    execution = remote.get("execution")
    require(type(execution) is dict and execution.get("forward_executed") is False and
            execution.get("capture_executed") is False and
            execution.get("attempt_consumed") is False and
            execution.get("remote_files_written") is False,
            "M1347 inventory must remain read-only CPU diagnostic")
    return tuple(names)


def verify_m1347_failure() -> tuple[str, ...]:
    regular_exact(M1347_REVIEW, M1347_REVIEW_SHA256, "M1347 review")
    regular_exact(M1347_MANIFEST, M1347_MANIFEST_SHA256, "M1347 manifest")
    regular_exact(M1347_OUTER, M1347_OUTER_SHA256, "M1347 outer seal")
    regular_exact(M1347_INVENTORY, M1347_INVENTORY_SHA256, "M1347 inventory")
    require(M1347_OUTER.read_text(encoding="utf-8") ==
            M1347_MANIFEST_SHA256 + "  SHA256SUMS\n", "M1347 outer content mismatch")
    rows = {}
    for line in M1347_MANIFEST.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        require(name not in rows, "duplicate M1347 manifest member")
        rows[name] = digest
    require(rows.get(str(M1347_REVIEW.relative_to(ROOT))) == M1347_REVIEW_SHA256 and
            rows.get(str(M1347_INVENTORY.relative_to(ROOT))) == M1347_INVENTORY_SHA256,
            "M1347 authority members are not sealed")
    review = strict_json(M1347_REVIEW)
    require(review.get("verdict") == "FAIL_SOURCE__DO_NOT_AUTHORIZE_RELEASE" and
            review.get("authorization", {}).get("minimum_additive_successor_required") is True and
            review.get("authorization", {}).get("release_author") is False,
            "M1347 failure/authorization verdict mismatch")
    first = validate_authority_payload(strict_json(M1347_INVENTORY))
    second = validate_authority_payload(strict_json(M1347_INVENTORY))
    require(first == second, "two local authority rebuilds disagree")
    return first


EXPECTED_ATLIF_NAMES = verify_m1347_failure()


def expected_live105_inventory(static_inventory: dict[str, list[str]]) -> dict[str, list[str]]:
    require(type(static_inventory) is dict and
            set(static_inventory) == set(R1.EXPECTED_STATIC_COUNTS),
            "static inventory category set drift")
    require(all(type(value) is list for value in static_inventory.values()),
            "static inventory categories must be lists")
    counts = {key: len(value) for key, value in static_inventory.items()}
    require(counts == R1.EXPECTED_STATIC_COUNTS, "static inventory count drift")
    names = static_inventory["atlif"]
    require(names == list(EXPECTED_ATLIF_NAMES),
            "ATLIF inventory differs from sealed ordered live-105 authority")
    require(terminal_lf_digest(names) == EXPECTED_ATLIF_NAMES_SHA256,
            "ATLIF terminal-LF digest mismatch")
    result = {key: sorted(value) for key, value in static_inventory.items()}
    require(sum(len(value) for value in result.values()) == EXPECTED_LIVE_MODULES,
            "live inventory is not 259")
    return result


def validate_snapshot_population_live105(staging: Path) -> None:
    return M1343.validate_snapshot_population_live105(staging)


def final_validate_and_seal_live105(staging, writer_type, selected_identity) -> None:
    old_digest = M1343.EXPECTED_ATLIF_NAMES_SHA256
    try:
        M1343.EXPECTED_ATLIF_NAMES_SHA256 = EXPECTED_ATLIF_NAMES_SHA256
        M1343.final_validate_and_seal_live105(staging, writer_type, selected_identity)
    finally:
        M1343.EXPECTED_ATLIF_NAMES_SHA256 = old_digest


@contextlib.contextmanager
def patched_live105_capture_chain() -> Iterator[None]:
    originals = (R1.DEAD_SN_V, R1.EXPECTED_LIVE_COUNTS,
                 R1.expected_live_inventory, R1.validate_snapshot_population,
                 R1.final_validate_and_seal, M1249.CANONICAL_RESULT)
    require(originals[0] and len(originals[0]) == 12,
            "sealed predecessor dead inventory already modified")
    try:
        R1.DEAD_SN_V = tuple()
        R1.EXPECTED_LIVE_COUNTS = dict(R1.EXPECTED_STATIC_COUNTS)
        R1.expected_live_inventory = expected_live105_inventory
        R1.validate_snapshot_population = validate_snapshot_population_live105
        R1.final_validate_and_seal = final_validate_and_seal_live105
        M1249.CANONICAL_RESULT = CANONICAL_RESULT
        yield
    finally:
        (R1.DEAD_SN_V, R1.EXPECTED_LIVE_COUNTS,
         R1.expected_live_inventory, R1.validate_snapshot_population,
         R1.final_validate_and_seal, M1249.CANONICAL_RESULT) = originals


def build_runtime() -> tuple[dict[str, Any], dict[str, Any]]:
    old_runtime, binding = M1327.validate_identity_and_project()
    identity = binding.get("identity", {})
    require(identity.get("checkpoint_sha256") == CHECKPOINT_SHA256 and
            identity.get("config_sha256") == CONFIG_SHA256,
            "ep34 checkpoint/config identity drift")
    remote = strict_json(M1347_INVENTORY)
    validate_authority_payload(remote)
    runtime = copy.deepcopy(old_runtime)
    runtime["contract_path"] = str(SOURCE_CONTRACT.relative_to(ROOT))
    runtime["output"] = {"path": str(CANONICAL_RESULT.relative_to(ROOT))}
    require(set(runtime) == {"contract_path", "capture", "cohort", "output"} and
            runtime["capture"] == {"attention_windows_per_call": 100},
            "M1349 runtime projection drift")
    return runtime, binding


def validate_static_binding(policy: dict[str, Any] | None = None) -> None:
    """Source-only identity proof that does not require remote selection paths."""
    policy = strict_json(SOURCE_CONTRACT) if policy is None else policy
    require(policy.get("identity") == {
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "config_sha256": CONFIG_SHA256,
        "profile_source_sha256": PROFILE_SOURCE_SHA256,
        "atlif_overlay_source_sha256": ATLIF_OVERLAY_SOURCE_SHA256,
        "checkpoint_load": {"missing": 0, "unexpected": 0},
        "attention_mode": "h60",
    }, "static checkpoint/config/profile/overlay contract drift")
    validate_authority_payload(strict_json(M1347_INVENTORY))


def delegate_for_future_release(runtime: dict[str, Any], binding: dict[str, Any], substrate: Any) -> Path:
    expected, rebound = build_runtime()
    require(runtime == expected and binding == rebound, "under-lease M1349 binding drift")
    with patched_live105_capture_chain():
        output = M1249.run_capture(runtime, binding, substrate=substrate)
    require(Path(output) == CANONICAL_RESULT, "capture returned non-M1349 result")
    return Path(output)


def require_fresh_namespaces() -> None:
    paths = (CANONICAL_RESULT, CANONICAL_ATTEMPT, CANONICAL_LOG)
    require(len(set(paths)) == 3 and all("m1349_" in path.name for path in paths),
            "M1349 namespaces are not distinct/freshly named")
    require(all(not os.path.lexists(str(path)) for path in paths),
            "M1349 namespace is not fresh")


def validate_source_policy(policy: dict[str, Any] | None = None) -> dict[str, Any]:
    policy = strict_json(SOURCE_CONTRACT) if policy is None else policy
    require(type(policy) is dict and policy.get("schema") == SOURCE_SCHEMA and
            policy.get("status") == SOURCE_STATUS, "M1349 source policy mismatch")
    require(policy.get("source") == {
        "path": str(Path(__file__).resolve().relative_to(ROOT)),
        "sha256": sha256(Path(__file__).resolve())}, "M1349 source identity mismatch")
    test = policy.get("test")
    require(type(test) is dict and set(test) == {"path", "sha256", "passed", "failed"},
            "M1349 test schema must be exact")
    require(test == {"path": str(TEST.relative_to(ROOT)), "sha256": sha256(TEST),
                     "passed": 20, "failed": 0}, "M1349 test identity/result mismatch")
    require(policy.get("m1347_failure_authority") == {
        "path": str(M1347_ROOT.relative_to(ROOT)),
        "review_sha256": M1347_REVIEW_SHA256,
        "manifest_sha256": M1347_MANIFEST_SHA256,
        "outer_file_sha256": M1347_OUTER_SHA256,
        "remote_cpu_inventory_sha256": M1347_INVENTORY_SHA256,
    }, "M1347 exact authority contract mismatch")
    require(policy.get("production_authorized") is False,
            "source policy cannot authorize production")
    regular_exact(DOCS359, DOCS359_SHA256, "protected docs359")
    return policy


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-self-check", action="store_true")
    args = parser.parse_args()
    require(args.source_self_check, "M1349 is source-only")
    validate_source_policy()
    verify_m1347_failure()
    M1343.verify_m1329_failure()
    validate_static_binding()
    require_fresh_namespaces()
    require(len(R1.DEAD_SN_V) == 12 and R1.EXPECTED_LIVE_COUNTS["atlif"] == 93,
            "source self-check must leave predecessor globals unchanged")
    print(PASS_TOKEN)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
