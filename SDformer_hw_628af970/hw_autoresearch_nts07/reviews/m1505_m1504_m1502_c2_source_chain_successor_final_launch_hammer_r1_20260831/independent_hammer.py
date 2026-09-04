#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""Fresh different-author, no-EDA final launch hammer for M1502 C2."""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import inspect
import io
import json
import os
from pathlib import Path
import stat
import sys
import tempfile
import textwrap
import unittest


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
RUNNER = HW / (
    "dc_handoff/scripts/run_m1502_m1493_c2_source_chain_successor_"
    "one_shot.py")
CHECKER = HW / (
    "verif_m1502_c2_source_chain_successor/"
    "check_m1502_c2_source_chain_successor_source.py")
TESTS = HW / (
    "verif_m1502_c2_source_chain_successor/"
    "test_m1502_c2_source_chain_successor_source.py")
SOURCE_CONTRACT = HW / (
    "contracts/m1502_m1493_c2_source_chain_successor_source_contract_"
    "r1_20260831.json")
M1503 = HW / (
    "reviews/m1503_m1502_c2_source_chain_successor_source_blind_hammer_"
    "r1_20260831")
M1504 = HW / (
    "contracts/m1504_m1503_m1502_c2_source_chain_successor_launch_release_"
    "r1_20260831.json")
M1505 = HW / (
    "reviews/m1505_m1504_m1502_c2_source_chain_successor_final_launch_hammer_"
    "r1_20260831")
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
UCLI_KEY = ROOT / "ucli.key"

PINS = {
    "runner": "91fc6a8867a138098b660e4d450eda50f5bd1850f9127bc349c2a303aac36df1",
    "checker": "7535c11d878d0582c47b9247ef8be7b2b5e7104f5197ca031de8772ab24cfba1",
    "tests": "9fa4aa08e9033cd3d913bddc6932affc65377a5b1e8c504085306f32b8fe619a",
    "source_contract": "8ee9286fc59a536ef8e61d19b6111102933ea167eb40e910cf7fa3c17b7e0eb6",
    "source_contract_sidecar": "a113601324228f43470fc1951f910bfb47ff06df5e8b049e51289b33712efaf4",
    "source_contract_outer": "d157fd31abfe2eb48d01dd1d54fdd363059cec22fb05ba29596fa76c926abea1",
    "m1503_review": "8b10e352ec48bb50ae3fae00b5607295797316bd1e3b8b4e92136e51f4350f92",
    "m1503_manifest": "5d24b24d0787419e39faf30087a9a8c2ecee6cb36848f4c7595bf0d82eb0c9b1",
    "m1503_outer": "2697b629a3469c72dc36f584b5e46254a272f4f3c3def89c14d402018a873708",
    "m1504_release": "a41a0524172e1f795a0a62c636a62370d18c8b95dfcb5227bb7647e11f405161",
    "m1504_sidecar": "c5415a9426337c75fb9b1623086c70d40be35808b471148cc0e7d2108e154d67",
    "m1504_outer": "fab80b151d87ace11e19327c566340dde37b83c0df99a242b2ae10067dd1038e",
    "m1493_failure_payload": "43497b8701400b6c7c5d3f0cc29a2a41955a135fff4be6720968cbeb736cc5e7",
    "m1493_failure_manifest": "53e77670cd0f07ea457dc35f041e3885f7d73b304149c8d52e116fd06d6a5f88",
    "m1493_failure_outer": "8cb2e41374f9b827c118b949e1a37b66baeec5bef578d81ee68a0d95a90d4a7e",
    "m1494_review": "65435aca804c486d50d8332774c70e87083d66d5c2e7acc30485dc84ba458340",
    "m1494_manifest": "b2ff59fd22bd0bd6463ae9ac9aa31ee82d77099d40ea4890fd99600255b9811b",
    "m1494_outer": "329ed4435761eb7d00be969d43ac05221c837cc3f79cedefd03d557034c432f7",
    "m1495_release": "838ea0f3714167c43c6f4e40829c2d1a59d1b84ee7468758798c82f21114eb94",
    "m1496_review": "ef0af9fbf0ab094f40052de8fc552b7b97e2519dd5db88c6f3c2bf7505acb810",
    "m1496_manifest": "72da922a5b652bf07eecc2ecc75ade847c7950c1c3a056299cca613bc1a19049",
    "m1496_outer": "2c8a99c7a9f0d2f56d6b77583f09cdc9ade265ba55b47c721e0ff44680d98e79",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
    "ucli_key": "1107aa2b8d30b14e7e4f9237ff461fb058ae4e07c8a5bed30bef3ad3eb9c30ac",
}
CLAIMS = {key: False for key in (
    "functional_vcs_verified", "production_saif", "ptpx", "power",
    "energy", "performance", "system_speedup", "paper_ppa_ready",
    "headline")}
EXPECTED_AUTHORIZATION = {
    "launch": True, "campaigns": 1, "automatic_retry": False}
EXPECTED_BINDINGS = {
    "runner_sha256": PINS["runner"],
    "source_contract_sha256": PINS["source_contract"],
    "m1503_review_sha256": PINS["m1503_review"],
    "m1504_release_sha256": PINS["m1504_release"],
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("import spec: " + str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def strict_json(path: Path):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise RuntimeError("duplicate JSON key")
            value[key] = item
        return value
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("JSON nonregular")
    value = json.loads(path.read_text(), object_pairs_hook=pairs,
                       parse_constant=lambda token: (_ for _ in ()).throw(
                           RuntimeError("nonfinite JSON: " + token)))
    if type(value) is not dict:
        raise RuntimeError("JSON root")
    return value


def verify_sidecars(path: Path, file_sha: str, sidecar_sha: str,
                    outer_sha: str) -> None:
    sidecar = Path(str(path) + ".sha256")
    outer = Path(str(sidecar) + ".seal.sha256")
    if (sha(path) != file_sha or sha(sidecar) != sidecar_sha
            or sha(outer) != outer_sha):
        raise RuntimeError("sidecar identity")
    if sidecar.read_text().split() != [file_sha, path.name]:
        raise RuntimeError("sidecar content")
    if outer.read_text().split() != [sidecar_sha, sidecar.name]:
        raise RuntimeError("outer content")


def verify_seal(root: Path, review_sha: str, manifest_sha: str,
                outer_sha: str) -> dict:
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError("sealed root")
    manifest = root / "SHA256SUMS"
    outer = root / "SHA256SUMS.seal.sha256"
    if (sha(root / "review.json") != review_sha
            or sha(manifest) != manifest_sha or sha(outer) != outer_sha
            or outer.read_text().split() != [manifest_sha, "SHA256SUMS"]):
        raise RuntimeError("sealed identity")
    listed = set()
    for row in manifest.read_text().splitlines():
        digest, name = row.split(maxsplit=1)
        name = name.lstrip("*")
        rel = Path(name)
        member = root / rel
        if (name in listed or rel.is_absolute() or ".." in rel.parts
                or len(digest) != 64 or not member.is_file()
                or member.is_symlink()
                or not stat.S_ISREG(member.lstat().st_mode)
                or sha(member) != digest):
            raise RuntimeError("sealed member")
        listed.add(name)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*")
              if path.is_file() and path.name not in
              {"SHA256SUMS", "SHA256SUMS.seal.sha256"}}
    if listed != actual:
        raise RuntimeError("sealed population")
    return strict_json(root / "review.json")


def changed(value):
    if type(value) is bool:
        return not value
    if type(value) is int:
        return value + 1
    if type(value) is str:
        return value + "__M1505_MUTATION"
    if type(value) is list:
        return value + ["__M1505_MUTATION"]
    raise TypeError(type(value).__name__)


def walk_dicts(value, path=()):
    if isinstance(value, dict):
        yield path, value
        for key, item in value.items():
            yield from walk_dicts(item, path + (key,))


def walk_leaves(value, path=()):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from walk_leaves(item, path + (key,))
    else:
        yield path, value


def parent_at(value, path):
    for key in path:
        value = value[key]
    return value


def validate_release(value, exact):
    if value != exact:
        raise RuntimeError("release exact-set/value")
    if (value["status"] !=
            "RELEASE_M1502_C2_SOURCE_CHAIN_SUCCESSOR__FRESH_M1505_REQUIRED__NO_LAUNCH"
            or value["launch_now"] is not False
            or value["automatic_retry"] is not False):
        raise RuntimeError("release boundary")
    if value["identity"] != {
            "runner_path": RUNNER.relative_to(HW).as_posix(),
            "runner_sha256": PINS["runner"],
            "source_contract_path": SOURCE_CONTRACT.relative_to(HW).as_posix(),
            "source_contract_sha256": PINS["source_contract"],
            "source_hammer_path": M1503.relative_to(HW).as_posix(),
            "source_hammer_review_sha256": PINS["m1503_review"],
            "source_hammer_manifest_sha256": PINS["m1503_manifest"],
            "source_hammer_outer_file_sha256": PINS["m1503_outer"],
            "source_hammer_status":
                "PASS_M1503_M1502_C2_SOURCE_CHAIN_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE",
            "source_hammer_score": 100,
            "docs359_sha256": PINS["docs359"]}:
        raise RuntimeError("release identity")
    auth = value["authorization"]
    if auth != {
            "campaigns": 1, "axes": ["k8", "k1x8"],
            "workload_cases_per_axis": [0, 1, 2, 3, 4],
            "vcs_compiles": 2, "simv_runs": 10,
            "production_saif_files": 10, "ptpx_runs": 10,
            "all_ten_saif_before_first_ptpx": True,
            "attempt_before_first_eda": True,
            "partial_axis_publication": False,
            "automatic_retry": False, "effective_before_m1505": False}:
        raise RuntimeError("campaign authority")
    gate = value["final_hammer_gate"]
    if (gate["path"] != M1505.relative_to(HW).as_posix()
            or gate["present_at_release_authoring"] is not False
            or gate["fresh_different_author_required"] is not True
            or gate["required_status"] !=
            "PASS_M1505_AUTHORIZE_ONE_M1502_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH"
            or gate["required_authorization"] != EXPECTED_AUTHORIZATION
            or value["claim_boundary"] != CLAIMS):
        raise RuntimeError("final gate")


def validate_final(value):
    if value != {
            "status":
                "PASS_M1505_AUTHORIZE_ONE_M1502_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH",
            "authorization": EXPECTED_AUTHORIZATION,
            "bindings": EXPECTED_BINDINGS,
            "claim_boundary": CLAIMS}:
        raise RuntimeError("final exact authorization")


def resource_gate_is_read_only(function) -> bool:
    source = textwrap.dedent(inspect.getsource(function))
    if 'Path("/proc/meminfo").read_text()' not in source:
        return False
    forbidden = {
        "write_text", "write_bytes", "open", "mkdir", "unlink", "rename",
        "rmdir", "touch", "chmod", "chown", "run", "Popen",
        "system", "exec", "eval", "remove", "rmtree", "copy", "copy2",
    }
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in forbidden:
                return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"open", "exec", "eval", "compile"}:
                return False
    return True


def verify_execution_stack_exact(R) -> int:
    E = R.EXEC
    pairs = (
        (E.M1361_CHECKER, E.STATIC_SHA["m1361_checker"]),
        (E.M1361_TEST, E.STATIC_SHA["m1361_test"]),
        (E.M1361_CONTRACT, E.STATIC_SHA["m1361_contract"]),
        (E.SOURCE_CHECKER, E.STATIC_SHA["source_checker"]),
        (E.CELL_MODEL, E.STATIC_SHA["cell_model"]),
        (E.RESET_MEMORY_MODEL, E.STATIC_SHA["reset_memory_model"]),
        (E.CASE_TB, E.STATIC_SHA["case_tb"]),
        (E.ASSERTIONS, E.STATIC_SHA["assertions"]),
        (E.MAPPED_TB, E.STATIC_SHA["mapped_tb"]),
        (E.FILELIST["k8"], E.STATIC_SHA["filelist_k8"]),
        (E.FILELIST["k1x8"], E.STATIC_SHA["filelist_k1x8"]),
        (E.UCLI, E.STATIC_SHA["ucli"]),
        (E.PTPX_TCL, E.STATIC_SHA["ptpx_tcl"]),
        (E.VCS, E.STATIC_SHA["vcs"]),
        (E.PT, E.STATIC_SHA["pt"]),
        (E.LMUTIL, E.STATIC_SHA["lmutil"]),
        (E.PYTHON, E.STATIC_SHA["python"]),
        (E.LIB_DB, E.STATIC_SHA["lib_db"]),
        (E.DOCS359, E.STATIC_SHA["docs359"]),
    )
    for path, digest in pairs:
        E.exact(path, digest)
    E.verify_dir(E.M1361_AUTHOR, E.STATIC_SHA["m1361_review"],
                 E.STATIC_SHA["m1361_manifest"], E.STATIC_SHA["m1361_outer"])
    E.verify_dir(E.M1362, E.STATIC_SHA["m1362_review"],
                 E.STATIC_SHA["m1362_manifest"], E.STATIC_SHA["m1362_outer"])
    E.verify_dir(E.M1440, R.OLD.BASE.OLD_M1440_SHA["review"],
                 R.OLD.BASE.OLD_M1440_SHA["manifest"],
                 R.OLD.BASE.OLD_M1440_SHA["outer"])
    count = len(pairs)
    for axis in ("k8", "k1x8"):
        netlist = E.M872 / axis / "netlist" / f"{E.DESIGN}_mapped.v"
        sdc = E.M872 / axis / "netlist" / f"{E.DESIGN}_mapped.sdc"
        E.exact(netlist, E.STATIC_SHA[axis + "_netlist"])
        E.exact(sdc, E.STATIC_SHA[axis + "_sdc"])
        count += 2
    return count


def main() -> int:
    checks = []
    attacks = []
    def check(name, value, category):
        checks.append({"check": name, "category": category, "pass": bool(value)})
    def attack(name, thunk, category):
        try:
            thunk()
            caught = False
        except BaseException:
            caught = True
        attacks.append({"attack": name, "category": category,
                        "rejected": caught, "false_negative": not caught})

    for name, path in (("runner", RUNNER), ("checker", CHECKER),
                       ("tests", TESTS), ("source_contract", SOURCE_CONTRACT),
                       ("docs359", DOCS359), ("ucli_key", UCLI_KEY)):
        check(name + "_exact", sha(path) == PINS[name], "identity")
    verify_sidecars(SOURCE_CONTRACT, PINS["source_contract"],
                    PINS["source_contract_sidecar"], PINS["source_contract_outer"])
    verify_sidecars(M1504, PINS["m1504_release"], PINS["m1504_sidecar"],
                    PINS["m1504_outer"])
    check("source_and_release_sidecars", True, "identity")
    m1503 = verify_seal(M1503, PINS["m1503_review"],
                        PINS["m1503_manifest"], PINS["m1503_outer"])
    check("m1503_status", m1503.get("status") ==
          "PASS_M1503_M1502_C2_SOURCE_CHAIN_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE",
          "authority")
    check("m1503_score", m1503.get("score") == 100, "authority")
    if os.path.lexists(M1505):
        raise RuntimeError("M1505 not fresh")

    C = load("m1505_bound_m1502_checker", CHECKER)
    T = load("m1505_bound_m1502_tests", TESTS)
    R = C.R
    check("source_checker", C.check_source(False).get("status") ==
          "PASS_M1502_C2_SOURCE_CHAIN_SUCCESSOR_SOURCE__NO_EDA", "source")
    stream = io.StringIO()
    replay = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(T))
    check("source_tests_17", replay.testsRun == 17 and not replay.failures
          and not replay.errors, "source")

    R.verify_predecessor_failure()
    check("m1493_to_m1496_chain", True, "predecessor")
    failure_members = R.AUTH.verify_seal(
        R.OLD_FAILURE, PINS["m1493_failure_manifest"],
        PINS["m1493_failure_outer"])
    check("m1493_failure_seal", failure_members == {"failure.json"}
          and sha(R.OLD_FAILURE / "failure.json") ==
          PINS["m1493_failure_payload"], "predecessor")
    old_blind = R.AUTH.verify_authority(
        R.M1494, PINS["m1494_review"], PINS["m1494_manifest"],
        PINS["m1494_outer"])
    old_final = R.AUTH.verify_authority(
        R.M1496, PINS["m1496_review"], PINS["m1496_manifest"],
        PINS["m1496_outer"])
    check("m1494_authority", old_blind.get("status") ==
          "PASS_M1494_M1493_C2_LCA_SUCCESSOR_SOURCE_ZERO_FALSE_NEGATIVE",
          "predecessor")
    check("m1495_release_exact", sha(R.M1495) == PINS["m1495_release"],
          "predecessor")
    check("m1496_authority", old_final.get("status") ==
          "PASS_M1496_AUTHORIZE_ONE_M1493_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH",
          "predecessor")

    saved = {name: os.environ.pop(name, None) for name in R.ENV_PINS}
    callpath_error = None
    try:
        try:
            R.verify_frozen_execution_inputs()
        except BaseException as error:
            callpath_error = error
    finally:
        for name, value in saved.items():
            if value is not None:
                os.environ[name] = value
    check("corrected_callpath", type(callpath_error) is R.Failure and
          str(callpath_error) ==
          "M1502 authority absent: required exact SHA environment" and
          not isinstance(callpath_error, AttributeError), "source")
    exact_inputs = verify_execution_stack_exact(R)
    check("frozen_execution_inputs", exact_inputs == 23, "execution")
    R.namespaces_fresh()
    check("m1502_namespaces_fresh", True, "execution")
    check("resource_gate_read_only", resource_gate_is_read_only(
        R.EXEC.resource_gate), "execution")

    text = RUNNER.read_text()
    C.check_execution_text(text)
    check("campaign_counts", R.COUNTS == {
        "vcs_compiles": 2, "simv_runs": 10,
        "saif_files": 10, "ptpx_runs": 10}, "execution")
    check("compile_flags", R.COMPILE_PREFIX[-4:] == [
        "-debug_access+r", "-lca", "+vcs+lic+wait", "-Mdir=csrc"],
        "execution")
    all_saif = text.index(
        'if any(state[key] != COUNTS[key] for key in\n'
        '               ("vcs_compiles", "simv_runs", "saif_files")):')
    first_ptpx = text.index('state["phase"] = f"PTPX_{axis}_{case}"')
    check("all_saif_before_ptpx", all_saif < first_ptpx, "execution")
    check("runner_claims_false", R.CLAIMS == CLAIMS and
          not any(R.CLAIMS.values()), "claims")

    release = strict_json(M1504)
    frozen = copy.deepcopy(release)
    validate_release(release, frozen)
    check("release_semantics", True, "release")
    for path, leaf in walk_leaves(frozen):
        candidate = copy.deepcopy(frozen)
        parent_at(candidate, path[:-1])[path[-1]] = changed(leaf)
        attack("release_value_" + "_".join(path),
               lambda value=candidate: validate_release(value, frozen),
               "release_value")
    for path, mapping in list(walk_dicts(frozen)):
        for key in tuple(mapping):
            candidate = copy.deepcopy(frozen)
            del parent_at(candidate, path)[key]
            attack("release_delete_" + "_".join(path + (key,)),
                   lambda value=candidate: validate_release(value, frozen),
                   "release_key")
        candidate = copy.deepcopy(frozen)
        parent_at(candidate, path)["__M1505_EXTRA__"] = False
        attack("release_extra_" + ("_".join(path) or "root"),
               lambda value=candidate: validate_release(value, frozen),
               "release_key")

    final_candidate = {
        "status":
            "PASS_M1505_AUTHORIZE_ONE_M1502_C2_MAPPED_VCS_SAIF_PTPX_LAUNCH",
        "authorization": copy.deepcopy(EXPECTED_AUTHORIZATION),
        "bindings": copy.deepcopy(EXPECTED_BINDINGS),
        "claim_boundary": copy.deepcopy(CLAIMS),
    }
    validate_final(final_candidate)
    for path, leaf in walk_leaves(final_candidate):
        candidate = copy.deepcopy(final_candidate)
        parent_at(candidate, path[:-1])[path[-1]] = changed(leaf)
        attack("final_value_" + "_".join(path),
               lambda value=candidate: validate_final(value), "final")
    for path, mapping in list(walk_dicts(final_candidate)):
        for key in tuple(mapping):
            candidate = copy.deepcopy(final_candidate)
            del parent_at(candidate, path)[key]
            attack("final_delete_" + "_".join(path + (key,)),
                   lambda value=candidate: validate_final(value), "final")
        candidate = copy.deepcopy(final_candidate)
        parent_at(candidate, path)["__M1505_EXTRA__"] = False
        attack("final_extra_" + ("_".join(path) or "root"),
               lambda value=candidate: validate_final(value), "final")

    for name, tokens in {
            "release_sidecar_digest": ["0" * 64, M1504.name],
            "release_sidecar_path": [PINS["m1504_release"], "wrong.json"],
            "release_sidecar_extra": [PINS["m1504_release"], M1504.name,
                                      "extra"]}.items():
        expected = [PINS["m1504_release"], M1504.name]
        attack(name, lambda value=tokens: (_ for _ in ()).throw(RuntimeError())
               if value != expected else None, "sidecar")
    with tempfile.TemporaryDirectory() as temp_name:
        duplicate = Path(temp_name) / "duplicate.json"
        duplicate.write_text('{"status":1,"status":2}\n')
        attack("release_duplicate_json", lambda: strict_json(duplicate), "json")
        nonfinite = Path(temp_name) / "nonfinite.json"
        nonfinite.write_text('{"status":NaN}\n')
        attack("release_nonfinite_json", lambda: strict_json(nonfinite), "json")

    p0 = sum(item["false_negative"] for item in attacks)
    p1 = sum(not item["pass"] for item in checks)
    categories = {}
    for item in attacks:
        categories[item["category"]] = categories.get(item["category"], 0) + 1
    output = {
        "schema": "m1505_m1504_m1502_c2_final_launch_hammer_output_r1_v1",
        "status": "PASS_ZERO_FALSE_NEGATIVE" if p0 == 0 and p1 == 0
                  else "FAIL_DO_NOT_LAUNCH",
        "summary": {
            "checks_passed": sum(item["pass"] for item in checks),
            "checks_total": len(checks),
            "mutations_rejected": sum(item["rejected"] for item in attacks),
            "mutations_total": len(attacks),
            "mutation_categories": categories,
            "p0_count": p0, "p1_count": p1,
            "source_tests_passed": 17,
            "source_tests_total": 17,
            "frozen_inputs_exact": exact_inputs,
        },
        "checks": checks,
        "authorization_candidate": final_candidate,
        "corrected_callpath": {
            "terminal": "M1502 authority absent: required exact SHA environment",
            "attribute_error": False},
        "execution": {
            "license_query": 0, "vcs": 0, "simv": 0, "saif": 0,
            "pt": 0, "ptpx": 0, "eda": 0, "ssh": 0, "gpu": 0,
            "attempts_consumed": 0},
    }
    if p0 or p1:
        raise RuntimeError(json.dumps(output, sort_keys=True))
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
