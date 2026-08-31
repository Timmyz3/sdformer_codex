#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1118r3 different-author static launch hammer; never launches M1119r3 or EDA."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Callable


HW = Path(__file__).resolve().parents[2]
LAUNCHER = HW / "dc_handoff/scripts/run_m1112r3_c2_async_observation_authorized_launch_r1.py"
ENGINE = HW / "dc_handoff/scripts/m1112r3_c2_async_observation_authorized_engine_source_r1.py"
ENGINE_CONTRACT = HW / "contracts/m1112r3_c2_async_observation_shadow_source_contract_r1_20260830.json"
LAUNCH_RECEIPT = HW / "contracts/m1112r3_c2_async_observation_authorized_launch_receipt_r1_20260830.json"
SOURCE_CONTRACT = HW / "contracts/m1119r3_m1112r3_c2_zero_arg_launcher_source_contract_r1_20260830.json"
AUTHOR = HW / "reviews/m1119r3_m1112r3_c2_zero_arg_launcher_author_receipt_r1_20260830"
M1117R3 = HW / "reviews/m1117r3_m1112r3_c2_async_observation_engine_hammer_r1_20260830"
ENGINE_AUTHOR = HW / "reviews/m1112r3_c2_launch_chain_source_receipt_r1_20260830"
WRAPPER_R2 = HW / "rtl_m1112r2/m1112r2_c2_k1_async_observation_shadow_wrapper.sv"
WRAPPER = HW / "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
TB_R2 = HW / "dc_handoff/tb/tb_m1112r2_c2_k1_async_observation_shadow_case0_short.sv"
TB = HW / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"

ATTEMPT = HW / "results/.m1112r3_c2_async_observation_dc_mapped_vcs_attempt_consumed"
RESULT = HW / "results/m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830"
LOCK = Path("/tmp/m1112r3_c2_async_observation_eda.lock")
WORK_GLOB = ".m1112r3_c2_async_observation_dc_mapped_vcs_work.*"
FAILURE_GLOB = "m1112r3_c2_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*"

EXPECTED = {
    "launcher": "fda1778b12caa9f365ab0ec4227fc587bb1b6bb784039c0963ebec642ebbe158",
    "engine": "48616ebde16e07b132bbb2e686bd34a9f18270d0bc0693ab0ee956beb60f02be",
    "engine_contract": "92117a56e50a946d674c82ce9fc084548b480df139e0a4e5a9b4aed391292bef",
    "engine_contract_side": "cfe40a1d11bcdf77cd4ac33e149381b202c57cc8edc22cb1131559fba8e412fd",
    "engine_contract_outer": "ddda54a99c1638f39c828faf75775a7f5c0dae975ee26f7b251cbafa926906cf",
    "launch_receipt": "f24870b60b940d91f3280427995567931fe8e53e14db2bc58c68b61601b801f4",
    "launch_receipt_side": "6acb4b55754aadc0b3c1039e93476cfeea2b5ed5d58d92a635893514bfb225a0",
    "launch_receipt_outer": "3fbb26e397bd5425d93ea8d364558582f94c1cd7d24d0ed6cfd2f85b828f14c3",
    "source_contract": "c1437d77feb51072a30a697f2a36d099f7225b151f905f37314584941fbc5af0",
    "source_contract_side": "5ca822f0f0b880f7b8810230eda8b8c37b4890347a5f7287c61c4b5b38bf1cf0",
    "source_contract_outer": "bd0322fa5eb0b369304667f256532b3c6108b646aee8aec240b91786c5ceab45",
    "author_review": "42269a1ca6d949992306cfa74ca368fa811e7dd686753b15aa0abf926f729b8d",
    "author_manifest": "c12cb740901f906d1ad2c44ca1ffd8c8e56ff5f4a00a38b773cbd6eea9971aa5",
    "author_outer": "81c0a1be32c8f2d3bde05d66f0f847baf64575450ba8638085094363ce17bb0c",
    "m1117_review": "cc35e5a21f148f8da7f04ca71cd2385da46a2a37af5f0387fdd0c3f0b3d7e12c",
    "m1117_manifest": "b1f3b03f3ecf1f7a8fc2b38f5d50dec3a452cc30967cd00fe851d183b32fa1b3",
    "m1117_outer": "41b4950ac4e1a175379e4d0ae34fd5335e339e320f716cd5e2b073dc9aa00d82",
    "m1117_hammer": "d54d3d1c412c8c0bb4f89a49e3dc65cd0ec3a48f133b4cd561e96089e29181ea",
    "engine_author_review": "0cf65f1015e45ae70fb352bb86518d98784e3f436de4648bdf0d9c726efbf69b",
    "engine_author_manifest": "e30b75f496507f1d34ebf25fa6cdc9d5087adfc758bb5ce0b99e9a35cb8d3e69",
    "engine_author_outer": "7f9d0205b9ba2f53fd642b05b0cd4faf9aa3e8e5bf14a6047c23ac6fba3ea7ff",
    "wrapper_r2": "b1fccaa03b1e3c69205d440ed0e2af93beb0f6eca68e7f7291c67f56322e89f5",
    "wrapper": "95c31bc70a7617c6653eaca2f77a54388119f744b814dfc909c75edad1c39218",
    "tb_r2": "134c4a430d1daa257d73403612cdf41a2bb75369a4f16026413304d38e828d9c",
    "tb": "ff6bd371c3b1371c520b38680960ad0297a8c01eb92eb7b4a0f4d2e59fc861b6",
    "python": "9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115",
    "license": "fc6e1face2ac074043db2bef5c789d5ef747ef76333bc17e62d45389f48a3490",
    "docs359": "dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4",
}

checks: list[str] = []
attacks: dict[str, str] = {}


def require(value: bool, label: str) -> None:
    if not value:
        raise RuntimeError(label)
    checks.append(label)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise RuntimeError("duplicate JSON key: " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite JSON: " + token)))


def regular(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " regular no-symlink")
    require(sha(path) == expected, label + " SHA")


def verify_double(path: Path, identity: tuple[str, str, str], label: str) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0], label + " primary")
    regular(side, identity[1], label + " sidecar")
    regular(outer, identity[2], label + " outer")
    require(side.read_text(encoding="utf-8").split() ==
            [identity[0], path.relative_to(HW).as_posix()], label + " sidecar content")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], side.relative_to(HW).as_posix()], label + " outer content")


def verify_flat(directory: Path, identity: tuple[str, str, str], status_text: str,
                label: str) -> None:
    mode = directory.lstat().st_mode
    require(stat.S_ISDIR(mode) and not directory.is_symlink(), label + " direct directory")
    review = directory / "review.json"
    manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(review, identity[0], label + " review")
    regular(manifest, identity[1], label + " manifest")
    regular(outer, identity[2], label + " outer")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], "SHA256SUMS"], label + " outer content")
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1)
        require(len(fields) == 2 and len(fields[0]) == 64, label + " manifest row")
        relative = fields[1].lstrip("*")
        relpath = Path(relative)
        require(relative not in expected and relative == relpath.as_posix() and
                not relpath.is_absolute() and ".." not in relpath.parts,
                label + " safe unique member")
        expected[relative] = fields[0]
    actual: set[str] = set()
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        if relative in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        member_mode = member.lstat().st_mode
        require(not stat.S_ISLNK(member_mode), label + " no live symlink")
        if stat.S_ISREG(member_mode):
            actual.add(relative)
        else:
            require(stat.S_ISDIR(member_mode), label + " no special member")
    require(actual == set(expected), label + " no live extras/missing members")
    for relative, digest in expected.items():
        regular(directory / relative, digest, label + " member " + relative)
    require(strict_json(review).get("status") == status_text, label + " status")


def function_text(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                node.name == name)
    return ast.get_source_segment(source, node) or ast.unparse(node)


def analyze_launcher(source: str) -> None:
    ast.parse(source)
    validate = function_text(source, "validate_hardcoded_authorities")
    namespace = function_text(source, "namespace_resource_gate")
    collision = function_text(source, "collision_gate")
    child_env = function_text(source, "clean_child_environment")
    main = function_text(source, "main")
    require('len(sys.argv) == 1' in validate, "launcher zero-argument runtime gate")
    require('tuple(sys.version_info[:3]) == (3, 10, 18)' in validate,
            "launcher pinned Python version gate")
    require('set(os.environ) == ROOT_ENV_KEYS' in validate and
            'M1119r3 requires exact env -i root environment' in validate,
            "launcher exact env-i root gate")
    require('Path(sys.executable) == PYTHON' in validate,
            "launcher exact executable gate")
    require('verify_regular(PYTHON, PYTHON_SHA256)' in validate and
            'verify_regular(LICENSE_FILE, LICENSE_FILE_SHA256)' in validate,
            "launcher Python/license byte pins")
    require('verify_double(ENGINE_CONTRACT, ENGINE_CONTRACT_ID)' in validate and
            'verify_flat(' in validate and 'M1117R3' in validate,
            "launcher sealed upstream authority gates")
    require(all(token in namespace for token in (
        'not ATTEMPT.exists() and not ATTEMPT.is_symlink()',
        'not RESULT.exists() and not RESULT.is_symlink()',
        'not LOCK.exists() and not LOCK.is_symlink()',
        'glob(WORK_GLOB)', 'glob(FAILURE_GLOB)', 'collision_gate()',
        'MemAvailable', 'CommitLimit', 'Committed_AS',
        'info["MemAvailable"] >= MIN_MEM_AVAILABLE_KIB',
        'headroom >= MIN_COMMIT_HEADROOM_KIB')),
        "launcher freshness/collision/resource gates")
    require(all(('"' + name + '"') in source for name in
                ("vcs", "vcs1", "vlogan", "dc_shell", "dc_shell-t", "fm_shell",
                 "pt_shell", "simv")) and
            'uid = str(os.getuid())' in collision and
            '["/usr/bin/pgrep", "-u", uid, "-x", name]' in collision,
            "launcher same-UID exact EDA collision gate")
    require('caller environment value is consulted' in child_env and
            'os.environ' not in child_env and
            all(('"' + key + '"') in child_env for key in (
                "HOME", "LANG", "LC_ALL", "PATH", "TMPDIR", "PYTHONNOUSERSITE",
                "PYTHONDONTWRITEBYTECODE", "SNPSLMD_LICENSE_FILE", "LM_LICENSE_FILE")),
            "launcher constant caller-blind child environment")
    require('tempfile.mkdtemp(prefix="m1112r3_c2_home.", dir="/tmp")' in main and
            'private_home.chmod(0o700)' in main and
            'shutil.rmtree(private_home)' in main,
            "launcher private HOME 0700 and exact cleanup")
    require(main.index('validate_hardcoded_authorities(enforce_runtime=True)') <
            main.index('namespace_resource_gate()') < main.index('tempfile.mkdtemp') <
            main.index('subprocess.run') < main.index('shutil.rmtree(private_home)'),
            "launcher fail-closed call order")
    require(main.count('subprocess.run') == 1 and
            '[str(PYTHON), "-I", str(ENGINE), "--authorized-launch"]' in main and
            'cwd=str(HW)' in main and 'env=clean_child_environment(private_home)' in main and
            'close_fds=True' in main and 'check=False' in main and 'shell=True' not in main,
            "launcher exactly one pinned child and no shell")
    require(not any(isinstance(node, (ast.For, ast.While)) for node in
                    ast.walk(ast.parse(main))), "launcher main has no retry loop")


def analyze_engine(engine: str, contract: dict[str, Any], wrapper: str, tb: str) -> None:
    ast.parse(engine)
    flow = function_text(engine, "flow")
    require(flow.index('static_gate()') < flow.index('collision_gate()') <
            flow.index('resource_gate()') < flow.index('license_gate()') <
            flow.index('ATTEMPT.mkdir()') < flow.index('WORK.mkdir()') <
            flow.index('FRESH_DC_M1112R3') < flow.index('DC_TARGET') <
            flow.index('structural_reset_gate(netlist)') < flow.index('str(VCS)') <
            flow.index('str(simv)') < flow.index('write_json(WORK / "receipt.json"') <
            flow.index('os.rename(WORK, RESULT)'),
            "engine consumes attempt before one DC then mapped VCS")
    require(flow.count('DC_TARGET') == 1 and flow.count('run([\n            str(VCS)') == 1 and
            flow.count('run([str(simv)') == 1 and flow.count('ATTEMPT.mkdir()') == 1,
            "engine one-shot call multiplicity")
    require(not any(isinstance(node, ast.While) for node in ast.walk(ast.parse(flow))) and
            contract['frozen_stopped_namespaces']['m1112r3_maximum_attempts_after_all_hammers'] == 1 and
            contract['frozen_stopped_namespaces']['automatic_retry'] is False and
            contract['frozen_stopped_namespaces']['post_attempt_failure_quarantine_required'] is True,
            "engine no retry and post-attempt quarantine")
    observed = contract['preserved_observation_contract']
    require(observed == {
        'service_adapter_shadow_counters': 13,
        'mapped_shadow_bits_expected': 337,
        'observation_signals': 22,
        'atomic_unknown_bitmap_each_cycle': True,
        'first_x_cycle_and_bitmap': True,
        'continue_later_sampling': True,
        'fatal_only_after_128_cycles': True,
        'functional_feedback': False,
        'frozen_synchronous_debug_consumed': False,
        'preprocessed_rtl_identity': 'M1112 r1 observation RTL reused under a fresh M1112r3 module token; trust-gate-only repair',
    }, "engine exact 13/337/22 observation contract")
    fields = re.findall(r'logic\s+(?:\[(\d+):0\]\s+)?(shadow_\w+_q)\s*;', wrapper)
    require(len(fields) == 13 and
            sum((int(width) + 1) if width else 1 for width, _ in fields) == 337,
            "RTL independently recounts 13 shadows / 337 bits")
    instance = wrapper.split(") implementation (", 1)[1].split("));", 1)[0]
    require(instance.count("unused_frozen_debug_") == 13 and
            "obs_" not in instance and "shadow_" not in instance,
            "shadow observation has no functional feedback")
    predicates = re.findall(
        r'sample_unknown_bitmap\[(\d+)\]=\$isunknown\((obs_\w+)\);', tb)
    require(sorted(int(index) for index, _ in predicates) == list(range(22)) and
            len({signal for _, signal in predicates}) == 22,
            "TB independently recounts 22 atomic X predicates")
    require("unknown_union_bitmap|sample_unknown_bitmap" in tb and
            "if(window_cycle==128)" in tb and
            '$fatal(1,"M1112 fail-closed after complete 22-signal window sampling")' in tb,
            "TB atomic union and fatal only at 128-cycle close")
    require(contract['claim_boundary'] == {
        'source_only': True, 'mutation_selftest_only': True, 'eda_executed': False,
        'attempt_consumed': False, 'mapped_functionality': False,
        'paper_citable': False, 'activity_or_power': False, 'performance': False,
        'system_speedup': False, 'paper_ppa_ready': False,
    }, "engine all source claims false")


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise RuntimeError("mutation accepted: " + label)


def mutate_once(source: str, old: str, new: str) -> str:
    require(source.count(old) >= 1, "mutation anchor present: " + old[:48])
    return source.replace(old, new, 1)


def snapshot() -> dict[str, Any]:
    work = sorted(path.name for path in (HW / "results").glob(WORK_GLOB))
    failures = sorted(path.name for path in (HW / "results").glob(FAILURE_GLOB))
    return {
        "attempt_exists_or_symlink": ATTEMPT.exists() or ATTEMPT.is_symlink(),
        "result_exists_or_symlink": RESULT.exists() or RESULT.is_symlink(),
        "lock_exists_or_symlink": LOCK.exists() or LOCK.is_symlink(),
        "work_entries": work,
        "failure_entries": failures,
    }


def main() -> None:
    regular(LAUNCHER, EXPECTED['launcher'], "launcher")
    regular(ENGINE, EXPECTED['engine'], "engine")
    regular(WRAPPER_R2, EXPECTED['wrapper_r2'], "wrapper r2 include")
    regular(WRAPPER, EXPECTED['wrapper'], "wrapper implementation")
    regular(TB_R2, EXPECTED['tb_r2'], "TB r2 include")
    regular(TB, EXPECTED['tb'], "TB implementation")
    regular(DOCS359, EXPECTED['docs359'], "docs359")
    regular(Path('/opt/anaconda3/envs/pytorch310/bin/python3.10'), EXPECTED['python'], "Python")
    regular(Path('/opt/synopsys/Synopsys.dat'), EXPECTED['license'], "license file")
    verify_double(ENGINE_CONTRACT, (EXPECTED['engine_contract'], EXPECTED['engine_contract_side'],
                  EXPECTED['engine_contract_outer']), "engine contract")
    verify_double(LAUNCH_RECEIPT, (EXPECTED['launch_receipt'], EXPECTED['launch_receipt_side'],
                  EXPECTED['launch_receipt_outer']), "launch receipt")
    verify_double(SOURCE_CONTRACT, (EXPECTED['source_contract'], EXPECTED['source_contract_side'],
                  EXPECTED['source_contract_outer']), "source contract")
    verify_flat(AUTHOR, (EXPECTED['author_review'], EXPECTED['author_manifest'],
                EXPECTED['author_outer']),
                'PASS_M1119R3_M1112R3_ZERO_ARG_LAUNCHER_AUTHOR_SOURCE__M1118R3_FINAL_HAMMER_REQUIRED__NO_EDA',
                "M1119r3 author receipt")
    verify_flat(M1117R3, (EXPECTED['m1117_review'], EXPECTED['m1117_manifest'],
                EXPECTED['m1117_outer']),
                'PASS_M1117R3_M1112R3_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA',
                "M1117r3 GO")
    regular(M1117R3 / 'independent_engine_hammer.py', EXPECTED['m1117_hammer'],
            "M1117r3 hammer")
    verify_flat(ENGINE_AUTHOR, (EXPECTED['engine_author_review'], EXPECTED['engine_author_manifest'],
                EXPECTED['engine_author_outer']),
                'PASS_M1112R3_ACYCLIC_LAUNCH_CHAIN_SOURCE_AUTHOR_RECEIPT__M1117R3_REQUIRED__NO_EDA',
                "engine author receipt")

    launcher_source = LAUNCHER.read_text(encoding='utf-8')
    engine_source = ENGINE.read_text(encoding='utf-8')
    contract = strict_json(ENGINE_CONTRACT)
    receipt = strict_json(LAUNCH_RECEIPT)
    source_contract = strict_json(SOURCE_CONTRACT)
    author_review = strict_json(AUTHOR / 'review.json')
    m1117_review = strict_json(M1117R3 / 'review.json')
    analyze_launcher(launcher_source)
    analyze_engine(engine_source, contract, WRAPPER.read_text(encoding='utf-8'),
                   TB.read_text(encoding='utf-8'))

    require(receipt['launcher_sha256'] == EXPECTED['launcher'] and
            receipt['engine_sha256'] == EXPECTED['engine'] and
            receipt['m1117r3_outer_seal_file_sha256'] == EXPECTED['m1117_outer'] and
            receipt['maximum_attempts'] == 1 and receipt['automatic_retry'] is False,
            "launch receipt exact one-shot pins")
    require('m1118r3_outer_seal_file_sha256' not in receipt and
            EXPECTED['launch_receipt'] not in launcher_source and
            EXPECTED['launch_receipt_outer'] not in launcher_source and
            EXPECTED['source_contract_outer'] not in launcher_source and
            EXPECTED['author_outer'] not in launcher_source,
            "acyclic chain has no future-hammer/fixed-point back-edge")
    require(source_contract['acyclicity']['sha256_fixed_point_required'] is False and
            source_contract['acyclicity']['launch_receipt_contains_m1118r3_outer_seal_file_sha256'] is False and
            source_contract['acyclicity']['future_m1118r3_discovers_its_own_self_consistent_outer'] is True,
            "source contract acyclic discovery")
    require(author_review['identity']['launcher_sha256'] == EXPECTED['launcher'] and
            author_review['identity']['launch_receipt_outer_seal_file_sha256'] == EXPECTED['launch_receipt_outer'] and
            author_review['identity']['source_contract_outer_seal_file_sha256'] == EXPECTED['source_contract_outer'] and
            m1117_review['verdict'] == 'GO_DIFFERENT_AUTHOR_ZERO_ARGUMENT_LAUNCHER_AUTHORING_ONLY',
            "author and M1117r3 exact authority binding")
    require(source_contract['claim_boundary'] == {
        'source_only': True, 'attempt_consumed': False, 'mapped_functionality': False,
        'activity_or_power': False, 'performance': False, 'system_speedup': False,
        'paper_citable': False, 'paper_ppa_ready': False,
    }, "launcher source claims all false")

    before = snapshot()
    require(before == {
        'attempt_exists_or_symlink': False, 'result_exists_or_symlink': False,
        'lock_exists_or_symlink': False, 'work_entries': [], 'failure_entries': [],
    }, "production namespace fresh before static hammer")

    launcher_mutations = {
        'argv_relaxed': ('len(sys.argv) == 1', 'len(sys.argv) >= 1'),
        'root_env_relaxed': ('set(os.environ) == ROOT_ENV_KEYS', 'ROOT_ENV_KEYS <= set(os.environ)'),
        'child_isolation_removed': ('[str(PYTHON), "-I", str(ENGINE), "--authorized-launch"]',
                                    '[str(PYTHON), str(ENGINE), "--authorized-launch"]'),
        'private_home_mode_relaxed': ('private_home.chmod(0o700)', 'private_home.chmod(0o755)'),
        'private_home_cleanup_removed': ('shutil.rmtree(private_home)', 'pass # cleanup removed'),
        'attempt_gate_removed': ('not ATTEMPT.exists() and not ATTEMPT.is_symlink()', 'True'),
        'result_gate_removed': ('not RESULT.exists() and not RESULT.is_symlink()', 'True'),
        'lock_gate_removed': ('not LOCK.exists() and not LOCK.is_symlink()', 'True'),
        'failure_gate_removed': ('glob(FAILURE_GLOB)', 'glob("never-match")'),
        'same_uid_removed': ('["/usr/bin/pgrep", "-u", uid, "-x", name]',
                             '["/usr/bin/pgrep", "-x", name]'),
        'simv_collision_removed': ('"pt_shell", "simv",', '"pt_shell",'),
        'memory_gate_removed': ('info["MemAvailable"] >= MIN_MEM_AVAILABLE_KIB',
                                'info["MemAvailable"] >= 0'),
        'shell_injection': ('env=clean_child_environment(private_home),\n            close_fds=True, check=False,',
                            'env=clean_child_environment(private_home),\n            close_fds=True, check=False, shell=True,'),
    }
    for label, (old, new) in launcher_mutations.items():
        mutated = mutate_once(launcher_source, old, new)
        rejected(label, lambda mutated=mutated: analyze_launcher(mutated))

    rejected('engine_attempt_after_dc', lambda: analyze_engine(
        mutate_once(engine_source, 'ATTEMPT.mkdir(); attempted = True',
                    'attempted = True # ATTEMPT.mkdir moved after DC'),
        contract, WRAPPER.read_text(encoding='utf-8'), TB.read_text(encoding='utf-8')))
    changed_contract = json.loads(json.dumps(contract)); changed_contract['preserved_observation_contract']['mapped_shadow_bits_expected'] = 336
    rejected('shadow_census_336', lambda: analyze_engine(
        engine_source, changed_contract, WRAPPER.read_text(encoding='utf-8'),
        TB.read_text(encoding='utf-8')))
    changed_contract = json.loads(json.dumps(contract)); changed_contract['claim_boundary']['performance'] = True
    rejected('performance_claim_flip', lambda: analyze_engine(
        engine_source, changed_contract, WRAPPER.read_text(encoding='utf-8'),
        TB.read_text(encoding='utf-8')))
    rejected('shadow_counter_12', lambda: analyze_engine(
        engine_source, contract,
        mutate_once(WRAPPER.read_text(encoding='utf-8'),
                    'logic [31:0] shadow_adapter_bundle_response_count_q;', ''),
        TB.read_text(encoding='utf-8')))
    rejected('unknown_predicate_21', lambda: analyze_engine(
        engine_source, contract, WRAPPER.read_text(encoding='utf-8'),
        mutate_once(TB.read_text(encoding='utf-8'),
                    'sample_unknown_bitmap[21]=$isunknown(obs_adapter_bundle_response_count);',
                    'sample_unknown_bitmap[21]=1\'b0;')))
    changed_receipt = dict(receipt); changed_receipt['m1118r3_outer_seal_file_sha256'] = '0' * 64
    rejected('future_outer_fixed_point_injection', lambda: require(
        'm1118r3_outer_seal_file_sha256' not in changed_receipt,
        'receipt injected future outer'))

    with tempfile.TemporaryDirectory(prefix='m1118r3_seal_attack.', dir='/tmp') as raw:
        root = Path(raw)
        copied = root / 'author'; shutil.copytree(AUTHOR, copied)
        (copied / 'live.extra').write_text('attack\n', encoding='utf-8')
        rejected('live_extra', lambda: verify_flat(
            copied, (EXPECTED['author_review'], EXPECTED['author_manifest'], EXPECTED['author_outer']),
            'PASS_M1119R3_M1112R3_ZERO_ARG_LAUNCHER_AUTHOR_SOURCE__M1118R3_FINAL_HAMMER_REQUIRED__NO_EDA',
            'mutated author'))
        copied = root / 'm1117'; shutil.copytree(M1117R3, copied)
        manifest = copied / 'SHA256SUMS'; real = copied / 'SHA256SUMS.real'
        manifest.rename(real); manifest.symlink_to(real.name)
        rejected('live_manifest_symlink', lambda: verify_flat(
            copied, (EXPECTED['m1117_review'], EXPECTED['m1117_manifest'], EXPECTED['m1117_outer']),
            'PASS_M1117R3_M1112R3_ENGINE_HAMMER__AUTHOR_ZERO_ARG_LAUNCHER_ONLY__NO_EDA',
            'mutated M1117'))
        receipt_link = root / 'receipt.json'; receipt_link.symlink_to(LAUNCH_RECEIPT)
        rejected('launch_receipt_symlink', lambda: regular(
            receipt_link, EXPECTED['launch_receipt'], 'mutated launch receipt'))

    require(len(attacks) == 22, "all 22 independent mutations rejected")
    after = snapshot()
    require(after == before, "static hammer did not mutate production namespace")

    result = {
        'schema': 'm1118r3_m1112r3_c2_final_launch_hammer_mechanical_v1',
        'status': 'PASS_M1118R3_M1112R3_LAUNCH_HAMMER__GO_ONE_ATTEMPT',
        'checks_passed': len(checks),
        'mutations_rejected': len(attacks),
        'attacks': attacks,
        'identity': EXPECTED,
        'acyclic': {
            'sha256_fixed_point_required': False,
            'launch_receipt_contains_future_hammer_outer': False,
            'future_hammer_outer_self_consistently_discovered': True,
        },
        'namespace_before': before,
        'namespace_after': after,
        'launcher': {
            'arguments': 0, 'exact_env_i': True, 'caller_blind': True,
            'private_home_mode': '0700', 'same_uid_collision_gate': True,
            'minimum_mem_available_kib': 8388608,
            'minimum_commit_headroom_kib': 8388608,
        },
        'engine': {
            'attempt_consumed_before_dc': True, 'maximum_attempts': 1,
            'maximum_dc_invocations': 1, 'maximum_mapped_vcs_invocations': 1,
            'automatic_retry': False, 'shadow_counters': 13,
            'shadow_bits': 337, 'unknown_predicates': 22,
            'functional_feedback': False,
        },
        'execution': {
            'launcher_main_executed': False, 'engine_main_executed': False,
            'attempt_created': False, 'work_created': False,
            'failure_created': False, 'result_created': False,
            'dc_executed': False, 'mapped_vcs_executed': False,
        },
        'claims': {
            'mapped_functionality': False, 'activity_or_power': False,
            'performance': False, 'system_speedup': False,
            'paper_citable': False, 'paper_ppa_ready': False,
        },
        'docs359_sha256': sha(DOCS359),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
