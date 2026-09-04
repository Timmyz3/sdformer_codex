#!/opt/anaconda3/envs/pytorch310/bin/python3.10
"""M1122r4 source-author static/mutation check; no launcher, EDA, or attempt."""
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


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
ENGINE = HW / "dc_handoff/scripts/m1122r4_c2_dc_selector_async_observation_engine_source_r1.py"
OLD_ENGINE = HW / "dc_handoff/scripts/m1112r3_c2_async_observation_authorized_engine_source_r1.py"
CONTRACT = HW / "contracts/m1122r4_c2_dc_selector_async_observation_source_contract_r1_20260830.json"
OLD_CONTRACT = HW / "contracts/m1112r3_c2_async_observation_shadow_source_contract_r1_20260830.json"
M1121 = HW / "reviews/m1121_m1112r3_c2_dc_invocation_failure_audit_r1_20260830"
WRAPPER = HW / "rtl_m1112/m1112_c2_k1_async_observation_shadow_wrapper.sv"
TB = HW / "dc_handoff/tb/tb_m1112_c2_k1_async_observation_shadow_case0_short.sv"
DOCS359 = HW / "docs/359_DATE终局冻结_20260813.md"
ATTEMPT = HW / "results/.m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_attempt_consumed"
RESULT = HW / "results/m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830"
LOCK = Path("/tmp/m1122r4_c2_dc_selector_async_observation_eda.lock")

EXPECTED = {
    "engine": "f278052d251af0c2d150872391306c2f3922049ca04c7df2a0d9d3d074b55007",
    "old_engine": "48616ebde16e07b132bbb2e686bd34a9f18270d0bc0693ab0ee956beb60f02be",
    "contract": "cee4ddc66c244bf4e19e2ce193573b55bf4fd973c7c1bcd53d609d77a9b8cea3",
    "contract_side": "0a1ed1ad054b8a778c17c71eb9fdd82d5df943d77989280bd43660736795b617",
    "contract_outer": "373e6b86bdfdf94584f289f8c0fc1af1dc9a7ea19be656cba93159b3efb06987",
    "m1121_review": "910bdf5733a2287fa17ef6186f4814ef2c40b1216e4c6dc7378026b9a9cff525",
    "m1121_manifest": "ac977fb671794a7efffbadcd7cd9f23b6f1185dad15fa2e27d69ee69f1390dcf",
    "m1121_outer": "dc0135b61750134c37b6e3eba47350a0d9838c9ed0a07ca5ecab3bb93c3ff828",
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
                raise RuntimeError("duplicate key " + key)
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          RuntimeError("nonfinite " + token)))


def regular(path: Path, expected: str, label: str) -> None:
    mode = path.lstat().st_mode
    require(stat.S_ISREG(mode) and not path.is_symlink(), label + " regular")
    require(sha(path) == expected, label + " SHA")


def verify_double(path: Path, identity: tuple[str, str, str]) -> None:
    side = Path(str(path) + ".sha256")
    outer = Path(str(path) + ".sha256.seal.sha256")
    regular(path, identity[0], "contract primary")
    regular(side, identity[1], "contract side")
    regular(outer, identity[2], "contract outer")
    require(side.read_text(encoding="utf-8").split() ==
            [identity[0], path.relative_to(HW).as_posix()], "contract side content")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], side.relative_to(HW).as_posix()], "contract outer content")


def verify_flat(directory: Path, identity: tuple[str, str, str], status_text: str) -> None:
    mode = directory.lstat().st_mode
    require(stat.S_ISDIR(mode) and not directory.is_symlink(), "sealed direct directory")
    review = directory / "review.json"; manifest = directory / "SHA256SUMS"
    outer = directory / "SHA256SUMS.seal.sha256"
    regular(review, identity[0], "sealed review")
    regular(manifest, identity[1], "sealed manifest")
    regular(outer, identity[2], "sealed outer")
    require(outer.read_text(encoding="utf-8").split() ==
            [identity[1], "SHA256SUMS"], "sealed outer content")
    expected = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        fields = line.split(None, 1); require(len(fields) == 2, "manifest row")
        relative = fields[1].lstrip("*"); rel = Path(relative)
        require(relative not in expected and relative == rel.as_posix() and
                not rel.is_absolute() and ".." not in rel.parts, "safe manifest member")
        expected[relative] = fields[0]
    actual = set()
    for member in directory.rglob("*"):
        relative = member.relative_to(directory).as_posix()
        if relative in {"SHA256SUMS", "SHA256SUMS.seal.sha256"}:
            continue
        member_mode = member.lstat().st_mode
        require(not stat.S_ISLNK(member_mode), "sealed evidence has no symlink")
        if stat.S_ISREG(member_mode): actual.add(relative)
        else: require(stat.S_ISDIR(member_mode), "sealed evidence has no special member")
    require(actual == set(expected), "sealed evidence exact member set")
    for relative, digest in expected.items():
        regular(directory / relative, digest, "sealed member " + relative)
    require(strict_json(review)["status"] == status_text, "sealed status")


def function_text(source: str, name: str) -> str:
    tree = ast.parse(source)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and
                node.name == name)
    return ast.get_source_segment(source, node) or ast.unparse(node)


def definitions(path: Path) -> dict[str, Any]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    allowed = (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign,
               ast.ClassDef, ast.FunctionDef)
    namespace: dict[str, Any] = {"__file__": str(path), "__name__": "m1122r4_author_model"}
    module = ast.Module(body=[node for node in tree.body if isinstance(node, allowed)],
                        type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def analyze(source: str, contract: dict[str, Any]) -> None:
    ast.parse(source)
    selector = function_text(source, "verify_dc_selector")
    capture = function_text(source, "run_dc_with_selector_capture")
    static = function_text(source, "static_gate")
    flow = function_text(source, "flow")
    require("DC_TARGET" not in source, "old direct DC_TARGET absent")
    require('DC_SHELL = DC_INSTALL_ROOT / "bin/dc_shell"' in source and
            'DC_WRAPPER_TARGET = DC_INSTALL_ROOT / "bin/snps_shell"' in source and
            'DC_ACTUAL = DC_INSTALL_ROOT / "linux64/syn/bin/common_shell_exec"' in source,
            "exact three-part DC selector identity")
    require('stat.S_ISLNK(mode)' in selector and
            'os.readlink(DC_SHELL) != "snps_shell"' in selector and
            'DC_SHELL.resolve(strict=True) != DC_WRAPPER_TARGET' in selector and
            'verify_regular(DC_WRAPPER_TARGET, EXTERNAL_SHA256[DC_WRAPPER_TARGET])' in selector and
            'verify_regular(DC_ACTUAL, EXTERNAL_SHA256[DC_ACTUAL])' in selector,
            "selector lstat/raw/resolved/two-payload gates")
    require('subprocess.Popen(' in capture and capture.count('subprocess.Popen(') == 1 and
            '[str(DC_SHELL), "-f", str(DC_TCL)]' in capture and
            'str(DC_ACTUAL), "-shell", "dc_shell", "-r", str(DC_INSTALL_ROOT),\n        "-f", str(DC_TCL),' in capture and
            'identity["exe"] == str(DC_ACTUAL)' in capture and
            'identity["argv"] != expected_argv' in capture and
            'verify_regular(DC_ACTUAL, EXTERNAL_SHA256[DC_ACTUAL])' in capture and
            'common_shell_exec_sha256' in capture and 'process.wait(timeout=timeout)' in capture,
            "runtime exact common_shell_exec capture")
    require(flow.count('run_dc_with_selector_capture(') == 1 and
            'run([str(DC_SHELL)' not in flow and 'run([str(DC_ACTUAL)' not in flow and
            flow.index('ATTEMPT.mkdir()') < flow.index('run_dc_with_selector_capture(') <
            flow.index('structural_reset_gate(netlist)') < flow.index('str(VCS)') <
            flow.index('str(simv)'), "one-shot attempt then selector then mapped VCS")
    require('M1121_OUTER_SHA256 = "dc0135b61750134c37b6e3eba47350a0d9838c9ed0a07ca5ecab3bb93c3ff828"' in source and
            'M1121_OUTER_SHA256' in static and
            'verify_exact_flat(OLD_M1112R3_ATTEMPT' in static and
            'verify_exact_flat(OLD_M1112R3_FAILURE' in static and
            'old_failure["m1112_retry"] is not False' in static and
            'OLD_M1112R3_RESULT.exists()' in static and
            'glob(WORK_GLOB)' in static and 'glob(FAILURE_GLOB)' in static,
            "M1121 and old-NO_RETRY/new-fresh gates")
    require('PASS_M1112R3_RESET_PROVENANCE_MAPPED_SHORT_WINDOW' not in source and
            'DC_TARGET' not in source and 'M1117R3 =' not in source and 'M1118R3 =' not in source,
            "no stale executable authority/status")
    selector_contract = contract['dc_selector_contract']
    require(selector_contract['launch_argv'] == [
        '/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell', '-f',
        '/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl'],
        "contract exact dc_shell launch argv")
    require(selector_contract['dc_shell'] == {
        'path': '/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell',
        'exact_lstat_kind': 'symlink', 'raw_readlink': 'snps_shell',
        'resolved_path': '/opt/synopsys/syn/V-2023.12-SP3/bin/snps_shell',
        'resolved_target_sha256': '23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2'},
        "contract exact selector link")
    require(selector_contract['common_shell_exec']['exact_runtime_argv'] == [
        '/opt/synopsys/syn/V-2023.12-SP3/linux64/syn/bin/common_shell_exec',
        '-shell', 'dc_shell', '-r', '/opt/synopsys/syn/V-2023.12-SP3', '-f',
        '/home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_dc_m519_r8_setup_area_three_axis.tcl'] and
        selector_contract['common_shell_exec']['sha256'] ==
        'bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391',
        "contract exact backend identity/argv")
    require(contract['future_chain']['placeholder_or_hash_fixed_point_allowed'] is False and
            contract['future_chain']['launch_receipt_contains_future_m1125r4_outer'] is False and
            contract['future_chain']['launcher_exists_now'] is False and
            contract['future_chain']['launch_receipt_exists_now'] is False,
            "no future SHA cycle or launcher")
    require(contract['frozen_stopped_namespaces']['m1112r3_attempt_consumed'] is True and
            contract['frozen_stopped_namespaces']['m1112r3_retry_allowed'] is False and
            contract['frozen_stopped_namespaces']['m1112r3_namespace_reused'] is False and
            contract['frozen_stopped_namespaces']['m1122r4_maximum_attempts_after_all_hammers'] == 1 and
            contract['frozen_stopped_namespaces']['automatic_retry'] is False,
            "contract old stop/new one-shot")
    require(contract['preserved_observation_contract']['service_adapter_shadow_counters'] == 13 and
            contract['preserved_observation_contract']['mapped_shadow_bits_expected'] == 337 and
            contract['preserved_observation_contract']['observation_signals'] == 22 and
            contract['preserved_observation_contract']['fatal_only_after_128_cycles'] is True,
            "exact 13/337/22/128 observation contract")
    require(contract['claim_boundary'] == {
        'source_only': True, 'mutation_selftest_only': True, 'eda_executed': False,
        'attempt_consumed': False, 'mapped_functionality': False,
        'paper_citable': False, 'activity_or_power': False, 'performance': False,
        'system_speedup': False, 'paper_ppa_ready': False},
        "all claims false")


def rejected(label: str, action: Callable[[], Any]) -> None:
    try:
        action()
    except Exception as error:
        attacks[label] = type(error).__name__ + ": " + str(error)
        return
    raise RuntimeError("mutation accepted: " + label)


def replace_once(text: str, old: str, new: str) -> str:
    require(old in text, "mutation anchor " + old[:48])
    return text.replace(old, new, 1)


def main() -> None:
    regular(ENGINE, EXPECTED['engine'], "engine")
    regular(OLD_ENGINE, EXPECTED['old_engine'], "old engine")
    regular(DOCS359, EXPECTED['docs359'], "docs359")
    verify_double(CONTRACT, (EXPECTED['contract'], EXPECTED['contract_side'],
                  EXPECTED['contract_outer']))
    verify_flat(M1121, (EXPECTED['m1121_review'], EXPECTED['m1121_manifest'],
                EXPECTED['m1121_outer']),
                'PASS_M1121_FAILURE_AUDIT__M1112R3_DO_NOT_RETRY__ADDITIVE_R4_INVOCATION_SELECTOR_REPAIR_ONLY')
    source = ENGINE.read_text(encoding='utf-8')
    contract = strict_json(CONTRACT)
    old_contract = strict_json(OLD_CONTRACT)
    analyze(source, contract)
    require(contract['source_sha256'] == old_contract['source_sha256'] and
            contract['frozen_filelist_member_sha256'] ==
                old_contract['frozen_filelist_member_sha256'],
            "RTL/TB/filelist/SDC/Tcl exact rebinding")
    for name in ('fail', 'sha', 'load', 'write_json', 'verify_regular',
                 'safe_manifest_names', 'verify_double', 'verify_exact_flat',
                 'verify_double_self_consistent', 'verify_flat_self_consistent',
                 'verify_historical_m1080', 'parse_instances', 'is_allowed_inverter',
                 'structural_reset_gate_text', 'structural_reset_gate',
                 'resource_gate', 'license_gate', 'run', 'seal'):
        require(function_text(source, name) ==
                function_text(OLD_ENGINE.read_text(encoding='utf-8'), name),
                "preserved function " + name)
    wrapper = WRAPPER.read_text(encoding='utf-8')
    fields = re.findall(r'logic\s+(?:\[(\d+):0\]\s+)?(shadow_\w+_q)\s*;', wrapper)
    require(len(fields) == 13 and
            sum((int(width) + 1) if width else 1 for width, _ in fields) == 337,
            "13 shadows / 337 bits")
    tb = TB.read_text(encoding='utf-8')
    predicates = re.findall(r'sample_unknown_bitmap\[(\d+)\]=\$isunknown\((obs_\w+)\);', tb)
    require(sorted(int(index) for index, _ in predicates) == list(range(22)) and
            len({signal for _, signal in predicates}) == 22 and
            'if(window_cycle==128)' in tb, "22 X predicates / 128 window")
    require(not ATTEMPT.exists() and not ATTEMPT.is_symlink() and
            not RESULT.exists() and not RESULT.is_symlink() and
            not LOCK.exists() and not LOCK.is_symlink() and
            not any((HW / 'results').glob('.m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_work.*')) and
            not any((HW / 'results').glob('m1122r4_c2_dc_selector_async_observation_dc_mapped_vcs_r1_20260830.failed_or_incomplete.*')),
            "new production namespace absent")

    namespace = definitions(ENGINE)
    namespace['verify_dc_selector']()
    require(True, "live controlled dc_shell selector accepted read-only")
    with tempfile.TemporaryDirectory(prefix='m1122r4_selector_attack.', dir='/tmp') as raw:
        root = Path(raw); wrapper_path = root / 'snps_shell'; actual = root / 'common_shell_exec'
        wrapper_path.write_bytes(b'wrapper'); actual.write_bytes(b'actual')
        link = root / 'dc_shell'; link.symlink_to('snps_shell')
        saved = (namespace['DC_SHELL'], namespace['DC_WRAPPER_TARGET'], namespace['DC_ACTUAL'],
                 namespace['EXTERNAL_SHA256'])
        namespace['DC_SHELL'] = link; namespace['DC_WRAPPER_TARGET'] = wrapper_path
        namespace['DC_ACTUAL'] = actual
        namespace['EXTERNAL_SHA256'] = {wrapper_path: sha(wrapper_path), actual: sha(actual)}
        namespace['verify_dc_selector'](); require(True, "temporary legal selector accepted")
        link.unlink(); link.symlink_to('wrong')
        rejected('selector_wrong_raw_readlink', namespace['verify_dc_selector'])
        link.unlink(); shutil.copy2(wrapper_path, link)
        rejected('selector_regular_not_symlink', namespace['verify_dc_selector'])
        link.unlink(); link.symlink_to('snps_shell'); wrapper_path.write_bytes(b'drift')
        rejected('selector_wrapper_byte_drift', namespace['verify_dc_selector'])
        wrapper_path.write_bytes(b'wrapper'); actual_real = root / 'actual.real'
        actual.rename(actual_real); actual.symlink_to(actual_real.name)
        rejected('actual_backend_symlink', namespace['verify_dc_selector'])
        namespace['DC_SHELL'], namespace['DC_WRAPPER_TARGET'], namespace['DC_ACTUAL'], \
            namespace['EXTERNAL_SHA256'] = saved

        copied = root / 'm1121'; shutil.copytree(M1121, copied)
        (copied / 'live.extra').write_text('x\n', encoding='utf-8')
        rejected('sealed_live_extra', lambda: verify_flat(
            copied, (EXPECTED['m1121_review'], EXPECTED['m1121_manifest'], EXPECTED['m1121_outer']),
            'PASS_M1121_FAILURE_AUDIT__M1112R3_DO_NOT_RETRY__ADDITIVE_R4_INVOCATION_SELECTOR_REPAIR_ONLY'))
        copied = root / 'm1121_symlink'; shutil.copytree(M1121, copied)
        manifest = copied / 'SHA256SUMS'; real = copied / 'SHA256SUMS.real'
        manifest.rename(real); manifest.symlink_to(real.name)
        rejected('sealed_manifest_symlink', lambda: verify_flat(
            copied, (EXPECTED['m1121_review'], EXPECTED['m1121_manifest'], EXPECTED['m1121_outer']),
            'PASS_M1121_FAILURE_AUDIT__M1112R3_DO_NOT_RETRY__ADDITIVE_R4_INVOCATION_SELECTOR_REPAIR_ONLY'))

    source_mutations = {
        'direct_snps_shell_target': ('DC_SHELL = DC_INSTALL_ROOT / "bin/dc_shell"',
                                     'DC_SHELL = DC_INSTALL_ROOT / "bin/snps_shell"'),
        'raw_readlink_relaxed': ('os.readlink(DC_SHELL) != "snps_shell"',
                                 'os.readlink(DC_SHELL) != "anything"'),
        'resolved_target_removed': ('DC_SHELL.resolve(strict=True) != DC_WRAPPER_TARGET', 'False'),
        'wrapper_sha_removed': ('verify_regular(DC_WRAPPER_TARGET, EXTERNAL_SHA256[DC_WRAPPER_TARGET])', 'pass'),
        'actual_path_wrong': ('DC_ACTUAL = DC_INSTALL_ROOT / "linux64/syn/bin/common_shell_exec"',
                              'DC_ACTUAL = DC_INSTALL_ROOT / "bin/snps_shell"'),
        'actual_sha_removed': ('verify_regular(DC_ACTUAL, EXTERNAL_SHA256[DC_ACTUAL])', 'pass'),
        'launch_direct_actual': ('[str(DC_SHELL), "-f", str(DC_TCL)]',
                                 '[str(DC_ACTUAL), "-f", str(DC_TCL)]'),
        'backend_shell_selector_removed': ('str(DC_ACTUAL), "-shell", "dc_shell", "-r", str(DC_INSTALL_ROOT)',
                                           'str(DC_ACTUAL), "-r", str(DC_INSTALL_ROOT)'),
        'backend_root_wrong': ('"-r", str(DC_INSTALL_ROOT)', '"-r", "/tmp"'),
        'backend_tcl_wrong': (
            'str(DC_ACTUAL), "-shell", "dc_shell", "-r", str(DC_INSTALL_ROOT),\n        "-f", str(DC_TCL),',
            'str(DC_ACTUAL), "-shell", "dc_shell", "-r", str(DC_INSTALL_ROOT),\n        "-f", "/tmp/unpinned.tcl",'),
        'capture_bypassed': ('rc = run_dc_with_selector_capture(', 'rc = run('),
        'old_direct_identity_injected': ('DC_ACTUAL = DC_INSTALL_ROOT', 'DC_TARGET = DC_INSTALL_ROOT'),
        'm1121_pin_drift': ('M1121_OUTER_SHA256 = "dc0135b6', 'M1121_OUTER_SHA256 = "00000000'),
    }
    for label, (old, new) in source_mutations.items():
        rejected(label, lambda mutated=replace_once(source, old, new): analyze(mutated, contract))

    for label, mutate in (
        ('future_fixed_point', lambda value: value['future_chain'].__setitem__('placeholder_or_hash_fixed_point_allowed', True)),
        ('old_retry_enabled', lambda value: value['frozen_stopped_namespaces'].__setitem__('m1112r3_retry_allowed', True)),
        ('old_namespace_reused', lambda value: value['frozen_stopped_namespaces'].__setitem__('m1112r3_namespace_reused', True)),
        ('performance_claim', lambda value: value['claim_boundary'].__setitem__('performance', True)),
        ('shadow_bits_336', lambda value: value['preserved_observation_contract'].__setitem__('mapped_shadow_bits_expected', 336)),
        ('shadow_counters_12', lambda value: value['preserved_observation_contract'].__setitem__('service_adapter_shadow_counters', 12)),
        ('x_predicates_21', lambda value: value['preserved_observation_contract'].__setitem__('observation_signals', 21)),
    ):
        changed = json.loads(json.dumps(contract)); mutate(changed)
        rejected(label, lambda changed=changed: analyze(source, changed))
    require(len(attacks) == 26, "all 26 mutations rejected")

    result = {
        'schema': 'm1122r4_c2_dc_selector_engine_author_mechanical_v1',
        'status': 'PASS_M1122R4_DC_SELECTOR_ENGINE_SOURCE_AUTHOR_STATIC_AND_MUTATION_CHECK__NO_EDA',
        'checks_passed': len(checks), 'mutations_rejected': len(attacks),
        'attacks': attacks,
        'identity': EXPECTED,
        'selector': {
            'dc_shell_symlink': True, 'raw_readlink': 'snps_shell',
            'resolved_wrapper_sha256': '23a4101c711fdb4747a57e6f9524d4989eb76a2b3120f0fa9766f4b25ae8e6d2',
            'common_shell_exec_sha256': 'bf91e6abfb9e2523c3c4884844117c629bef9dd83e2959934029a409118aa391',
            'runtime_exact_argv_capture_required': True,
        },
        'preserved': {'shadow_counters': 13, 'shadow_bits': 337,
                      'unknown_predicates': 22, 'window_cycles': 128,
                      'mapped_vcs': True, 'claims_false': True},
        'execution': {'launcher': False, 'eda': False, 'attempt': False,
                      'result': False, 'production_namespace_created': False},
        'docs359_sha256': sha(DOCS359),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
