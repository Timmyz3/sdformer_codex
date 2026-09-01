#!/usr/bin/env python3
"""Independent mutation-heavy, zero-EDA review of frozen M1740 source."""
from __future__ import print_function

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import tempfile


HERE = Path(__file__).resolve().parent
HW = HERE.parents[1]
RUNNER = HW / "dc_handoff/scripts/run_m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_one_shot.py"
TEST = HW / "system_simulator/tests/test_m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source.py"
CONTRACT = HW / "contracts/m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source_contract_r1_20260901.json"
AUTHOR = HW / "reviews/m1740_m1733_m1722_m1701_c1_readonly_formality_pt_salvage_source_author_receipt_r1_20260901"


class HammerFailure(RuntimeError):
    pass


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_runner():
    spec = importlib.util.spec_from_file_location("m1740_review_target", str(RUNNER))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def must_fail(callable_):
    try:
        callable_()
    except Exception:
        return 1
    raise HammerFailure("mutation survived")


def strict_kv(text, expected):
    rows = text.splitlines()
    got = []
    seen = set()
    for row in rows:
        if row.count("=") != 1:
            raise HammerFailure("scope syntax")
        key, value = row.split("=", 1)
        if key in seen:
            raise HammerFailure("scope duplicate")
        seen.add(key)
        got.append((key, value))
    if got != expected:
        raise HammerFailure("scope mapping/order drift")


def coverage_gate(text):
    patterns = (
        r"^setup\s+13860\s+13851 \(100%\)\s+0 \(  0%\)\s+9 \(  0%\)$",
        r"^hold\s+13860\s+13851 \(100%\)\s+0 \(  0%\)\s+9 \(  0%\)$",
        r"^out_setup\s+2680\s+2679 \(100%\)\s+0 \(  0%\)\s+1 \(  0%\)$",
        r"^out_hold\s+2680\s+2679 \(100%\)\s+0 \(  0%\)\s+1 \(  0%\)$",
        r"^min_pulse_width\s+78506\s+50526 \( 64%\)\s+0 \(  0%\)\s+27980 \( 36%\)$",
    )
    lines = text.splitlines()
    if any(sum(re.fullmatch(pattern, line) is not None for line in lines) != 1
           for pattern in patterns):
        raise HammerFailure("coverage row drift")
    if text.count("untested  no_paths") != 2:
        raise HammerFailure("output no_paths drift")
    if text.count("untested  no_clock") != 27980:
        raise HammerFailure("pulse no_clock drift")


def raw_gate(module, text, tcl_text):
    lines = text.splitlines()
    errors = [line for line in lines if line.startswith("Error:")]
    expected_errors = [
        "Error: Library Compiler executable path is not set. (PT-063)",
        'Error: can\'t read "::env(HOME)": no such variable',
    ]
    cmd013 = "\tUse error_info for more info. (CMD-013)"
    cmd081 = "\tstopped at line 993 due to error. (CMD-081)"
    summary = "Diagnostics summary: 2 errors, 5 warnings, 30 informationals"
    epilogue = "Thank you for using pt_shell!"
    if errors != expected_errors:
        raise HammerFailure("Error set/order drift")
    for line in expected_errors + [cmd013, cmd081, summary, epilogue]:
        if lines.count(line) != 1:
            raise HammerFailure("diagnostic cardinality drift")
    commands = module.logical_tcl(tcl_text)
    normalized = [re.sub(r"\s+", " ", line).strip() for line in lines]
    counts = {}
    for command in commands:
        counts[command] = counts.get(command, 0) + 1
    if len(commands) != 89 or any(normalized.count(command) != count
                                  for command, count in counts.items()):
        raise HammerFailure("Tcl cardinality drift")
    cursor = -1
    for command in commands:
        try:
            cursor = normalized.index(command, cursor + 1)
        except ValueError:
            raise HammerFailure("Tcl deletion/reorder")
    first = normalized.index(commands[0])
    if not all(lines.index(line) < first for line in expected_errors + [cmd013, cmd081]):
        raise HammerFailure("startup ordering drift")
    if normalized[cursor] != "quit" or lines.index(summary) <= cursor or lines.index(epilogue) <= cursor:
        raise HammerFailure("completion ordering drift")
    return commands


def source_gate(text):
    main = text[text.index("def main()") :]
    ordered = (
        "verify_authority()", "verify_predecessor_authority()",
        "verify_m1722_formality_reuse()", "verify_formality_payload(M1722_FORMALITY)",
        "verify_pt_evidence()", "namespaces_fresh()", "ATTEMPT.mkdir()",
        "STAGE.mkdir()", "shutil.copytree(PTSTA, STAGE / \"ptsta\")",
        "shutil.copytree(M1722_FORMALITY, STAGE / \"formality\")",
        "verify_formality_payload(STAGE / \"formality\")",
        "seal_dir(STAGE)", "publish_no_replace(STAGE, RESULT)",
    )
    cursor = 0
    for token in ordered:
        position = main.find(token, cursor)
        if position < 0:
            raise HammerFailure("source order/membership drift: " + token)
        cursor = position + len(token)
    forbidden = ("import subprocess", "import socket", "subprocess.run",
                 "subprocess.Popen", "os.system", "fm_shell -f", "pt_shell -f",
                 "dc_shell -f", "lmutil lmdiag", "requests.", "urllib", "socket.",
                 "SNPSLMD_LICENSE_FILE")
    if any(token in text for token in forbidden):
        raise HammerFailure("tool/license/network path")
    for required in ('"eda_runs": 0', '"license_queries": 0',
                     '"network_calls": 0', '"automatic_retry": False'):
        if required not in text:
            raise HammerFailure("zero-execution contract drift")
    return ordered, forbidden


def contract_gate(module, value, baseline):
    module.verify_contract_sources(value)
    if value.get("transitive_authority") != baseline.get("transitive_authority"):
        raise HammerFailure("transitive authority drift")
    if value.get("sealed_m1733_execution") != baseline.get("sealed_m1733_execution"):
        raise HammerFailure("sealed execution drift")


def main():
    module = load_runner()
    if sha(RUNNER) != "86c359bf098f07e1a577ba5b171f08792b1afc7541bcb956d3a5ccabeec64cf7":
        raise HammerFailure("runner identity")
    if sha(TEST) != "3901100e896cdc126d993d8c1c8cf802776ae0ba5a75a98c6b42fb1fea6203f2":
        raise HammerFailure("test identity")
    module.verify_file_seal(CONTRACT,
                            "7e926cfccbfdfc27eeddef55f4e9bdd4978c753334fde7363a16a9ba6650f79f",
                            "ef4d875fd90d399c2fa501146dcb2a61618a3e2c692bca87cee5d3be6d2e90e8",
                            "b34398ade7d526e5ed4e9ec7e718f61a5ce8354d6e61f9a93f52add816735d9e")
    module.verify_seal(AUTHOR,
                       "552253221804c2b3c5118c8a3f3b341635370e8fa8ea478012ccec67843a5020",
                       "a1d7e566266bcb657194cdc0b29cd0906a900b8cd8134d3cd0e207882cc1797c")
    contract = module.strict_json(CONTRACT)
    module.verify_contract_sources(contract)
    module.verify_predecessor_authority()
    proof = module.load_m1733().verify_m1722_formality_reuse()
    module.verify_formality_payload(module.M1722_FORMALITY)
    machine = module.verify_pt_evidence()
    exact_source = RUNNER.read_text()
    ordered, forbidden = source_gate(exact_source)

    expected_scope = [tuple(row.split("=", 1)) for row in
                      (module.PTSTA / "reports/runtime_scope.rpt").read_text().splitlines()]
    if len(expected_scope) != 14:
        raise HammerFailure("runtime key count")
    scope_text = (module.PTSTA / "reports/runtime_scope.rpt").read_text()
    strict_kv(scope_text, expected_scope)
    coverage_text = (module.PTSTA / "reports/analysis_coverage.rpt").read_text()
    coverage_gate(coverage_text)
    raw_text = (module.PTSTA / "pt.raw.log").read_text(errors="replace")
    tcl_text = module.PT_TCL.read_text()
    commands = raw_gate(module, raw_text, tcl_text)

    attacks = 0
    frozen = [(module.PTSTA / "PTSTA_INTERNAL_COMPLETE.txt", module.FIXED_SHA["marker"]),
              (module.PTSTA / "pt.raw.log", module.FIXED_SHA["raw_log"])]
    frozen += [(module.PTSTA / "reports" / name, module.FIXED_SHA[key])
               for name, key in module.REPORT_SHA.items()]
    frozen += [(module.M1722_FORMALITY / relative, module.FIXED_SHA[key])
               for relative, key in module.FORMALITY_SHA.items()]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        for index, (source, digest) in enumerate(frozen):
            data = source.read_bytes()
            for variant_index, changed in enumerate((data + b"X", b"X" + data,
                                                      data[:-1] if data else b"X")):
                path = root / (str(index) + "_" + str(variant_index))
                path.write_bytes(changed)
                attacks += must_fail(lambda p=path, d=digest: module.exact(p, d))

    # Exact ordered runtime mapping: delete, replace, duplicate, extra, and swaps.
    for index, row in enumerate(scope_text.splitlines()):
        rows = scope_text.splitlines()
        attacks += must_fail(lambda x="\n".join(rows[:index] + rows[index + 1:]) + "\n":
                            strict_kv(x, expected_scope))
        attacks += must_fail(lambda x=scope_text.replace(row, row + "_MUT", 1):
                            strict_kv(x, expected_scope))
    attacks += must_fail(lambda: strict_kv(scope_text + "extra=true\n", expected_scope))
    swapped = scope_text.splitlines(); swapped[0], swapped[1] = swapped[1], swapped[0]
    attacks += must_fail(lambda: strict_kv("\n".join(swapped) + "\n", expected_scope))

    # Coverage values, output rows/reasons, and cardinalities.
    for token in ("13860", "13851", "78506", "50526", "27980", "2680", "2679",
                  "out_setup", "out_hold", "untested  no_paths", "untested  no_clock"):
        changed = coverage_text.replace(token, "MUTATED", 1)
        attacks += must_fail(lambda x=changed: coverage_gate(x))

    # Every Tcl echo occurrence is attacked by deletion and duplication.
    normalized_lines = raw_text.splitlines()
    normalized = [re.sub(r"\s+", " ", line).strip() for line in normalized_lines]
    command_indices, cursor = [], -1
    for command in commands:
        cursor = normalized.index(command, cursor + 1)
        command_indices.append(cursor)
    for line_index in command_indices:
        changed_lines = list(normalized_lines); changed_lines[line_index] = ""
        attacks += must_fail(lambda x="\n".join(changed_lines) + "\n": raw_gate(module, x, tcl_text))
        changed_lines = list(normalized_lines)
        changed_lines.insert(line_index, changed_lines[line_index])
        attacks += must_fail(lambda x="\n".join(changed_lines) + "\n": raw_gate(module, x, tcl_text))
    # Every adjacent logical command pair is attacked by reordering.
    for left, right in zip(commands[:-1], commands[1:]):
        left_index = next(i for i, line in enumerate(normalized_lines)
                          if re.sub(r"\s+", " ", line).strip() == left)
        right_index = next(i for i, line in enumerate(normalized_lines[left_index + 1:], left_index + 1)
                           if re.sub(r"\s+", " ", line).strip() == right)
        changed_lines = list(normalized_lines)
        changed_lines[left_index], changed_lines[right_index] = changed_lines[right_index], changed_lines[left_index]
        attacks += must_fail(lambda x="\n".join(changed_lines) + "\n": raw_gate(module, x, tcl_text))
    for changed in (raw_text + "\nError: injected\n",
                    raw_text.replace("(PT-063)", "(PT-064)", 1),
                    raw_text.replace("(CMD-013)", "(CMD-999)", 1),
                    raw_text.replace("(CMD-081)", "(CMD-999)", 1),
                    raw_text.replace("Diagnostics summary: 2 errors, 5 warnings, 30 informationals",
                                     "Diagnostics summary: 0 errors, 5 warnings, 30 informationals", 1),
                    raw_text.replace("Thank you for using pt_shell!", "", 1)):
        attacks += must_fail(lambda x=changed: raw_gate(module, x, tcl_text))

    # Delete/reorder canonical evidence checks and inject execution paths.
    main_start = exact_source.index("def main()")
    source_prefix, source_main = exact_source[:main_start], exact_source[main_start:]
    for token in ordered:
        changed = source_prefix + source_main.replace(token, "MUTATED", 1)
        attacks += must_fail(lambda x=changed: source_gate(x))
    for token in forbidden:
        attacks += must_fail(lambda x=exact_source + "\n" + token: source_gate(x))

    # Contract transitive authority, evidence, execution budget and claim mutations.
    contract_mutations = []
    for key in ("m1733_runner_sha256", "m1734_review_sha256", "m1735_release_sha256"):
        item = copy.deepcopy(contract); item["transitive_authority"][key] = "0" * 64
        contract_mutations.append(item)
    for key in ("out_setup", "out_hold"):
        item = copy.deepcopy(contract); del item["admitted_evidence"]["coverage_disclosure"][key]
        contract_mutations.append(item)
    for key in module.SOURCE_CLAIMS:
        item = copy.deepcopy(contract); item["claim_boundary"][key] = True
        contract_mutations.append(item)
    for key in ("eda_runs", "license_queries", "network_calls"):
        item = copy.deepcopy(contract); item["future_execution"][key] = 1
        contract_mutations.append(item)
    for item in contract_mutations:
        attacks += must_fail(lambda value=item: contract_gate(module, value, contract))

    result = {
        "status": "PASS",
        "mutation_attacks_rejected": attacks,
        "full_pt_tcl_commands": len(commands),
        "runtime_scope_exact_ordered_keys": len(expected_scope),
        "formality_artifacts": len(module.FORMALITY_SHA),
        "formality_passing_compare_points": proof["passing_compare_points"],
        "setup_wns_ns": machine["setup_wns_ns"],
        "hold_wns_ns": machine["hold_wns_ns"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
