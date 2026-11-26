#!/usr/bin/env python3
"""Check injected execute-loop global names without importing Torch or the runner."""
import ast
import builtins
import dis
import hashlib
import json
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMPL = HERE.parent / 'implementation'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def global_reads(code):
    result = {ins.argval for ins in dis.get_instructions(code) if ins.opname in ('LOAD_GLOBAL', 'LOAD_NAME')}
    for item in code.co_consts:
        if isinstance(item, types.CodeType):
            result.update(global_reads(item))
    return result


def main():
    runner = IMPL / 'run_coverage_recovery.py'
    implementation = IMPL / 'execute_recovery.py'
    tree = ast.parse(runner.read_text())
    entry = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'execute')
    injection = next(node for node in entry.body if isinstance(node, ast.For))
    injected = set(ast.literal_eval(injection.iter))
    tree = ast.parse(implementation.read_text())
    module_names = set(vars(builtins)) | {'__name__', '__file__', '__doc__', '__builtins__'}
    for node in tree.body:
        if isinstance(node, ast.Import):
            module_names.update(alias.asname or alias.name.split('.')[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_names.update(alias.asname or alias.name for alias in node.names)
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            module_names.add(node.name)
    # Compile only; never execute/import the model, arrays, torch or runner.
    code = compile(implementation.read_text(), str(implementation), 'exec')
    function = next(item for item in code.co_consts if isinstance(item, types.CodeType) and item.co_name == 'execute')
    missing = sorted(global_reads(function) - module_names - injected)
    result = {'schema': 'flow_round13_execution_global_namespace_audit_v1',
              'status': 'FAIL_UNBOUND_GLOBAL' if missing else 'PASS',
              'auditor_sha256': sha(Path(__file__)), 'runner_sha256': sha(runner),
              'execute_sha256': sha(implementation), 'injected_names': sorted(injected),
              'unresolved_global_names': missing, 'GPU_or_arrays_imported': False,
              'scope': 'Static compiled-bytecode namespace audit only. Does not prove runtime math, library compatibility or GPU execution.'}
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(1)
    out = HERE / 'execution_namespace_audit_v1.json'
    if out.exists():
        raise ValueError('Refuse overwrite')
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
