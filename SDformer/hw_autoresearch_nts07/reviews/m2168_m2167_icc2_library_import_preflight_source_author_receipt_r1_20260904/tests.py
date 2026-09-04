#!/usr/bin/python3.12
"""Pure M2168 startup and source-order mutations; no EDA, license, GPU, or git."""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


selfcheck = load("m2168_selfcheck", HERE / "selfcheck.py")
source = selfcheck.RUNNER.read_text()
tests = 0


def expect_reject(label: str, fn) -> None:
    global tests
    try:
        fn()
    except Exception:
        tests += 1
        print(f"PASS_MUTATION_REJECTED {label}")
        return
    raise AssertionError(f"mutation survived: {label}")


selfcheck.validate_runner_source(source)
tests += 1


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise AssertionError(f"nonunique mutation anchor {old!r}: {text.count(old)}")
    return text.replace(old, new, 1)


# The exact M2166 failure: the absent cache parent is not repaired by plain mkdir.
expect_reject(
    "missing_cache_parent_plain_mkdir",
    lambda: selfcheck.validate_runner_source(replace_once(source, "mkdir -p -- \\\n", "mkdir -- \\\n")),
)

# A moved or duplicated external site must not survive the static source contract.
license_site = '"${LMUTIL}" lmstat -c "${LICENSE_SERVER}" -f ICCompilerII >"${WORK}/license_preflight.log" 2>&1'
layout_marker = "M2168_LAYOUT_GATE_PASS paths=7 strictly_below=true symlinks=0"
before, sep, after = source.partition(license_site)
assert sep
early_license = before.replace(layout_marker, license_site + "\n" + layout_marker, 1) + after
expect_reject("license_before_layout_gate", lambda: selfcheck.validate_runner_source(early_license))

contract_marker = "M2168_EXECUTION_CONTRACT_REREAD_PASS"
before, sep, after = source.partition(license_site)
assert sep
early_contract_license = before.replace(contract_marker, license_site + "\n" + contract_marker, 1) + after
expect_reject("license_before_contract_reread", lambda: selfcheck.validate_runner_source(early_contract_license))

expect_reject(
    "duplicate_license_site",
    lambda: selfcheck.validate_runner_source(source.replace(license_site, license_site + "\n" + license_site, 1)),
)
icc2_site = '"${ICC2}" -no_init -f "${TCL}"'
expect_reject(
    "duplicate_icc2_site",
    lambda: selfcheck.validate_runner_source(source.replace(icc2_site, icc2_site + "\n      " + icc2_site, 1)),
)
expect_reject(
    "icc2_wait_guard_removed",
    lambda: selfcheck.validate_runner_source(replace_once(
        source,
        '  while [[ ! -e "${LAUNCH_GATE}" ]]; do /usr/bin/sleep 0.01; done\n',
        "  : # forged no-wait launch\n",
    )),
)
ready = '[[ -e "${MONITOR_READY}" ]] || exit 5'
release = ': >"${LAUNCH_GATE}"'
before, sep, after = source.partition(release)
assert sep
early_release = before.replace(ready, release + "\n" + ready, 1) + after
expect_reject("gate_released_before_monitor_ready", lambda: selfcheck.validate_runner_source(early_release))


def make_layout(root: Path) -> tuple[Path, list[Path], Path, Path]:
    isolated = root / "isolated_cwd"
    listed = [
        isolated / "home", isolated / "tmp", isolated / "cache/xdg",
        isolated / "cache/library", isolated / "frame_output",
        isolated / "frame_logs", isolated / "reports",
    ]
    for path in listed:
        path.mkdir(parents=True, exist_ok=True)
    design = isolated / "m2153_disposable_design.nlib"
    frame = isolated / "frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm"
    return isolated, listed, design, frame


with tempfile.TemporaryDirectory(prefix="m2168_layout_ok_") as raw:
    args = make_layout(Path(raw))
    selfcheck.validate_isolated_layout_for_test(*args)
    tests += 1


def endpoint_symlink() -> None:
    with tempfile.TemporaryDirectory(prefix="m2168_endpoint_symlink_") as raw:
        root = Path(raw)
        isolated, listed, design, frame = make_layout(root)
        target = root / "outside_endpoint"
        target.mkdir()
        listed[2].rmdir()
        os.symlink(target, listed[2])
        selfcheck.validate_isolated_layout_for_test(isolated, listed, design, frame)


expect_reject("symlinked_isolation_child", endpoint_symlink)


def intermediate_symlink() -> None:
    with tempfile.TemporaryDirectory(prefix="m2168_parent_symlink_") as raw:
        root = Path(raw)
        isolated = root / "isolated_cwd"
        isolated.mkdir()
        outside = root / "outside_cache"
        (outside / "xdg").mkdir(parents=True)
        (outside / "library").mkdir()
        os.symlink(outside, isolated / "cache")
        for name in ("home", "tmp", "frame_output", "frame_logs", "reports"):
            (isolated / name).mkdir()
        listed = [
            isolated / "home", isolated / "tmp", isolated / "cache/xdg",
            isolated / "cache/library", isolated / "frame_output",
            isolated / "frame_logs", isolated / "reports",
        ]
        selfcheck.validate_isolated_layout_for_test(
            isolated, listed, isolated / "m2153_disposable_design.nlib",
            isolated / "frame_output/m2153_tcbn28hpcplusbwp35p140_frame.ndm",
        )


expect_reject("symlinked_intermediate_cache_parent", intermediate_symlink)


def precreated_design() -> None:
    with tempfile.TemporaryDirectory(prefix="m2168_design_stale_") as raw:
        isolated, listed, design, frame = make_layout(Path(raw))
        design.mkdir()
        selfcheck.validate_isolated_layout_for_test(isolated, listed, design, frame)


expect_reject("precreated_design_nlib", precreated_design)


def precreated_frame() -> None:
    with tempfile.TemporaryDirectory(prefix="m2168_frame_stale_") as raw:
        isolated, listed, design, frame = make_layout(Path(raw))
        frame.write_bytes(b"stale")
        selfcheck.validate_isolated_layout_for_test(isolated, listed, design, frame)


expect_reject("precreated_frame_ndm", precreated_frame)


def symlinked_output() -> None:
    with tempfile.TemporaryDirectory(prefix="m2168_output_symlink_") as raw:
        root = Path(raw)
        isolated, listed, design, frame = make_layout(root)
        target = root / "outside_output"
        target.write_bytes(b"x")
        os.symlink(target, frame)
        selfcheck.validate_isolated_layout_for_test(isolated, listed, design, frame)


expect_reject("symlinked_frame_output", symlinked_output)

print(f"PASS_M2168_MUTATION_TESTS tests={tests} eda_runs=0 license_queries=0")
