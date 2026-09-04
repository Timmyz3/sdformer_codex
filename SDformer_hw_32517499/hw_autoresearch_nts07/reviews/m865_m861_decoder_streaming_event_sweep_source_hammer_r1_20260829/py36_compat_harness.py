#!/usr/bin/env python3
"""Run the 14 M861 logical pytest cases under system Python 3.6.

The host Python 3.6 installation lacks both pytest and the standard
``dataclasses`` backport.  This review-local harness injects only the minimal
dataclass and parametrization interfaces needed by the byte-frozen test file;
it neither edits source nor claims that the unmodified host environment can
import M861 natively.  Production is pinned to Python 3.10.
"""

import copy
import importlib.util
from pathlib import Path
import sys
import types


class FrozenInstanceError(AttributeError):
    pass


def _field_rows(cls):
    rows = []
    seen = set()
    for base in reversed(cls.__mro__[1:]):
        for row in getattr(base, "__dataclass_fields__", ()):  # reviewer shim
            if row[0] not in seen:
                rows.append(row)
                seen.add(row[0])
    annotations = getattr(cls, "__annotations__", {})
    for name in annotations:
        default = getattr(cls, name, _MISSING)
        if name in seen:
            rows = [row for row in rows if row[0] != name]
        rows.append((name, default))
        seen.add(name)
    return rows


_MISSING = object()


def dataclass(_cls=None, **options):
    frozen = bool(options.get("frozen", False))

    def wrap(cls):
        rows = _field_rows(cls)
        cls.__dataclass_fields__ = tuple(rows)

        def init(self, *args, **kwargs):
            if len(args) > len(rows):
                raise TypeError("too many positional arguments")
            values = {}
            for (name, _default), value in zip(rows, args):
                values[name] = value
            for name, default in rows[len(args):]:
                if name in kwargs:
                    values[name] = kwargs.pop(name)
                elif default is not _MISSING:
                    values[name] = copy.deepcopy(default)
                else:
                    raise TypeError("missing required argument: " + name)
            if kwargs:
                raise TypeError("unexpected arguments: " + repr(sorted(kwargs)))
            for name, _default in rows:
                object.__setattr__(self, name, values[name])

        def equal(self, other):
            return type(self) is type(other) and all(
                getattr(self, name) == getattr(other, name)
                for name, _default in rows)

        def represent(self):
            return "{}({})".format(
                type(self).__name__,
                ", ".join("{}={!r}".format(name, getattr(self, name))
                          for name, _default in rows))

        cls.__init__ = init
        cls.__eq__ = equal
        cls.__repr__ = represent
        if frozen:
            cls.__hash__ = lambda self: hash(tuple(
                getattr(self, name) for name, _default in rows))
        return cls

    if _cls is None:
        return wrap
    return wrap(_cls)


def asdict(value):
    if hasattr(value, "__dataclass_fields__"):
        return {name: asdict(getattr(value, name))
                for name, _default in value.__dataclass_fields__}
    if isinstance(value, dict):
        return type(value)((asdict(k), asdict(v)) for k, v in value.items())
    if isinstance(value, tuple):
        return tuple(asdict(item) for item in value)
    if isinstance(value, list):
        return [asdict(item) for item in value]
    return copy.deepcopy(value)


dataclasses_module = types.ModuleType("dataclasses")
dataclasses_module.dataclass = dataclass
dataclasses_module.asdict = asdict
dataclasses_module.FrozenInstanceError = FrozenInstanceError
sys.modules["dataclasses"] = dataclasses_module


class _Mark:
    def parametrize(self, _names, values):
        def decorate(function):
            function.__m865_parametrize__ = tuple(values)
            return function
        return decorate


pytest_module = types.ModuleType("pytest")
pytest_module.mark = _Mark()
sys.modules["pytest"] = pytest_module


class MonkeyPatch:
    def __init__(self):
        self.undo_rows = []

    def setattr(self, target, name, value):
        self.undo_rows.append((target, name, getattr(target, name)))
        setattr(target, name, value)

    def undo(self):
        for target, name, value in reversed(self.undo_rows):
            setattr(target, name, value)


ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "system_simulator/tests/test_m861_decoder_streaming_event_sweep.py"
spec = importlib.util.spec_from_file_location("m865_exact_m861_tests", str(TEST))
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


completed = []


def run(name, *args):
    getattr(module, name)(*args)
    completed.append(name)


run("test_manual_e_d_i_r_priority_exposes_all_classes")
run("test_interval_union_handles_touch_overlap_and_out_of_order_exactly")
for seed, count in ((861, 1), (862, 64), (863, 512), (864, 2048)):
    run("test_random_dag_exact_old_new_all_frozen_fields", seed, count)
run("test_1rw_1r1w_outstanding_and_same_cycle_return_slot_reuse")
run("test_outstanding_one_reuses_slot_at_exact_return_cycle")
run("test_streaming_summary_retains_no_expanded_or_compressed_rows")
run("test_compressed_count_matches_reference_even_when_endpoints_change")
blocked = []
try:
    run("test_bounded_real_prefix_exact_miter_and_streaming_summary")
except ModuleNotFoundError as error:
    # The system 3.6 environment has no torch, which the frozen real-prefix
    # oracle imports.  Record the environmental block rather than replacing
    # the oracle with a mock and falsely calling the case passed.
    blocked.append(("bounded_real_prefix", str(error)))
try:
    run("test_production_flag_is_fail_closed")
except TypeError as error:
    # ``subprocess.run(text=...)`` is itself a Python >=3.7 test dependency.
    blocked.append(("production_flag_test", str(error)))
patch = MonkeyPatch()
try:
    run("test_full_first_row_and_full_population_are_not_called", patch)
finally:
    patch.undo()
run("test_docs359_and_m857_failure_authority_remain_pinned")

assert len(completed) == 12
assert blocked[0] == ("bounded_real_prefix", "No module named 'torch'")
assert blocked[1][0] == "production_flag_test"
print("FAIL_M865_PY36_NATIVE_ENVIRONMENT passed=12 blocked=2 "
      "blocked_cases=bounded_real_prefix,production_flag_test "
      "reasons=missing_dataclasses_pytest_torch,subprocess_text_keyword")
