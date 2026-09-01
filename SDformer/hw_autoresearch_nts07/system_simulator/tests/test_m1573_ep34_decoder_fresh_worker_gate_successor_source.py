#!/usr/bin/env python3
from __future__ import print_function

import importlib.util
from pathlib import Path


SOURCE = (Path(__file__).resolve().parents[1] / "scripts" /
          "build_m1573_ep34_decoder_fresh_worker_gate_successor_source.py")


def load_source():
    spec = importlib.util.spec_from_file_location("m1573_source", str(SOURCE))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    module = load_source()
    checks = 0
    description = module.describe()
    assert description["claim_boundary"]["actual_execution"] is False
    assert description["claim_boundary"]["m1570_retry"] is False
    checks += 2
    result = module.synthetic_self_test()
    assert result["hardware_projection_exact"] is True
    assert result["rss"]["gate_calls"] > 0
    assert result["actual_execution"] is False
    checks += 3
    for config in module.CONFIGS:
        module.U.M.validate_config(config)
        checks += 1
    try:
        module.production_release()
    except module.M1573Error:
        checks += 1
    else:
        raise AssertionError("production release did not fail closed")
    print("PASS M1573 tests=%d actual_execution=false" % checks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
