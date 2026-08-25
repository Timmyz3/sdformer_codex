from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_m4_reducer_dse.py"


def load_module():
    spec = importlib.util.spec_from_file_location("m4_reducer", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_r4_knee_rule() -> None:
    module = load_module()
    variants = {}
    for slots, local, hybrid in ((2, 2.0, 2.1), (4, 3.2, 3.3), (8, 3.9, 4.0)):
        variants[slots] = {
            "variants": {
                "local": {"speedup_vs_same_width_dense_wall": local},
                "hybrid": {"speedup_vs_same_width_dense_wall": hybrid},
            }
        }
    selected, ratios = module.select_knee(variants)
    assert selected == 4
    assert ratios["local"]["r4_speedup_over_r2"] == 1.6
