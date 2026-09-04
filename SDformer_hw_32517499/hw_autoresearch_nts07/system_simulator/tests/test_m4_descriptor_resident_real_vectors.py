from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts/build_m4_descriptor_resident_real_vectors.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("m4_vectors", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_object_tag_is_stable_and_weight_geometry_sensitive() -> None:
    module = load_module()
    key = ("proj", "Linear", "0", "256", "1", "4")
    assert module.object_tag(key) == module.object_tag(key)
    assert module.object_tag(key) != module.object_tag((*key[:-1], "8"))


def test_stratified_selection_keeps_every_sample() -> None:
    module = load_module()
    values = []
    for sample in range(3):
        for index in range(5):
            values.append(((str(sample), f"op{index}"), [[index]]))
    selected = module.stratified_select(values, 6)
    assert len(selected) == 6
    assert {item[0][0] for item in selected} == {"0", "1", "2"}
