"""Read-only inherited source pins; no model or array import on module import."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
TARGET = "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.2.mlp"
OLD_TARGET = "sttmultires_unet.encoders.swin3d.layers.2.swin_blocks.0.mlp"
PREFIX_NAMES = tuple(
    "sttmultires_unet.encoders.swin3d.layers.{}.swin_blocks.{}.mlp".format(s, b)
    for s, b in ((0, 0), (0, 1), (2, 0), (2, 1), (2, 2)))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def pins():
    return json.loads((HERE / "source_pins.json").read_text())["sources"]


def source_path(key):
    pin = pins()[key]
    relative = Path(pin["path"])
    # Root's frozen deployment layout maps repository research/flow_story_DATE
    # to code/research_DATE. This is a directory-name mapping, never a search or
    # a fallback to unpinned files; every resolved byte stream is still hashed.
    if HERE.parent.name == "research_20261016":
        require(relative.parts[0] == "research" and relative.parts[1].startswith("flow_story_"),
                "Unsupported deployed source mapping")
        if "remote_path" in pin:
            path = Path(pin["remote_path"])
        elif key == "canonical":
            path = HERE.parents[2] / "setup/common_targets.json"
        elif key == "roles":
            path = HERE.parents[1] / "research_20261013/breadth/train_auxiliary_roles.csv"
        else:
            path = HERE.parents[1] / relative.parts[1].replace("flow_story_", "research_", 1)
            path = path.joinpath(*relative.parts[2:])
    else:
        path = ROOT / relative
    require(sha(path) == pin["sha256"], "Inherited source differs: " + key)
    return path


def verify_all():
    return {key: sha(source_path(key)) for key in pins()}


def load_source(key, module_name):
    path = source_path(key)
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
