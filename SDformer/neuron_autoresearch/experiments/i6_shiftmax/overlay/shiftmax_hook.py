"""Import hook: intercept Spiking_swin_transformer3D and inject shiftmax.

Registers a sys.meta_path finder that patches the attention module
at import time so shiftmax is inserted after position bias.
"""

import sys
import types
from pathlib import Path
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import SourceFileLoader


SHIFTMAX_INSERTION = """
        # --- I6 shiftmax (injected) ---
        from src.models.modules.sparse_ops.shiftmax import shiftmax
        attn = shiftmax(attn, dim=-1)
"""


class ShiftmaxLoader(SourceFileLoader):
    """Loads the original file but patches shiftmax into the source."""

    def get_source(self, fullname: str) -> str:
        source = super().get_source(fullname)

        # Patch occurrence 1 (line ~350 in baseline, swinv1)
        anchor = (
            "attn = attn + relative_position_bias.unsqueeze(0) #B_,n_head,N,N\n"
            "\n"
            "        if mask is not None:"
        )
        replacement = (
            "attn = attn + relative_position_bias.unsqueeze(0) #B_,n_head,N,N\n"
            + SHIFTMAX_INSERTION
            + "\n"
            "        if mask is not None:"
        )
        if anchor in source:
            source = source.replace(anchor, replacement, 1)

        # Patch occurrence 2 (line ~472 in baseline)
        anchor2 = (
            "attn = attn + relative_position_bias.unsqueeze(0)  # B_,n_head,N,N\n"
            "\n"
            "        if mask is not None:"
        )
        replacement2 = (
            "attn = attn + relative_position_bias.unsqueeze(0)  # B_,n_head,N,N\n"
            + SHIFTMAX_INSERTION
            + "\n"
            "        if mask is not None:"
        )
        if anchor2 in source:
            source = source.replace(anchor2, replacement2, 1)

        return source


class ShiftmaxFinder(MetaPathFinder):
    """Intercept import of Spiking_swin_transformer3D."""

    def __init__(self, original_path: Path, target_module: str):
        self.original_path = original_path
        self.target_module = target_module

    def find_spec(self, fullname, path, target=None):
        if fullname != self.target_module:
            return None
        # Use our loader instead of the default
        loader = ShiftmaxLoader(fullname, str(self.original_path))
        return loader.spec_from_loader(
            fullname, loader, origin=str(self.original_path)
        )


def install_shiftmax_hook(repo_root: Path) -> None:
    """Install import hook so Spiking_swin_transformer3D gets shiftmax."""
    target = str(
        repo_root
        / "third_party"
        / "SDformerFlow"
        / "models"
        / "STSwinNet_SNN"
        / "Spiking_swin_transformer3D.py"
    )
    module_name = "models.STSwinNet_SNN.Spiking_swin_transformer3D"
    finder = ShiftmaxFinder(Path(target), module_name)
    sys.meta_path.insert(0, finder)
