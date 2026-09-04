"""Profile SDFormerFlow with the same sparse/voxel preprocessing used in training.

This wraps tools/profile_sops.py by patching one anchor after prepare_batch().
The output and CLI are intentionally compatible with tools/profile_sops.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


PROFILE_ANCHOR = """                chunk, label, mask = prepare_batch(chunk, label, mask, config, transform)
                pred_list = model(chunk.to(device))"""

PROFILE_PATCH = """                chunk, label, mask = prepare_batch(chunk, label, mask, config, transform)
                if config.get("sparsity", {}).get("enabled", False):
                    from sparse_preprocess import build_sparsity_pipeline
                    _sp_pipe = globals().get("_ar_sparse_profile_pipeline")
                    if _sp_pipe is None:
                        _sp_pipe = build_sparsity_pipeline(config)
                        globals()["_ar_sparse_profile_pipeline"] = _sp_pipe
                    if _sp_pipe is not None:
                        _sp_pipe.train(False)
                        chunk, _sp_stats = _sp_pipe(chunk)
                pred_list = model(chunk.to(device))"""


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main() -> None:
    repo_root = _repo_root()
    sparse_overlay = repo_root / "autoresearch_sparsity" / "overlay"
    profile_entry = repo_root / "tools" / "profile_sops.py"
    baseline_root = repo_root / "third_party" / "SDformerFlow"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(sparse_overlay))
    sys.argv = [str(profile_entry), *sys.argv[1:]]

    old_cwd = Path.cwd()
    try:
        os.chdir(repo_root)
        source = profile_entry.read_text()
        if PROFILE_ANCHOR not in source:
            raise RuntimeError(f"Could not patch {profile_entry}: missing profile anchor")
        source = source.replace(PROFILE_ANCHOR, PROFILE_PATCH, 1)
        code = compile(source, str(profile_entry), "exec")
        exec(code, {"__name__": "__main__", "__file__": str(profile_entry)})
    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    main()
