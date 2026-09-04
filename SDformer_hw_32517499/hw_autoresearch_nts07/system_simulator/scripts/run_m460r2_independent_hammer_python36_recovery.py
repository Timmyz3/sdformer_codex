#!/usr/bin/env python3
"""Python-3.6 compatibility entry for the frozen M460R2 auditor."""

import importlib.util
from pathlib import Path


AUDITOR = (Path(__file__).resolve().parent /
           "independent_m460r2_m460_capture_hammer.py")


def main():
    spec = importlib.util.spec_from_file_location(
        "m460r2_frozen_auditor", str(AUDITOR))
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen M460R2 auditor")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original_run = module.subprocess.run

    def compatible_run(*args, **kwargs):
        if kwargs.pop("text", False):
            kwargs["universal_newlines"] = True
        return original_run(*args, **kwargs)

    module.subprocess.run = compatible_run
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
