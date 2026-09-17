"""Supported regeneration: Python 3.12, NumPy-only; no Torch/GPU.

Historical GPU run: prepare_sources_gpu_legacy.py (Python 3.10).
Its provenance is retained explicitly in SOURCE_INPUTS.md and historical logs.
"""
from prepare_sources_numpy312 import main

if __name__ == '__main__':
    main()
