#!/usr/bin/python3.12
"""Compatibility entrypoint: ideafromai is now edited directly inside Git."""
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync", action="store_true", help="legacy option; no copy is performed")
    parser.add_argument("--source", type=Path, help="legacy option; external snapshots are retired")
    parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "ideafromai"
    print(f"ideafromai is directly Git-managed at {root}")
    print("No synchronization was performed or is needed. Edit this directory, then git add/commit.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
