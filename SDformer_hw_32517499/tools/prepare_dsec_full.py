from __future__ import annotations

import argparse
from pathlib import Path

from prepare_dsec_single_sequence import prepare_sequence, resolve_device, write_split_csvs


DEFAULT_FLOW_SEQUENCES = [
    "zurich_city_09_a",
    "zurich_city_07_a",
    "zurich_city_02_c",
    "zurich_city_11_b",
    "thun_00_a",
    "zurich_city_02_d",
    "zurich_city_11_c",
    "zurich_city_03_a",
    "zurich_city_10_a",
    "zurich_city_05_b",
    "zurich_city_08_a",
    "zurich_city_01_a",
    "zurich_city_10_b",
    "zurich_city_02_e",
    "zurich_city_05_a",
    "zurich_city_06_a",
    "zurich_city_11_a",
    "zurich_city_02_a",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/Datasets/DSEC")
    parser.add_argument("--num-bins", type=int, default=10)
    parser.add_argument("--valid-stride", type=int, default=10)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--sequences", nargs="*", default=DEFAULT_FLOW_SEQUENCES)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    device = resolve_device(args.device)

    all_train_names: list[str] = []
    all_valid_names: list[str] = []
    processed_sequences: list[str] = []

    for sequence in args.sequences:
        flow_dir = root / "train_optical_flow" / sequence / "flow" / "forward"
        events_dir = root / "train_events" / sequence / "events" / "left"
        if not flow_dir.exists() or not events_dir.exists():
            if args.skip_missing:
                print(f"Skipping missing sequence {sequence}", flush=True)
                continue
            raise FileNotFoundError(
                f"Missing raw data for {sequence}: flow={flow_dir.exists()} events={events_dir.exists()}"
            )

        _, train_names, valid_names = prepare_sequence(
            root=root,
            sequence=sequence,
            num_bins=args.num_bins,
            valid_stride=args.valid_stride,
            device=device,
            progress_every=args.progress_every,
            write_splits=False,
        )
        all_train_names.extend(train_names)
        all_valid_names.extend(valid_names)
        processed_sequences.append(sequence)
        print(
            f"Finished {sequence}: train={len(train_names)} valid={len(valid_names)} "
            f"aggregate_train={len(all_train_names)} aggregate_valid={len(all_valid_names)}",
            flush=True,
        )

    split_dir = root / "saved_flow_data" / "sequence_lists"
    write_split_csvs(split_dir, all_train_names, all_valid_names)
    print(f"Processed sequences: {len(processed_sequences)}", flush=True)
    print(f"Train samples total: {len(all_train_names)}", flush=True)
    print(f"Valid samples total: {len(all_valid_names)}", flush=True)
    print(f"saved_flow_data root: {root / 'saved_flow_data'}", flush=True)


if __name__ == "__main__":
    main()
