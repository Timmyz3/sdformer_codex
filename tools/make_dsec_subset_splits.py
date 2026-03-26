#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from pathlib import Path


def load_rows(path: Path) -> list[str]:
    with path.open("r", newline="") as f:
        return [row[0].strip() for row in csv.reader(f) if row]


def sequence_name(sample_name: str) -> str:
    # Example: zurich_city_09_a_0002.npy -> zurich_city_09_a
    return "_".join(sample_name.split("_")[:-1])


def select_rows(rows: list[str], sequences: list[str], limit_per_sequence: int | None) -> list[str]:
    if not sequences:
        return rows if limit_per_sequence is None else rows[:limit_per_sequence]

    allowed = set(sequences)
    counts: dict[str, int] = defaultdict(int)
    selected: list[str] = []
    for row in rows:
        seq = sequence_name(row)
        if seq not in allowed:
            continue
        if limit_per_sequence is not None and counts[seq] >= limit_per_sequence:
            continue
        selected.append(row)
        counts[seq] += 1
    return selected


def write_rows(path: Path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        for row in rows:
            writer.writerow([row])


def main() -> int:
    parser = argparse.ArgumentParser(description="Create fixed DSEC subset split CSVs")
    parser.add_argument(
        "--root",
        default="/root/private_data/work/SDformer/data/Datasets/DSEC/saved_flow_data",
        help="saved_flow_data root",
    )
    parser.add_argument(
        "--sequences",
        nargs="+",
        required=True,
        help="sequence names to keep, e.g. zurich_city_01_a thun_00_a",
    )
    parser.add_argument(
        "--train-limit-per-seq",
        type=int,
        default=200,
        help="max training samples per sequence",
    )
    parser.add_argument(
        "--valid-limit-per-seq",
        type=int,
        default=40,
        help="max validation samples per sequence",
    )
    parser.add_argument(
        "--train-source",
        default="train_split_seq.csv",
        help="source train csv filename under sequence_lists",
    )
    parser.add_argument(
        "--valid-source",
        default="valid_split_seq.csv",
        help="source valid csv filename under sequence_lists",
    )
    parser.add_argument(
        "--train-output",
        default="train_subset_split_seq.csv",
        help="output train csv filename under sequence_lists",
    )
    parser.add_argument(
        "--valid-output",
        default="valid_subset_split_seq.csv",
        help="output valid csv filename under sequence_lists",
    )
    args = parser.parse_args()

    root = Path(args.root)
    sequence_lists = root / "sequence_lists"
    train_rows = load_rows(sequence_lists / args.train_source)
    valid_rows = load_rows(sequence_lists / args.valid_source)

    selected_train = select_rows(train_rows, args.sequences, args.train_limit_per_seq)
    selected_valid = select_rows(valid_rows, args.sequences, args.valid_limit_per_seq)

    if not selected_train:
        raise RuntimeError("no training rows matched the requested sequences")
    if not selected_valid:
        raise RuntimeError("no validation rows matched the requested sequences")

    train_out = sequence_lists / args.train_output
    valid_out = sequence_lists / args.valid_output
    write_rows(train_out, selected_train)
    write_rows(valid_out, selected_valid)

    print(f"train_output: {train_out}")
    print(f"valid_output: {valid_out}")
    print(f"train_rows: {len(selected_train)}")
    print(f"valid_rows: {len(selected_valid)}")
    print("sequences:")
    for seq in args.sequences:
        print(f"  - {seq}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
