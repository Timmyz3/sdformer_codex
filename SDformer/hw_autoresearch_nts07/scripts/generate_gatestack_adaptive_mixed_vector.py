#!/usr/bin/env python3
"""由同语义IPD32W/FADC24向量生成逐head交错的Adaptive CSR回归向量。"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


COMMON_FILES = (
    "projection_bias_acc.memh",
    "projection_weight_scale_exp2.memh",
    "projection_weights_int8.memh",
    "expected_output_acc32.memh",
)
HEAD_FILES = (
    "payload_bits.memh",
    "payload_modes.memh",
    "payload_word_counts.memh",
    "term_counts.memh",
    "event_counts.memh",
)


def read_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="ascii").splitlines()]


def write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ipd-dir", type=Path, required=True)
    parser.add_argument("--fadc-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--heads", type=int, default=24)
    parser.add_argument("--words-per-head", type=int, default=104)
    parser.add_argument("--replace-raw-with-fadc", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in COMMON_FILES:
        ipd_data = (args.ipd_dir / name).read_bytes()
        fadc_data = (args.fadc_dir / name).read_bytes()
        if ipd_data != fadc_data:
            raise ValueError(f"同语义公共向量不一致: {name}")
        shutil.copyfile(args.ipd_dir / name, args.output_dir / name)

    ipd = {name: read_lines(args.ipd_dir / name) for name in HEAD_FILES}
    fadc = {name: read_lines(args.fadc_dir / name) for name in HEAD_FILES}
    for name in HEAD_FILES:
        if len(ipd[name]) != args.heads or len(fadc[name]) != args.heads:
            raise ValueError(f"{name} head数量错误")
    for name in ("term_counts.memh", "event_counts.memh"):
        if ipd[name] != fadc[name]:
            raise ValueError(f"同语义事件计数不一致: {name}")

    ipd_words = read_lines(args.ipd_dir / "payload_words.memh")
    fadc_words = read_lines(args.fadc_dir / "payload_words.memh")
    expected_words = args.heads * args.words_per_head
    if len(ipd_words) != expected_words or len(fadc_words) != expected_words:
        raise ValueError("payload_words固定head槽长度错误")

    selected = []
    outputs = {name: [] for name in HEAD_FILES}
    mixed_words: list[str] = []
    counts = {"IPD32W": 0, "FADC24": 0, "RAW41": 0}
    for head in range(args.heads):
        use_fadc = bool(head % 2)
        if args.replace_raw_with_fadc and ipd["payload_modes.memh"][head] == "0":
            use_fadc = True
        source_name = "FADC24" if use_fadc else "IPD32W"
        source = fadc if source_name == "FADC24" else ipd
        if source["payload_modes.memh"][head] == "0":
            source_name = "RAW41"
        counts[source_name] += 1
        selected.append(source_name)
        for name in HEAD_FILES:
            outputs[name].append(source[name][head])
        begin = head * args.words_per_head
        end = begin + args.words_per_head
        words = fadc_words[begin:end] if use_fadc else ipd_words[begin:end]
        mixed_words.extend(words)

    for name, lines in outputs.items():
        write_lines(args.output_dir / name, lines)
    write_lines(args.output_dir / "payload_words.memh", mixed_words)
    manifest = {
        "schema_version": 1,
        "contract": "H67 S3同一context逐head交错IPD32W/FADC24" +
                    ("，用FADC24替代RAW回退" if args.replace_raw_with_fadc else
                     "，并保留RAW41精确回退"),
        "heads": args.heads,
        "words_per_head": args.words_per_head,
        "formats_by_head": selected,
        "format_counts": counts,
        "source_ipd": str(args.ipd_dir),
        "source_fadc": str(args.fadc_dir),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
