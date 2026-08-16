#!/usr/bin/env python3
"""从qualified Local5 post-G0负载生成单颜色bank的GASR RTL向量。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from profile_local5_source_stationary_acc_cache import (
    BANK_DEPTH,
    HEAD_DIM,
    HEIGHT,
    ROLE_DX,
    ROLE_DY,
    ROLES,
    WIDTH,
    destination_bank_address,
)


OUT_DIM = 2
ACC_W = 32
TARGET_BANK = 0


def weight_value(lane: int, out: int) -> int:
    return (lane % 5 + 1) * (1 if out == 0 else -2)


def signed_hex(value: int, width: int) -> str:
    return f"{value & ((1 << width) - 1):0{(width + 3) // 4}x}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_signed_hex(line: str, width: int = ACC_W) -> int:
    value = int(line.strip(), 16)
    return value - (1 << width) if value & (1 << (width - 1)) else value


def source_bank_updates(
    plane: int,
    source_y: int,
    source_x: int,
    k_bitmap: int,
    gates: np.ndarray,
    valid_mask: int,
) -> tuple[int | None, list[tuple[int, int]]]:
    gate_values: list[int] = []
    gate_roles: list[list[int]] = []
    for role in range(ROLES):
        gate = int(gates[role])
        if not ((valid_mask >> role) & 1) or gate == 0:
            continue
        if gate in gate_values:
            gate_roles[gate_values.index(gate)].append(role)
        else:
            gate_values.append(gate)
            gate_roles.append([role])

    target_addr: int | None = None
    updates: list[tuple[int, int]] = []
    for lane in range(HEAD_DIM):
        if not ((k_bitmap >> lane) & 1):
            continue
        for gate, roles in zip(gate_values, gate_roles, strict=True):
            for role in roles:
                y = source_y + ROLE_DY[role]
                x = source_x + ROLE_DX[role]
                if not (0 <= y < HEIGHT and 0 <= x < WIDTH):
                    raise AssertionError("valid role越过Local5窗口边界")
                bank, addr = destination_bank_address(plane, y, x)
                if bank != TARGET_BANK:
                    continue
                if target_addr is not None and target_addr != addr:
                    raise AssertionError("同一source在目标颜色bank命中多个地址")
                target_addr = addr
                updates.append(
                    (gate * weight_value(lane, 0), gate * weight_value(lane, 1))
                )
    return target_addr, updates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("tb_qfit/vectors/local5_active_projection_postg0_100/manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tb_qfit/vectors/local5_gasr_singlebank_postg0_100"),
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload_path = Path(manifest["source_payload"])
    payload = np.load(payload_path, mmap_mode="r")
    offsets = np.asarray(payload["descriptor_group_offsets"])
    k_bitmaps = np.asarray(payload["descriptor_k_bitmap"])
    gates = np.asarray(payload["descriptor_incoming_gates"])
    valid_masks = np.asarray(payload["descriptor_valid_mask"])
    planes = np.asarray(payload["descriptor_source_plane"])
    ys = np.asarray(payload["descriptor_source_y"])
    xs = np.asarray(payload["descriptor_source_x"])

    group_source_offsets = [0]
    source_update_offsets = [0]
    source_addrs: list[int] = []
    update_vectors: list[tuple[int, int]] = []
    expected_vectors: list[tuple[int, int]] = []
    group_rows: list[dict[str, int]] = []
    full_expected_lines = (
        args.manifest.parent / "expected_acc.memh"
    ).read_text(encoding="ascii").splitlines()
    if len(full_expected_lines) != len(manifest["selection"]["rows"]) * 450 * OUT_DIM:
        raise ValueError("原post-G0 Acc32金向量长度错误")

    for vector_group, row in enumerate(manifest["selection"]["rows"]):
        input_group = int(row["input_group_index"])
        start = int(offsets[input_group])
        stop = int(offsets[input_group + 1])
        expected = [[0, 0] for _ in range(BANK_DEPTH)]
        group_sources = 0
        group_updates = 0
        for index in range(start, stop):
            addr, updates = source_bank_updates(
                int(planes[index]),
                int(ys[index]),
                int(xs[index]),
                int(k_bitmaps[index]),
                gates[index],
                int(valid_masks[index]),
            )
            if addr is None:
                if updates:
                    raise AssertionError("无目标地址却产生更新")
                continue
            if not updates:
                raise AssertionError("几何目标在exact payload中没有更新")
            source_addrs.append(addr)
            group_sources += 1
            for delta0, delta1 in updates:
                update_vectors.append((delta0, delta1))
                expected[addr][0] += delta0
                expected[addr][1] += delta1
                group_updates += 1
            source_update_offsets.append(len(update_vectors))
        group_source_offsets.append(len(source_addrs))
        for plane in range(2):
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    bank, addr = destination_bank_address(plane, y, x)
                    if bank != TARGET_BANK:
                        continue
                    destination = plane * HEIGHT * WIDTH + y * WIDTH + x
                    for out in range(OUT_DIM):
                        full_index = (vector_group * 450 + destination) * OUT_DIM + out
                        if expected[addr][out] != parse_signed_hex(
                            full_expected_lines[full_index]
                        ):
                            raise AssertionError(
                                f"bank金结果与原顶层向量不一致 group={vector_group} "
                                f"addr={addr} out={out}"
                            )
        expected_vectors.extend(expected)
        group_rows.append(
            {
                "vector_group": vector_group,
                "input_group": input_group,
                "sample": int(row["sample"]),
                "stage": int(row["stage"]),
                "window": int(row["window"]),
                "head": int(row["head"]),
                "sources": group_sources,
                "updates": group_updates,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, list[str]] = {
        "group_source_offsets.memh": [f"{value:08x}" for value in group_source_offsets],
        "source_update_offsets.memh": [f"{value:08x}" for value in source_update_offsets],
        "source_addr.memh": [f"{value:02x}" for value in source_addrs],
        "update_delta.memh": [
            signed_hex(delta1, ACC_W) + signed_hex(delta0, ACC_W)
            for delta0, delta1 in update_vectors
        ],
        "expected_acc.memh": [
            signed_hex(value1, ACC_W) + signed_hex(value0, ACC_W)
            for value0, value1 in expected_vectors
        ],
    }
    for name, lines in files.items():
        (args.output_dir / name).write_text("\n".join(lines) + "\n", encoding="ascii")

    output_manifest = {
        "schema": "local5_gasr_singlebank_postg0_vectors_v1",
        "evidence": "本机qualified Local5 post-G0 source-major trace",
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "source_payload": str(payload_path.resolve()),
        "source_payload_sha256": sha256(payload_path),
        "target_bank": TARGET_BANK,
        "groups": len(group_rows),
        "sources": len(source_addrs),
        "updates": len(update_vectors),
        "bank_depth": BANK_DEPTH,
        "out_dim": OUT_DIM,
        "acc_width": ACC_W,
        "full_acc32_crosscheck": "18000/18000 PASS",
        "rows": group_rows,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output_manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
