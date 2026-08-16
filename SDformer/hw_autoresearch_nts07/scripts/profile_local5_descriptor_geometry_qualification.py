#!/usr/bin/env python3
"""比较Local5 active source的边界geometry与descriptor有效mask。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HEIGHT = 15
WIDTH = 15
PLANES = 2
SOURCES = HEIGHT * WIDTH * PLANES


def read_hex(path: Path) -> list[int]:
    return [int(line.strip(), 16) for line in path.read_text().splitlines() if line.strip()]


def address(plane: int, y: int, x: int) -> int:
    return plane * HEIGHT * WIDTH + y * WIDTH + x


def valid(y: int, x: int) -> bool:
    return 0 <= y < HEIGHT and 0 <= x < WIDTH


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vector-dir",
        type=Path,
        default=Path("tb_qfit/vectors/local5_active_projection_postg0_100"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/local5_descriptor_geometry_qualification_20260804"),
    )
    args = parser.parse_args()

    candidate = read_hex(args.vector_dir / "input_valid.memh")
    active = read_hex(args.vector_dir / "input_active.memh")
    if len(candidate) != len(active) or len(active) % SOURCES:
        raise ValueError("向量长度与T450不一致")
    groups = len(active) // SOURCES

    geometry_roles = 0
    descriptor_roles = 0
    differing_sources = 0
    descriptor_subset_violations = 0
    active_sources_total = 0
    per_group = []

    # role在destination侧到source侧的偏移；frontier读回使用相反偏移。
    destination_to_source = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    source_to_relation = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))

    for group in range(groups):
        base = group * SOURCES
        active_sources: set[tuple[int, int, int]] = set()
        for plane in range(PLANES):
            for y in range(HEIGHT):
                for x in range(WIDTH):
                    mask = active[base + address(plane, y, x)]
                    for role, (dy, dx) in enumerate(destination_to_source):
                        sy, sx = y + dy, x + dx
                        if mask & (1 << role) and valid(sy, sx):
                            active_sources.add((plane, sy, sx))

        group_geometry = 0
        group_descriptor = 0
        group_differing = 0
        for plane, y, x in active_sources:
            geometry_mask = 0
            descriptor_mask = 0
            for role, (dy, dx) in enumerate(source_to_relation):
                ry, rx = y + dy, x + dx
                if valid(ry, rx):
                    geometry_mask |= 1 << role
                    if candidate[base + address(plane, ry, rx)] & (1 << role):
                        descriptor_mask |= 1 << role
            if descriptor_mask & ~geometry_mask:
                descriptor_subset_violations += 1
            group_geometry += geometry_mask.bit_count()
            group_descriptor += descriptor_mask.bit_count()
            group_differing += int(geometry_mask != descriptor_mask)

        active_sources_total += len(active_sources)
        geometry_roles += group_geometry
        descriptor_roles += group_descriptor
        differing_sources += group_differing
        per_group.append(
            {
                "group": group,
                "active_sources": len(active_sources),
                "geometry_roles": group_geometry,
                "descriptor_roles": group_descriptor,
                "differing_sources": group_differing,
            }
        )

    summary = {
        "schema": "local5_descriptor_geometry_qualification_v1",
        "evidence": "本机T450 post-G0 profile100输入向量，按RTL邻接映射重建",
        "groups": groups,
        "active_sources": active_sources_total,
        "geometry_roles": geometry_roles,
        "descriptor_roles": descriptor_roles,
        "roles_removed_by_descriptor_mask": geometry_roles - descriptor_roles,
        "role_reduction": 1 - descriptor_roles / geometry_roles,
        "sources_with_different_mask": differing_sources,
        "descriptor_subset_violations": descriptor_subset_violations,
        "per_group": per_group,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Local5 Descriptor 与 Geometry Mask 对照",
        "",
        "## 结论",
        "",
        f"profile100 共重建 {active_sources_total:,} 个 active sources。边界 geometry role 数为 {geometry_roles:,}，descriptor candidate-valid role 数为 {descriptor_roles:,}，减少 {geometry_roles - descriptor_roles:,}（{1 - descriptor_roles / geometry_roles:.2%}）；mask 不同的 source 为 {differing_sources:,} 个。",
        "",
        "该统计只回答 descriptor mask 是否比边界 geometry 更窄，不直接证明周期、面积或功耗。",
        "",
        "## 一致性",
        "",
        f"descriptor 超出 geometry 边界的违规数为 {descriptor_subset_violations}；应为0。active source总数应与RTL profile100 descriptors一致。",
    ]
    (args.output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
