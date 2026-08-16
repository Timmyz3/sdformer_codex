#!/usr/bin/env python3
"""检查 lane-local product cache 的 Yosys 存储与乘法器合同。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def binary_parameter(value: str) -> int:
    return int(value, 2)


def check_netlist(path: Path, ways: int, lanes: int, product_w: int, out_dim: int) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if len(payload["modules"]) != 1:
        raise AssertionError(f"{path}: 预期只有一个顶层模块")
    module = next(iter(payload["modules"].values()))
    cells = module["cells"]

    product_memories = []
    result_multipliers = []
    for name, cell in cells.items():
        if cell["type"] == "$mem_v2" and ".u_product_bank.mem" in name:
            params = cell["parameters"]
            item = {
                "name": name,
                "width": binary_parameter(params["WIDTH"]),
                "depth": binary_parameter(params["SIZE"]),
                "read_ports": binary_parameter(params["RD_PORTS"]),
                "write_ports": binary_parameter(params["WR_PORTS"]),
                "read_clock_enable": binary_parameter(params["RD_CLK_ENABLE"]),
            }
            product_memories.append(item)
        if cell["type"] == "$mul":
            params = cell["parameters"]
            if (
                binary_parameter(params["A_WIDTH"]) == 10
                and binary_parameter(params["B_WIDTH"]) == 8
                and binary_parameter(params["Y_WIDTH"]) == 17
                and binary_parameter(params["A_SIGNED"]) == 1
                and binary_parameter(params["B_SIGNED"]) == 1
            ):
                result_multipliers.append(name)

    expected_memory = {
        "width": product_w,
        "depth": lanes,
        "read_ports": 1,
        "write_ports": 1,
        "read_clock_enable": 1,
    }
    if len(product_memories) != ways:
        raise AssertionError(
            f"{path}: product bank 数={len(product_memories)}，预期={ways}"
        )
    for item in product_memories:
        actual = {key: item[key] for key in expected_memory}
        if actual != expected_memory:
            raise AssertionError(
                f"{path}: {item['name']} 合同错误 {actual} != {expected_memory}"
            )
    if len(result_multipliers) != out_dim:
        raise AssertionError(
            f"{path}: 10x8->17结果乘法器数={len(result_multipliers)}，预期={out_dim}"
        )

    return {
        "ways": ways,
        "product_banks": len(product_memories),
        "product_bank_width": product_w,
        "product_bank_depth": lanes,
        "product_storage_bits": ways * product_w * lanes,
        "result_multipliers": len(result_multipliers),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--netlist-pattern", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--ways", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--lanes", type=int, default=32)
    parser.add_argument("--product-width", type=int, default=68)
    parser.add_argument("--out-dim", type=int, default=4)
    args = parser.parse_args()

    rows = []
    for ways in args.ways:
        path = Path(args.netlist_pattern.format(ways=ways))
        rows.append(
            check_netlist(
                path,
                ways,
                args.lanes,
                args.product_width,
                args.out_dim,
            )
        )
    output = Path(args.output)
    output.write_text(
        json.dumps({"status": "PASS", "rows": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"PASS product-cache structure: {output}")


if __name__ == "__main__":
    main()
