# GateStack OBI 叶模块 RTL 验证（2026-07-15）

## 1. 实现范围

新增 `rtl_hitflow/gatestack_obi_iterator.sv`，实现 GateStack 的 Occupied-Bit Iterator：

```text
load {tag, occupied_mask[4][32]}
  -> 分层选择最低非空slot
  -> 在该slot内选择最低置位lane
  -> ready/valid输出{tag,slot,lane,last}
  -> 清除已提交bit
  -> done握手
```

该模块只负责 occupancy replay，不读取 gate code 和 destination bitmap，也不包含 product/multicast。模块单一时钟、同步复位，无 CDC。

## 2. 接口合同

| 接口 | 含义 |
|---|---|
| `load_valid/load_ready` | 装载一个 head 的 occupancy mask |
| `load_tag` | 窗口/head/tile 复合 tag，由上层编码 |
| `load_occupied_mask` | `SLOTS×LANES` 置位图，默认 128 bit |
| `entry_valid/entry_ready` | 单条有效 `{slot,lane}` 输出 |
| `entry_last` | 当前 mask 最后一项，与该 entry 同时握手 |
| `done_valid/done_ready` | 包括空 mask 在内的完成握手 |

任意 `entry_ready=0` 时，tag、slot、lane 和 last 必须稳定。mask 为空时不产生伪 entry，直接进入 done。

## 3. 验证产物

| 文件 | 作用 |
|---|---|
| `tb_hitflow/tb_gatestack_obi_iterator.sv` | 自检 TB、随机 backpressure、空/单点/稀疏/全满/随机 mask |
| `verif_hitflow/gatestack_obi_iterator_assertions.sv` | stall 稳定、范围、互斥和 load-progress assertion |
| `verif_hitflow/bind_gatestack_obi_iterator_assertions.sv` | assertion bind |
| `sim_hitflow/run_gatestack_obi_checks.sh` | iverilog、Verilator、Yosys 统一入口 |
| `build_hitflow/gatestack_obi/` | 日志和构建产物 |

## 4. 结果

```text
PASS: GateStack OBI iterator loads=104 entries=3351 stalls=1245
PASS: GateStack OBI iverilog + Verilator lint/assert + Yosys；Verilator 0 warning/error
```

| 阶段 | 结果 |
|---|:---:|
| iverilog 功能自检 | PASS |
| Verilator lint | PASS，0 warning/error |
| Verilator assertions | PASS |
| Yosys hierarchy/check | PASS，0 problem |
| 随机 load | 104 |
| 提交 entry | 3351 |
| entry backpressure 周期 | 1245 |

Yosys `memory -nomap` 结构统计：283 个通用 cell，包括 80 个 mux、32 个 `$shiftx` 和状态寄存器。该数字只用于后续 fixed-scan/OBI 相对结构比较，不是 DC 面积。

此外使用项目环境执行脚本目录全量回归：

```bash
PYTHONPATH=scripts /opt/conda/envs/sdformerflow/bin/python \
  -m unittest discover -s scripts -p 'test_*.py' -v
```

结果为 **54/54 PASS**。系统 `/usr/bin/python3` 不含 `torch`，会在既有量化合同测试导入阶段失败，因此全量回归必须使用上述项目环境；GateStack 自身新增测试不依赖 GPU。

## 5. 已验证不变量

1. 所有置位项按 `{slot,lane}` 递增顺序恰好输出一次。
2. 空 mask 不输出 entry，但仍完成 done。
3. 全 128 bit mask 在随机反压下无遗漏。
4. entry stall 时 payload 和 last 稳定。
5. entry 与 done 不同时有效。
6. slot/lane 永不越界。
7. load 后下一周期进入 entry 或 done，不悬挂。

## 6. 未验证边界

1. 叶模块当前锁存完整 128-bit pending mask；集成 head store 后可评估按 slot 读取 32-bit lane mask以减少寄存器。
2. priority 选择器尚无目标库 WNS、面积和动态功耗。
3. 还未连接 gate code、destination bitmap、SRAM 延迟和 product backpressure。
4. 未与 fixed-scan 控制器做同库消融。
5. 不代表完整 GateStack、多 head/tile 或 DC 签核。

## 7. 下一步

实现 `gatestack_head_store` 的 DIRECTORY/DIRECT 双格式合同，并将 OBI 的 `{slot,lane}` 接到 gate/bitmap 读口。集成验证必须强制构造第 5 个 gate class，证明整 head DIRECT fallback 不丢失先前已经写入的 token。
