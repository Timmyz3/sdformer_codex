# GateStack 容量安全模式选择器 RTL 验证（2026-07-15）

> **格式替换提示（2026-07-15）**：本文正文记录旧 `192+35×term+8×event` 选择器。RTL 已迁移到 `128+ceil(term/2)×64+8×event` 的 IPD32W，当前边界为 6640/6648，严格回归为 `110 CSR / 191 RAW-class / 203 RAW-capacity`。权威口径见 `docs/87_GateStack_IPD32W有界驻留与无损双格式架构收口_20260715.md`。

## 1. 功能

`rtl_hitflow/gatestack_capacity_mode_selector.sv` 接收一个 head 的统计：

```text
active_classes
class_terms
active_K_events
```

计算：

```text
CSR_bits = 192 + 35×class_terms + 8×active_K_events
```

选择优先级：

```text
active_classes > 4       -> RAW_CLASS_OVERFLOW
否则 CSR_bits > 6642     -> RAW_CAPACITY_OVERFLOW
否则                     -> TERM_CSR
```

class overflow 优先于 capacity overflow，便于论文和性能计数器分账。response 使用单项弹性寄存器，任意反压下 tag、mode、reason 和 CSR bit 数保持稳定。

## 2. 边界构造

方程可构造精确边界：

| classes | terms | events | CSR bits | 期望 |
|---:|---:|---:|---:|---|
| 4 | 6 | 780 | **6642** | TERM-CSR |
| 4 | 1 | 802 | **6643** | RAW capacity |
| 5 | 1 | 1 | 235 | RAW class |
| 0 | 0 | 0 | 192 | TERM-CSR |

比较条件因此锁为 `CSR_bits > 6642`，不能误写成 `>=`。

## 3. 验证产物

| 文件 | 作用 |
|---|---|
| `tb_hitflow/tb_gatestack_capacity_mode_selector.sv` | 4 组边界 + 500 组随机 + response stall |
| `verif_hitflow/gatestack_capacity_mode_selector_assertions.sv` | response 稳定、mode/reason 一致、reason 合法 |
| `verif_hitflow/bind_gatestack_capacity_mode_selector_assertions.sv` | assertion bind |
| `sim_hitflow/run_gatestack_capacity_checks.sh` | iverilog、Verilator assertion、Yosys 严格入口 |

## 4. 结果

```text
PASS: capacity selector req=504 csr=106 raw_class=191 raw_capacity=207
PASS: GateStack capacity selector；Verilator 0 warning/error
PASS: capacity selector无乘法单元
```

| 阶段 | 结果 |
|---|:---:|
| iverilog 自检 | PASS |
| Verilator assertion | PASS |
| Verilator warning/error | 0 |
| Yosys check | PASS，0 problem |

Yosys `memory -nomap` 结构统计为 37 个通用 cell、8 个 add、6 个 mux、0 个 mul。固定 35-bit descriptor 用 `32+2+1` 移位加，8-bit token ID 用左移 3，避免留下常数乘法单元。

## 5. 已闭合与未闭合

已闭合：

1. class 4/5 边界。
2. CSR 6642/6643 bit 边界。
3. class/capacity 原因优先级。
4. response backpressure 稳定性。
5. 三类统计计数器守恒。

未闭合：

1. selector 输出尚未连接 packed head slot commit。
2. TERM-CSR 实际 pack 字节数必须与 selector 的 bit 公式逐项一致。
3. RAW copy 和 CSR pack 的 `last_word_keep` 尚未定义。
4. 64-bit slot SRAM adapter 和跨 context 读写冲突尚未实现。

## 6. 下一步

下一模块为 `gatestack_head_slot_sram_adapter`：每个 slot 物理 104×64 bit，保存 6642 bit payload 和 metadata；CSR/RAW 共用地址空间。必须支持 context A replay 与 context B commit 并行，并禁止同一 slot 同时读写。
