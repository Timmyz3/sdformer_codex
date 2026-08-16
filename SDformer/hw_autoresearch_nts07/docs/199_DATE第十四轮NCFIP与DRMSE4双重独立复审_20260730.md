# DATE 第十四轮 NC-FIP 与 DRMS-E4 双重独立复审

## 1. 结论

两位独立审稿人分别从 DATE 和物理架构角度审查，结论一致：

| 角色 | Recommendation | 总分 | NC-FIP新颖性 |
|---|---|---:|---:|
| DATE审稿人 | Weak Reject | 2.5/5 | 2.8/5 |
| 架构/电路审稿人 | Major Revision | 2.5/5 | 有条件成立 |

共同判断：

- 原 FCIP 已由同带宽结果淘汰；
- NC-FIP 相对 B2 有真实关系存储差分；
- NC-FIP 相对 B1 是否成立，取决于 SCS 是否直接产生 gate relation；
- DRMS-E4 只能作为尾延迟微结构，不能单列 DATE 贡献；
- 当前没有资格进入 RTL 签核或论文结果。

---

## 2. 最关键的因果错误

旧 `fused` 模型把 class bitmap 转导周期设为零，理由是与 SCS occupied-class
scan 重叠。该假设不成立。

现有 `h67_score_class_row_engine`：

1. `ST_SUM_ACTIVE` 累加 active token 的 denominator；
2. `ST_FIND_FOLD/ST_SUM_FOLD` 累加 K-zero class；
3. 完整 `row_sum_q` 稳定后，`ST_EMIT` 才能用
   `ttx_gate_quant_q17` 产生 final gate。

因此：

> class-to-final-gate 不能在“求 denominator 的同一次扫描”中完成。

可实现结构至少需要两遍：

```text
pass 1: denominator
pass 2: class -> final gate -> relation fold
```

旧 `1.192x/1.201x fused` 只能视为不可达上界，不能写进论文。

---

## 3. 物理 P0

### 3.1 W4不是同物理端口

抽象 W4 未回答：

- 每拍四 token 的 class-word 冲突如何合并；
- 最多 128 个 K bit 更新如何落 bank；
- occupancy、allocator 和 fallback 是否同拍；
- B1 与 FCIP 每拍真实读写 bit 数是否相同。

所以目前只是同数值宽度，不是同面积、同端口或同能耗。

### 3.2 4-way intersection 必须 segment-major

若四个 context 任意访问不同 G segment，则 G bank 需要多读口或复制。可接受
的最小组织是：

```text
one G-segment read
  -> broadcast
  -> four independent K-lane reads
  -> four 64-bit AND
```

这要求严格 segment-major 调度，不能继续使用抽象 task 任意减
`read_work` 的周期模型。

### 3.3 context 必须是真实位图状态

每个 context 至少包含：

- T-bit destination bitmap；
- row/gate/lane tag；
- segment-valid；
- epoch；
- complete/valid。

当前脚本只模拟占用，不证明四路 fragment 的写端口。

### 3.4 回压与事务

固定周期 100/90/75% ready 不能代表 burst stall。真实回压必须沿：

```text
term sink -> context -> intersection -> gate fold -> SCS -> row input
```

S16/G4 overflow 在 term 对外可见前必须 abort，或者使用 epoch 清理和完整
replay。旧模型只加一拍不成立。

---

## 4. DATE晋级门槛

进入 RTL 候选：

1. profile100、多样本、多 window；
2. W4 相对最佳 B1/B2 至少 `1.15x`；
3. paired p99 slowdown `<=1.05x`；
4. 慢于10%的row不超过1%；
5. S16/G4 overflow均不超过0.1%，exact fallback零失配；
6. 不保留完整B1目录；
7. 同一物理端口图和相同sink trace。

进入 DATE 主贡献：

1. 同宏、同SDC、同memory rule；
2. attention-to-projection EDP改善至少15%；
3. 总面积不超过1.10x；
4. Fmax下降不超过5%；
5. score-to-projection traffic下降至少20%；
6. DC/STA/SAIF和多样本bit-exact闭合。

---

## 5. 对下一轮的直接指导

复审暴露出一个比 NC-FIP 更根本的机会：

> 现有 SCS 只折叠 K-zero class，active token 仍逐 token 求 denominator 和
> emit。只要 K value carrier 独立保存在 K-lane plane，active token 也可
> 进入 class histogram。

因此下一轮不再追求“把第二遍免费化”，而是重构为：

```text
all-class denominator
  -> active-class final-gate fold
  -> factorized G∩K projection
```

这就是后续 ACRT 路线。

