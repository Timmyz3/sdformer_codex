# M1125K — Motion ep29 到硬件证据的最小重绑定矩阵

状态：`PASS_M1125K_EP29_HARDWARE_MINIMAL_REBIND_MATRIX__WAIT_FINAL_CHECKPOINT__NO_REPLAY`

裁决：**现在不需要等 checkpoint 才能继续做静态硬件收口，但所有由权重或激活决定的数字必须等 ep29 后重放。ep24 只能保留候选身份，不能升级任何论文性能或精度 claim。**

本审计只读绑定 M1120/M1121，没有远端访问、等待、下载、训练、GPU、硬件 replay 或 EDA。`docs/359` 未修改。

## 1. 当前绑定身份

- M1120 已冻结 ep24 candidate：checkpoint SHA256 `1e55900cd0bb4e411d09a5e4cd7bd56c08c60874a1e4868f6494d18b3e691e32`，但 `final=false`、`valid825=false`、`hardware_admitted=false`。
- M1120 contract 外层封印文件 SHA256：`dba10344a47ec8c57d6b667c7df7e81ecb983bf5dff98cbe1439f5685f69923c`。
- M1121 独立 hammer 外层封印文件 SHA256：`8539072a4b1afe3c1f717632de0bbf5e3b21633e8845ab39a5030c45f1f1d7ce`；裁决明确为等待 ep29 与 valid825 后再重绑。
- `docs/359` SHA256：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 2. 可以保留的 checkpoint-invariant 证据

“保留”均以 ep29 的结构签名完全一致为前提，不代表 ep29 性能已经准入。

| 证据 | 可保留部分 | 不能顺带保留的部分 | ep29 门 |
|---|---|---|---|
| RTL 与 directed/mapped VCS 协议 | ready/valid、backpressure、reset、端口顺序、相同参数下算术/协议功能 | 真实 hit rate、稀疏率、周期、倍率、toggle | 层名、shape、精度、T、rank、value class、接口参数逐项相同 |
| DC/STA/Formality/静态 PPA | 相同 RTL/参数/SDC/lib/macro 接口下的 logic area、setup/hold、形式等价 | SAIF 动态功耗、energy/frame、有效吞吐、系统 PPA | 精确 source/filelist/constraint/library/macro manifest 相同 |
| 容量、地址和端口合同 | provisioned bytes、bank/port/address width、Acc24、descriptor 最大格式 | 实际 occupancy、working-set fit、refill、冲突、parent/PWP 驻留 | shape、编码位宽、banking、T10、rank/value class 相同 |
| 外部论文与工具方法 | Prosperity/Phi/FireFly-T/CICC 的论文数字、引用、比较定义和 artifact 版本 | 在 H67 ep35/ep24 tensor 上跑出的官方 artifact 数字 | ep29 workload replay 必须单列为 external baseline，不能写成 ours |
| 参数化正确性与协议负结果 | fail-closed 不变量、结构合法性/不可能性 | 空 tile、delta density、product reuse、Amdahl 等数据分布结论 | 所有证明假设必须逐项出现在 ep29 manifest |

额外边界：C2 的等带宽静态面积证据可在结构一致后保留；其 ep29 workload 周期与能量仍要重放。C1 的单口协议/VCS 可以保留；约 `1.75×` 账本数字不能继承。C3 Fixed-T10 的静态 island 只有在最终仍为 T10/同 rank/value class 时可保留。

## 3. ep29 必须重跑的证据

| 顺序 | 必跑项 | 最小输出 |
|---|---|---|
| E0 | ep29 最终身份 | SHA/size/mtime 双观察、预声明选择回执、config/source/dataset hash、strict load 0/0、拓扑与数制签名 |
| E1 | valid825 | standard、dyadic/quantized、hardware-order RTL-exact AEE/AAE，spike/activity 总量 |
| E2 | 一次统一 ordered capture | full-network trace；固定 C1 十样本；decoder 3 sequence × 10 sample；四 Conv、D0-D3、ATLIF/FC/patch/BN/QKV taps |
| E3 | C1 四层 Conv | 全 51.84M source-row/十样本账本；zero/bit/product/单口 capture cycle；occupancy/conflict/refill/DRAM；官方 Prosperity 单列 replay |
| E4 | decoder | D0/D2/D3 bitpack；D1 theta 与数值 miter；D0-D3 address-timed cycle/traffic；完整 decoder row |
| E5 | 非 attention 活动 | ATLIF、FC1/FC2、patch embed、BN 的 zero/value/event/K-group/commit/traffic 分布 |
| E6 | attention/RQTB | ep29 Q/K NPZ；Fixed↔RQTB exact miter；multiplicity/equality/reuse/K-zero；局部周期和全网 Amdahl |
| E7 | 活动功耗与系统表 | real-trace VCS、name-mapped SAIF、PTPX、宏读写能量、decoder-complete memory-inclusive Table A |
| E8 | 数值与压缩再准入 | ep29 weight/bias export、累加器无溢出、INT/CSD/pattern mismatch、压缩 occupancy 对结构容量 |

核心原则：**RTL 面积可以因结构不变而继承；倍率、流量、活动率、动态功耗、energy/frame 和系统表不能。**

## 4. 可机械绑定的 identity manifest

最终 manifest 至少包含：checkpoint SHA/size/mtime、final-selection receipt、config、model commit 与 dirty-diff digest、dataset/preprocess、strict load 0/0、逐层 name/shape/dtype、precision/rank/timestep/value-class、RTL/filelist/TB/SDC/Tcl/lib/macro、固定 cohort、raw capture、derived ledger、SAIF/PTPX 和 Table-A manifest。

每个派生行必须同时 join：`checkpoint SHA + deployment/config SHA + cohort SHA + generator SHA + simulator/RTL identity`。缺任一键或混入 ep35/ep44/ep24 数据均 fail closed。结构相同不能替代 weight、量化 payload、活动和 occupancy 的重放。

## 5. 依赖 DAG 与最短墙钟路径

```text
R0 ep29稳定身份
 └─R1 strict-load + 结构/数值签名
    ├─R2A valid825 ──────────────────────────────┐
    └─R2B 一次统一ordered capture                │
       ├─R3A C1 51.84M账本                       │
       ├─R3B decoder bitpack/address cycles      ├─R5 Table A─R6独立hammer/封印
       ├─R3C ATLIF/FC/patch/BN/RQTB活动          │
       ├─R3D 权重/范围/压缩fit                    │
       └─R4A real-trace VCS/SAIF─R4B PTPX────────┘
```

最短执行方式：A800 只加载一次 ep29，先后完成固定 cohort capture 与 valid825；capture 一封存，C1、decoder、活动/RQTB、数值范围四条 CPU 线立即并行，最多三 worker；结构签名一致时复用已有 netlist，只重跑 real-trace VCS/SAIF/PTPX，不重跑 DC。

在无排队、license 或工具失败时，规划墙钟为 **8–16 小时**，运营上预留 **24 小时**。这是排程估计，不是实验测量或论文指标。

## 6. ep24 禁止升级的 claim

1. 不得称 ep24 为 final/ep29，也不得把 epoch24 loss 写成 valid825 AEE/AAE。
2. 不得把 C1 `~1.75×`、51.84M opportunity、decoder cycle、C2 effective throughput、C3 activity 或 RQTB 压缩绑定到 ep24。
3. 不得继承 ep35/ep44 的 sparsity、reuse、traffic、SAIF、power、energy、Amdahl 或 Table A 到 ep29。
4. 不得在 E0–E8 以同一 ep29 identity 汇合前声称 decoder-complete、full-system speedup 或 energy/frame。
5. 不得把静态 DC/PPA 当作动态能量，也不得把官方 Prosperity/Phi replay 写成 ours 或把局部倍率相乘。
6. 不得仅因 YAML 文件名相同就假设拓扑相同；必须比较实际加载后的结构签名。
7. 不得用 valid825 调阈值/catalog 后又把同一 valid825 当独立 accuracy gate。
8. 不得为了“填时间”对 ep24 做完整硬件 replay；M1120/M1121 已明确禁止。

## 7. 可立即继续而无需等待 ep29 的工作

- 继续封 RTL/VCS 协议、DC/STA/Formality、SRAM port/capacity、宏模型和论文对标方法；这些都必须保持 checkpoint-invariant claim。
- 把统一 capture writer、manifest schema、CPU replay 与 Table-A joiner预先静态审阅好，但不在 ep24 上启动生产 replay。
- 论文可先写架构、协议、方法和静态 PPA；所有 activity/speedup/energy/accuracy 单元格保持 pending，等 ep29 重绑定后填入。
