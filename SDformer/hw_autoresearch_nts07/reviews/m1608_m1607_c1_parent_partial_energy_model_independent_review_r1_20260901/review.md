# M1608｜M1607 C1 parent partial energy model 独立复核

日期：2026-09-01

状态：`PASS_M1608_M1607_PARENT_PARTIAL_ENERGY_MODEL__EXACT_LABEL_REQUIRED`

评分：98/100；P0=0，P1=1，P2=0。

## 裁决

M1607 的输入身份、16B macro activation 换算、10.50786/10.07307 pJ 动态能量计算、10-sample 归一化以及 9/105 宏 leakage 算术全部通过独立重算。它可作为 C1 的已知部分能量模型进入组件表，但只能使用严格的 `[component partial model]` 标签。

它绝不能写成 total C1 energy、energy/frame、full-network/system energy 或 measured power。

## 输入身份

独立核对了以下 exact SHA：

- M1607 result：`adf0648fa3b9b1ac2d085d094fb060cfe57ed376bad49c808c6f8c5c717f2e60`；
- author source/test：`4929659c...` / `6baab4a7...`；
- M1597 cycle/traffic authority：`bfa3414e...`；
- M1125C capacity/energy authority：`348e18eb...`；
- M1006 3-ns setup coordinate：`d7b30ff3...`；
- M623 generated-macro coefficient authority：`96812391...`；
- docs/359：`dedde7ce...`。

author test 2/2 PASS；重新执行 source 后 JSON 与冻结 result 完全一致。独立脚本没有复用作者的能量函数，而是用 Decimal 从三份输入 authority 重新计算。

## 16B macro activation 换算

M1597 的 parent-only traffic 为：

| 项 | 字节 | 144B 完整向量访问 | ×9 个 16B 宏 | byte/16 交叉校验 |
|---|---:|---:|---:|---:|
| read | 16,711,429,248 | 116,051,592 | 1,044,464,328 | 1,044,464,328 |
| write | 10,449,510,912 | 72,566,048 | 653,094,432 | 653,094,432 |

两项均可被 144 整除，没有 partial-vector remainder；`vector_accesses×9` 与 `bytes/16` 完全相同。因此，在 144B 全向量映射到九个 128b 1RW 宏的组件模型内，activation 换算成立。

## 能量独立重算

系数来自 M623 独立复核的 `TS1N28HPCPHVTB128X128M4S`、`ssg0p9v125c` datasheet component model：read `10.50786 pJ/activated macro`，write `10.07307 pJ/activated macro`，leakage `0.06001047 mW/native macro`。

动态能量：

- read：`1,044,464,328 × 10.50786 = 10,975,084,933.61808 pJ`；
- write：`653,094,432 × 10.07307 = 6,578,665,930.14624 pJ`；
- 10 samples aggregate：`17.55375086376432 mJ`；
- per captured sample：`1.755375086376432 mJ`。

leakage 时间来自 `382,848,700 cycles × 3.0 ns = 1.1485461000 s`，这是十个 sample 的 aggregate cycle-model time：

- 9-macro parent leakage：`0.062032312149900300 mJ/sample`；
- 105-macro capacity-equivalent leakage：`0.723710308415503500 mJ/sample`。

M1607 的 known partial 为：

`1.755375086376432 + 0.723710308415503500 = 2.479085394791935500 mJ/captured-sample [model]`。

105-macro leakage 只加一次。单列的 9-macro leakage 是诊断值，没有再次叠加，因此不存在 leakage double count。

## 唯一 P1

`P1_EXACT_COMPONENT_MODEL_LABEL_REQUIRED`：作者 JSON 的 Boolean boundary 是安全的，但 top-level scope 没有直接写出 macro cell/corner，也没有直接写出 105-macro capacity model 尚未物理集成。因此禁止脱离 M1608 单独引用 M1607 数字。

合法标签必须同时包含：

> parent dynamic + 105-macro capacity-equivalent leakage；TS1N28HPCPHVTB128X128M4S at ssg0p9v125c；3-ns cycle-model time；ten captured samples、one sequence、four bottleneck Conv3x3；component partial [model]。

## 禁止升级

当前缺少 weight、psum、metadata、logic、DRAM dynamic energy，也没有 physically integrated 105-macro top 或 measured power。这里的 sample 是冻结 capture 的 event window，不是完整 camera frame。因此以下说法全部非法：

- `2.479 mJ total C1 energy`；
- `2.479 mJ/frame`；
- `2.479 mJ full-network/system energy`；
- `measured power/energy`。

本复核只读，没有运行 EDA/GPU/RTL，没有修改 M1607 作者结果或 docs/359。
