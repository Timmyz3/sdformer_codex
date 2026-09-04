# M513 decoder PGPR/TDR analyzer 独立静态与数学打铁 r1

日期：2026-08-27  
结论：`NO_GO__MATH_PASS_BUT_INPUT_TRUST_AND_ATOMIC_PUBLICATION_NOT_CLOSED`  
评分：**74/100**  
P0：**2**  
P1：**2**  
生产 capture / payload / analyzer / GPU / VCS / DC / DSE 实际执行：**否**

## 总结

M513 的核心数学是正确的，PGPR 也被公平地快杀，TDR 只被保留为带状态税的
条件候选；但是 analyzer SHA
`303863453d56bf6472ecaf55315b2a5e895494eb019ef70e0f25e13233d089be`
当前不能执行生产数据。两个 P0 都在 fail-closed 工程链：输入 contract/verifier
代际没有外部 trust anchor，输出在 canonical 发布后校验失败不会 quarantine。

## 数学独立重构：通过

本评审新增完全合成的独立重构脚本，不打开 contract、checkpoint、capture
bitpack 或 payload-verifier 结果。固定随机种子完成：

- 25 组 `H,W=1..5` scatter 与逐 source/tap 参考循环逐点相等；
- 24 组 T10 合成记录与逐 channel/coordinate 参考循环相等；
- current/destination contributor、delta/rise/fall 全部守恒；
- `a1_cycles=a1_products/96`、`tdr_cycles=tdr_products/96`；
- 四层 state 容量重构为 input bitmap `870,300 B`、output
  `10,598,400 elements`，即 INT16 `21,196,800 B`、Acc24 `31,795,200 B`。

### K3/S2/P1/OP1 边界

坐标是 `o=2i-1+k, k∈{0,1,2}`，输出范围为 `[0,2H)` / `[0,2W)`。
只有 `i=0,k=0` 在 top/left 产生 `-1`；bottom/right 因 output-padding=1
没有裁剪。因此每个 source 的合法 tap 数严格为 corner 4、top/left edge 6、
interior/bottom/right 9。analyzer 的 `arange(1 if k==0 else 0, ...)` 和
fanout `{2,3}×{2,3}` 正确。

### source/destination 与 96-lane 周期

每个 active `(t,cin,y,x)` 对每个合法 tap 产生一个完整 `Cout` weight vector。
四层 `Cout={384,192,96,96}=96q`，不存在 lane tail，因此：

```text
products = source_tap_vectors × Cout
cycles   = source_tap_vectors × q = products / 96
```

scatter 后所有 destination contributor 的总和严格等于 source-tap vector 数；
实现中的 conservation gate 正确。

### T10 temporal delta

`analyze_record` 每次调用都重新建立全零 previous，且每条 record 就是一层一个
sample，所以每个 sample 的 `t0` 都是相对 0；`t>0` 使用前一 timestep。
`XOR = rise ⊎ fall`，fanout 加权后仍有 `delta_vectors=rise+fall`。这里准入的只是
exact transition-product count，不是有限位宽/不同累加顺序下的输出 bit-exact
证明；脚本正确地保持 RTL、cycle-with-SRAM 和性能主张为 false。

### PGPR 与 commit/RMW

强 A1 已拥有 96-wide full slices、1R1W psum 和 output-stationary execution，
每拍可发一个完整 product vector，达到 `P/96` 下界。PGPR products 与 A1 相同，
所以 product-issue 上限只能是 `1.0×`。`source_rmw_over_ideal_commit` 只能说明
源驱动 dataflow 的 psum traffic/energy 机会；相同 destination commit lower bound
也属于强 A1，不能转写成 cycle speedup。analyzer 的 NO-GO 口径正确。

## P0-01｜输入身份链没有固定 trust anchor

analyzer 计算 `contract_start`，但没有要求它等于冻结 contract `e556743d...`，
没有要求 contract canonical path/schema/status，也没有逐项要求四层
K3/S2/P1/OP1/dilation1/group1/bias-null/shape identity。scatter 数学完全依赖这些
前提。

同样，任意自封存目录只要 JSON 写出通用 PASS status、相同 contract hash、
capture seal hash 和 40 个键，就可充当 `payload-verify-dir`；analyzer 不检查
verifier code SHA、verifier schema/claim boundary、M511 final attempt admission、
runner SHA `788d674e...` 或 exact sealed member population。record list 也先经 dict
comprehension 折叠，未先要求原始列表恰为 40 且键唯一。

影响：漂移或伪造的 contract/capture/verifier 三元组可以生成标为 H67 exact S10
的 sealed PASS，且几何可能不再满足被证明的 scatter 前提。

必修：新版本必须硬绑定 frozen contract SHA/path；完整验证 contract
schema/status、S10×4 module population 和全部 transposed-conv 属性；只接受完成
M511 final-attempt admission 的新版 payload verifier，并 pin verifier SHA/schema、
runner SHA、attempt seal 与 claim boundary；在构造 dict 前拒绝重复/多余 records。

## P0-02｜canonical 发布后失败不会回滚

成功尾部先 `os.replace(staging, output_dir)`，再调用
`verify_directory(output_dir)`。没有先验证 staging，也没有 `try/except`、唯一
quarantine 或 postcondition。若发布后 seal/member 校验失败，进程非零退出但
canonical 目录保留；仅凭目录存在的消费者可能误收失败事务。发布前异常还会
遗留 staging。

必修：写 seal 后先验证 staging；发布前预生成并检查唯一 quarantine；用
`published` transaction 包围 rename 与 post-publication verify；任一发布后异常
第一恢复动作必须原子移走 canonical，并断言 canonical absent + quarantine dir。
修改后产生新 analyzer SHA 并重审。

## S10/S100 口径

`OLD_INCLUDED_SCOPE_CYCLES=620,302,905` 来自旧 S100 included-scope ledger，
decoder 是新 S10 exact cohort。脚本没有把相加结果用于 PGPR/TDR 决策，且
claim boundary 明确 `mixed_cohort_system_sensitivity_only=true`、
`system_speedup=false`、`date_headline=false`，所以目前没有错误准入。

但字段名 `mixed_cohort_corrected_envelope_sensitivity` 仍含 “corrected envelope”，
容易被后续表格误读。它只能作为非准入敏感性，不能用于系统 share、Amdahl、摘要
或性能主表。真正 repaired envelope 必须在同一 cohort、同一 sample weighting、
同一 cycle model 下重跑全部 included scope + decoder。

## P1

1. 把 mixed-cohort 字段重命名为显式
   `nonadmissible_s10_decoder_plus_s100_included_scope_sensitivity`，并同时记录两边
   cohort/sample 数，避免 “corrected envelope” 被误用。
2. TDR 目前只有 transition product 计数，没有 canonical accumulation-order、
   weight scale、accumulator width/overflow 和逐输出 miter。这与当前
   `new_performance_rtl_authorized=false` 一致；若 ideal ratio 过 1.30 门，下一阶段
   必须先补 numeric miter 和 SRAM state cycle/energy，再决定 RTL。

## 裁决

数学结论可保留：PGPR speedup KILL，TDR 仅可根据 ideal product ratio决定是否
进入 state-aware cycle fast-kill。但 P0 非零，当前 analyzer 不准读取生产
payload、不准发布 result，也不准据此开发 RTL。`docs/359` 未修改。
