# M513 decoder PGPR/TDR analyzer 独立静态与数学复审 r2

日期：2026-08-27  
结论：`STATIC_GO__EXACT_M511_CHAIN__FASTKILL_ONLY`  
评分：**98/100**  
P0：**0**  
P1：**2**  
生产 capture / payload / analyzer / GPU / VCS / DC / DSE 实际执行：**否**

## 结论

r1 的两个 P0 已关闭。新 analyzer SHA：

```text
9790f62d7a3e8fa4ca0ab98947bc6bfb49ae4720bbfb075ec75cebcd3cf7e299
```

本 SHA 可在同一 isolated repo 中、M511 capture 与最终 payload-verifier 结果均
完成后，运行一次 canonical M513 fast-kill。授权只覆盖 exact product-issue、
destination contributor/traffic bounds 和 TDR ideal transition ratio；不授权
PGPR/TDR RTL、SRAM-aware cycle、energy/PPA、系统倍率或 DATE headline。

## r1 P0-01：输入 trust chain 已关闭

1. 四个参数先做 lexical canonical path 比较，contract/capture/verifier-output/
   M513-output 都只能落在 isolated hardware root 的固定位置，leaf symlink 在
   resolve 前拒绝。
2. contract 固定为 `e556743d...`；schema/status 以及 capture schema/status 都
   检查。
3. 最终 payload verifier 源固定为 `222d0402...`。其 r3 独立静态评审为
   98/100、P0=0，outer-seal-file SHA `74982f64...` 本轮复核通过。
4. verifier output 必须恰好只有 `m511_payload_verify.json` 和
   `RUN_COMPLETE.txt` 两个 sealed members；JSON schema/status、population、
   payload-only claim boundary、contract/capture identity 都精确匹配。
5. verifier JSON 必须声明 runner `788d674e...`；M513 还把当前固定 attempt
   目录的 final `SHA256SUMS` 与 outer-seal-file SHA 对回 verifier 已封存身份。
6. contract 四层逐项要求 K3/S2/P1/output-padding1/dilation1/group1、固定
   Cin/Cout/input/output/weight shape；runtime bias 必须为 null。最终 verifier
   `222d...` 已独立验证 runtime weight shape/content identity。
7. capture 和 verifier raw record list 都先要求长度 40，再构造 dict；dict 仍须
   长度 40 且键集合精确等于 S10×4 expected keys，因此重复、缺失、额外键都
   fail closed。

这套链足以把 M511 one-shot runner 的 SIGKILL admission、payload 全 bitpack
复核和 M513 几何前提传递到 fast-kill 输出。

## r1 P0-02：原子发布已关闭

输出先在同父目录 unique staging 中写 JSON/RUN_COMPLETE/seal，再执行
`verify_directory(staging)` 并要求 exact-two member 集合和 completion 文本；
只有通过后才 `os.replace` 到 canonical。unique quarantine 在发布前生成并检查
不存在；普通 postpublication verify 异常的第一恢复操作会把 canonical 原子
移到 quarantine。

若 SIGKILL 恰好发生在 publish 与 postverify 之间，canonical 仍是已经做过
exact-two preverify 的同一目录对象，因而不会出现部分写入或未封存 PASS；这与
M511 verifier r3 已审定的发布模型一致。

## 数学回归：全部保持

更新后的 synthetic wrapper pin 新 analyzer SHA，并复用 r1 已封存独立参考实现
（SHA `ffe84dff...`）。它不打开任何生产 contract/capture/payload：

- 25 组 `H,W=1..5` scatter 逐点等于 brute-force K3/S2/P1/OP1 reference；
- 24 组随机 T10 record 的 current/destination、XOR/rise/fall、products/cycles
  全部等于逐 channel/coordinate reference；
- 每个 sample 的 record 独立调用，previous 从 0 开始，所以 t0 语义正确；
- Cout `{384,192,96,96}` 全部为完整 96-lane slice，A1/TDR cycles 均为
  products/96；
- state 容量保持 input bitmap `870,300 B`、previous output
  `10,598,400 elements`，INT16 `21,196,800 B`、Acc24 `31,795,200 B`。

## 机制裁决保持

- **PGPR：KILL as speedup。** 强 96-wide 1R1W output-stationary A1 已达到
  `P/96` 下界；PGPR products 相同，product-issue 上限为 `1.0×`。
  source-RMW/ideal-commit ratio 只能作 traffic/energy opportunity，不能作 cycle。
- **TDR：只准 ideal fast-kill。** exact XOR/signed transition 产品数成立；若
  ideal speedup 低于 1.30× 立即 KILL。若超过，也只准进入 canonical-order
  numeric miter + state SRAM cycle/energy model，仍不直接准 RTL。

## S10/S100 混合口径

字段已改为
`nonadmissible_s10_decoder_plus_s100_included_scope_sensitivity_cycles` 和
`nonadmissible_decoder_share_mixed_s10_s100_sensitivity`；model 同时记录 decoder
cohort=10、included-scope cohort=100，并写死
`mixed_cohort_sensitivity_admitted=false`。它没有参与 PGPR/TDR 判决，仍禁止进入
系统/Amdahl/性能主表。

## P1

1. TDR 的 “exact” 目前只指 transition product count；浮点/定点 canonical
   accumulation order、weight scale、accumulator width/overflow 和逐输出 miter
   尚未证明。现有 no-RTL claim boundary 已正确隔离；仅在 ideal ratio 过门后补。
2. postpublication quarantine 使用原子 `os.replace`，但没有显式断言 canonical
   absent + quarantine directory present，也没有给 prepublish staging failure 写
   FAILED marker。原子 rename 成功即已保证 canonical 被移走，因此不阻塞；可在
   通用 publication helper 中统一加后置条件和失败回执。

## 唯一运行条件

必须由后续 exact runner 硬绑定本 analyzer SHA、contract `e556...`、verifier
`222d...`、runner `788d...` 和本 r2 review outer seal；四个 canonical 输入/
输出路径必须原样传入且 M513 output 不存在。禁止手工直接替换 payload-verifier
目录或跳过 final attempt receipt。

`docs/359` 未修改。
