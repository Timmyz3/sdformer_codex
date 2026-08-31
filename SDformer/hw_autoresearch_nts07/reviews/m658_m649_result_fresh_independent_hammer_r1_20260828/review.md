# M658｜M649 canonical result fresh independent hammer

## 裁决

`PASS_NUMERIC_RESULT__NO_GO_GLOBAL_SPLIT_CONFIRMED`，99/100；P0=0、P1=0、P2=1。

M649 canonical 的双封印、40-record 数值账和 `NO_GO_EXACT_TYPED_SPLIT` 均成立。这个裁决只准许把 M649 当作 decoder 输入类型的数值审计和负结果引用；它不准许 activation capture、cycle、speedup、RTL、VCS/DC/PT、energy/PPA、system 或 DATE headline。

## 身份与 population

- canonical 目录严格只有结果 JSON、完成回执和两层 seal；内层、外层 SHA 均通过。
- 独立重哈希 contract、checkpoint、config/source、失败态等 24 项输入，以及 10 个严格顺序样本的 event/flow/mask 共 30 个原始文件；checkpoint load receipt 为 `missing=0/unexpected=0/overlay_missing=0/overlay_unexpected=0`。
- 4 个 ConvTranspose2d 的名称、结构参数、weight shape/bytes/content digest 格式均闭合；40 条记录严格等于 sample-major `(s00,d0..d3),...,(s09,d0..d3)`，无重复或缺格。
- M511 one-shot 仍为 consumed，M511 canonical 仍缺失，失败 staging 仍严格只有 `FAILED.json` 与 `s00_d0` bitpack；首次 M649 prefetch 失败 staging 仍严格只有冻结 `FAILED.json`。`docs/359` 仍为 `dedde7ce...`。

## 独立数值重算

审计器没有信任 M649 的 full-tensor 或 typed-partition summary。它从每条记录的逐通道 `zero/one/nonbinary/nonfinite/integer` 计数重新加总，再复算 first2/suffix 与 prefix/last2 两种分区及所有 gate。

| module | 10/10 full binary | elements | zero | one | finite nonbinary | 结论 |
|---|---:|---:|---:|---:|---:|---|
| d0 | 10 | 46,080,000 | 37,783,828 | 8,296,172 | 0 | 全通道 exact `{0,1}` |
| d1 | 0 | 92,400,000 | 75,314,174 | 0 | 17,085,826 | 全局 typed split 的反例 |
| d2 | 10 | 185,280,000 | 153,544,434 | 31,735,566 | 0 | 全通道 exact `{0,1}` |
| d3 | 10 | 372,480,000 | 267,646,872 | 104,833,128 | 0 | 全通道 exact `{0,1}` |

d1 的 exact-binary fraction 是 `75,314,174 / 92,400,000 = 81.508846%`，但这些 exact binary 全部是 0；finite nonbinary 是 18.491154%，nonfinite 为 0。更关键的是，每个样本的 770/770 通道都至少出现一个 nonbinary，因此它不是“只有两个 flow 通道为 analog”的混合张量：

- source-ordered first2：240,000 项中 21,537 nonbinary；`[2,770)` suffix：92,160,000 项中 17,064,289 nonbinary。
- 旧 last2 诊断：240,000 项中 73,574 nonbinary；`[0,768)` prefix：92,160,000 项中 17,012,252 nonbinary。

两种切法都失败。故 M649 的全局 `NO_GO_EXACT_TYPED_SPLIT` 是正确裁决，不能通过阈值、coercion 或换 last2 布局绕过。

另有一条独立于 M649 JSON 的交叉证据：对失败 M511 保留的 576,000-byte `s00_d0` bitpack 直接 popcount，得到 839,586 个 1，与 M649 `s00_d0` 完全一致。

## P2 与精确边界

P2：canonical 没有持久化 hostname、Python executable/hash、PyTorch/CUDA 版本、GPU UUID/name/driver。已有证据包括 M656 冻结的绝对命令、M649 内嵌的 launcher/source/checkpoint/sample 哈希、40 条 CUDA record，以及 audit 前 1 次、每样本后 10 次、最终 1 次 synchronize。这个缺口不改变当前负面分类结论，但下一份 capture contract 必须加入双封 runtime receipt。

唯一合理的下一步是新立合同：只针对已测得 exact-binary 的 d0/d2/d3 设计 payload/cycle 路径，并把 d1 作为独立的 lossless numeric-representation 问题。新合同必须 fresh static hammer 后才能运行；本 review 本身不授权 capture、simulator 或 RTL，也不能把 d0/d2/d3 的局部事实写成 decoder speedup。
