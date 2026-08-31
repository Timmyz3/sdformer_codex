# M427：M426r3 dual-bank / seed-fusion 独立打铁

结论：**86/100，P0/P1/P2 = 0/3/3**。允许进入 standalone RTL + Synopsys VCS/DC/Formality，但只能作为 **throughput–area/storage Pareto** 继续；现阶段禁止把 `1.695794×` 直接当作资源公平的主性能结论，更不能上升成 full-network/system headline。

## 独立复算

M427r2 未 import 或执行 M401/M418/M419/M426 analyzer，直接解码冻结 M410R2 的 51,840,000 行 memh，并读取 M401 的 17,280-phase ledger。结果如下：

| 口径 | cycles | 相对 one-port strong zero |
|---|---:|---:|
| one-SHARED96 strong zero | 742,148,386 | 1.000000× |
| M401 serial low8/high4 | 641,790,704 | 1.156371× |
| M426 dual-bank co-read | 530,606,660 | 1.398679× |
| M426 seed-first-correction fused | 437,640,532 | 1.695794× |

复现 `exact_pwp=5,350,591`、positive residual PWP `11,620,766`、wide block `111,184,044`。co-read 节省量恰为每个 wide block 一拍；fusion 节省量恰为 `8 × 11,620,766 = 92,966,128` 拍；总节省 `204,150,172`，0 conservation mismatch。

对 M426r3 CSV 做了 51,840 行、1,296,000 个字段比较，variant/phase key 唯一且顺序完全匹配，timestamp/component mismatch 为 0。12-bit codec 对全部 4,096 个 lane 编码穷举，serial low + signed high 与直接 `{high4,low8}` 重建 0 mismatch。

`d=0` 保留一个 PWP seed；`d>0` 保留全部 `d` 个 correction issue，只把第一拍左操作数改成 PWP，未重复减 correction；fallback 仍为 population 个 source issue。该结论证明的是 issue ledger，不是 96-lane RTL 数值 miter。

## 关键资源公平性否决

M400 strong zero 是 one SHARED96，即一拍一个 96-byte source vector。M405 RTL 已经有独立 low 768-bit、high 512-bit 输入和 1152-bit contribution 输出，因此 M426 不一定增加容量或每-bank port 数；但它把原本 serial assemble/emit 改成同拍使用：

- co-read：144 logical B/cycle；若沿用 M405 padding，则输入信号为 160 B/cycle。
- seed-fused 的第一 positive-residual 拍：PWP 144 B + correction 96 B = 240 logical B/cycle；沿用 padding 为 256 input-signal B/cycle。
- correction 仍是一拍一个 96-byte vector，不是 correction rate 翻倍；新增的是 PWP 与 correction 的同拍并发，以及 reconstruction→mux→adder 的互连/关键路径。

独立等带宽敏感性复算：

| zero 参考 | peak read | cycles | fused/zero |
|---|---:|---:|---:|
| K1，`pop` | 96 B/cycle | 742,148,386 | 1.695794× |
| K2，`ceil(pop/2)` | 192 B/cycle | 435,149,895 | **0.994309×** |
| K3，`ceil(pop/3)` | 288 B/cycle | 335,550,364 | 0.766726× |

K2 自己需要两路 weight bank/adder/merge datapath，所以它是 optimistic lower-bound sensitivity，不是已经 admitted 的公平实现；但它的 peak 192 B 仍低于 fused 第一拍的 240 B，却已经消掉全部 `1.696×` 优势。故该倍率不具备 resource-normalized headline 资格。后续必须对 candidate 和 K2/common-area zero 同时做 RTL、macro port、DC/STA 和 power，画 throughput–area/energy Pareto。

12.507× / 15.164× 相对 weak dense16 只能保留为 secondary 列。

## 严重项

P1：

1. `1.695794×` 对 one-port strong zero 不是资源归一化比较；K2 sensitivity 为 `0.994309×`。
2. seed fusion 尚无 integrated RTL/value miter；row transport 不含 96-lane correction payload，必须验证 d=0/d=1/max-d/fallback/stall/reset/protocol attack，且不得重复 subtraction。
3. low/high/correction concurrent ports、1152-bit reconstruct、768-bit correction operand、mux/fanout 的 macro/interconnect timing、PPA、SAIF/PTPX 均未关闭。

P2：

1. 4,096 点 scalar signed12 proof 不能替代 96-lane accumulator width/cast/overflow proof。
2. high sidecar 是 48 logical bytes 还是 padded 64 physical bytes的 macro 组织未冻结。
3. 范围仍仅是 frozen H67 四个 bottleneck Conv3x3，不是 full network/system。

## Recovery 审计

M426 r1/r2 均在 phase 0 前 fail closed，目录只含失败 log/marker，没有 candidate JSON/CSV；r3 只增加 raw-row decode adapter，原 analyzer SHA 未变，因此没有利用候选数值反馈。M427 自己的 r1 因服务器 Python 3.6 不支持 `int.bit_count` 在 phase 0 前失败；r2 仅换为 `bin(value).count('1')`，输入、公式和目标均未变。

`docs/359` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
