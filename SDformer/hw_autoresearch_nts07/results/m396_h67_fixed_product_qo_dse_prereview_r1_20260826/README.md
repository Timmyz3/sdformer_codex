# M396-pre：H67 fixed-product q/O DSE 独立预审

结论：**修复两处合同问题后 GO；原始四点合同原样执行 NO-GO。** 评分
84/100，P0/P1/P2 = 0/2/5。本里程碑只冻结下一次 H67 DSE 的公平执行
合同，不产生 speedup，也不选择赢家。

两个 P1 都能在执行前修掉：

- q128/O1 的 useful PWP 槽是 `144 B/center`，不是 32 B 对齐。直接使用
  `base+center*144` 会让奇数 center 落在半拍地址，奇数长度 run 也会有半拍
  长度。合同因此区分 useful payload 与 physical stride：前三档不变，q128/O1
  物理 stride 冻结为 160 B，16 B padding/selected center 必须计入容量和 DMA。
- M394 是 q32/O4 专用模型，不能只改两个参数就扩成四档。serial16 matcher 必须
  使用 `3000+(ceil(q/16)-1)*eligible+2`；pattern/config 传输、32-bit bitmap
  seal word、q-specific run 和 `tile_count=8/O` 都必须逐档计费。

修约后的容量全部安全：

| q/O | tiles=8/O | config | weight/tile | useful PWP/center | physical stride | worst slot | 32 KiB headroom |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16/8 | 1 | 32 B | 12,288 B | 1,152 B | 1,152 B | 30,752 B | 2,016 B |
| 32/4 | 2 | 64 B | 6,144 B | 576 B | 576 B | 24,640 B | 8,128 B |
| 64/2 | 4 | 128 B | 3,072 B | 288 B | 288 B | 21,632 B | 11,136 B |
| 128/1 | 8 | 256 B | 1,536 B | 144 B | 160 B | 22,272 B | 10,496 B |

`q*O=128` 只使每 tile 的 worst-case useful PWP 容量恒为 18,432 B；它不等于
等面积。q 越宽，serial16 的额外 eligible-row pass 为 0/1/3/7，pattern DMA
beat 为 1/2/4/8，bitmap seal word 为 1/1/2/4；O 越窄，完整八个 output block
的 replay 数为 1/2/4/8。descriptor SRAM 继续冻结为 48 bit、II1、L8、D8，
7-bit center 字段不缩窄，避免靠 descriptor 位宽变化取得不公平收益。

非空 phase 的最小递推冻结为：

`config_dma + matcher + bitmap_seal + tile_dma + (tiles-1)*max(replay,tile_dma) + replay + tail2`

其中 `replay=O*(correction+2*pwp_rows)+L8`；pattern 每 phase 一个 cmd32，
每 tile 一个 weight cmd32，再加 q-specific used-center bitmap 中每个 maximal run
一个 cmd32。只有后继 tile DMA 可以与当前 replay 在两个 32 KiB slot 间 overlap，
首 tile DMA 和末 tile replay 都 exposed；禁止 cross-phase overlap。

公平 baseline 必须是 M394 同一个 H67、八 output、SHARED96 bit-sparse oracle，
cmd32 总周期固定为 742,148,386，四个 q/O 行不得改变它，更不能把 baseline
错误地按 `8/O` 串行化。M358 的 PAFT 性能数也不得引用，只允许用它核对
ping-pong 递推与容量代数。

旧 `WIDE144_PWP_96_WEIGHT` 数字增加了独立 144-bit PWP port，把 PWP source
从 SHARED96 的两拍降为一拍；`SYSTOLIC_Q_II1` 则用全 q 并行搜索替换 serial16
额外 pass。它们既不是同端口，也不是同 matcher 资源；同时旧结果来自
PAFT ep4/running-BN M248，而非 H67 ep35/no-running M40。因此两类数字都不能
混入 M396 排名或对标。

下一执行必须先精确复现 M394 q32/O4 cmd32/L8 点，然后才允许比较四档；任何
input SHA、exact fallback、population、alignment、slot、command、replay 或
baseline mismatch 都 fail closed。当前 M384 仍固定 32-bit bitmap、576 B stride
和两次 replay，只能作 q32/O4 边界参考，不能把未来 DSE 的其他三档称为已由
RTL 支持。

`docs/359` 与所有既有证据均未修改。
