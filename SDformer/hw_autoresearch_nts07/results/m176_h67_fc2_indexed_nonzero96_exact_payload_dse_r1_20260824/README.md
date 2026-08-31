# M176 H67 FC2 indexed-nonzero96 exact-payload DSE r1

M176 对冻结 H67 ep35 的 120 个 FC2 payload 全量重放，评估一个 96-bit
indexed sparse-beat transport：零 bitmap beat 不进入前端；每个非零 beat 保留
absolute 96-bit beat index（硬件以 `index×12` 恢复 base-row）；每个 token 额外发送
一个显式 zero-payload EOT。因此模型不假设
较早的非零 beat 能预测未来，也没有免费删除 token termination。

## 结果

- payload：120 个，437,760,000 bytes，143,894,510 events，全部逐文件
  SHA/size/popcount 检查。
- raw96 beat：36,480,000；其中 zero beat 17,610,624（48.274737%）。
- nonzero beat：18,869,376；EOT：5,580,000；indexed descriptor 合计
  24,449,376。
- 相对 raw96，descriptor 数量下降 32.978684%，即 `1.492063x` 更少传输拍。
- K4 wall：raw96 `157,504,597`，indexed96 `144,146,504`，改善
  `1.092670x`。
- indexed96 K1/K4 wall：`424,060,394 / 144,146,504 = 2.941871x`。
- 四 stage K1/K4：`2.215418x / 2.947335x / 3.112231x /
  3.159226x`，全部超过 2x。
- indexed96 K4 wall 还比 raw128 的 `146,423,753` 少 1.555246%，对应吞吐
  高 1.579816%，同时 bitmap payload width 低 25%。

与 raw96/raw128 一样，K1 与 K4 都使用同一 indexed transport，避免只给 K4
免费跳零。beat 内 grouping 仍受 modulo-8 bank uniqueness 约束，未跨 beat 合并
events。variable-length recurrence 与独立 scalar edge model 做了 9,600 个随机
case，0 mismatch；固定 recurrence 另有 9,600 case，0 mismatch。

## 硬件含义与边界

下一 RTL 应在 M175 的共享分层 selector 基础上接受单调递增但可跳跃的 absolute
beat index，以 output-block extent 限定 stage0/1/2/3 的 4/8/16/32 beats，并把
显式 EOT 与 payload beat 区分。前一 token 尚未 done-accept 时再次 EOT 必须
fail-close；同拍 done-accept + EOT 则定义为合法的新 all-zero token。必须验证
zero token、跨多 beat gap、EOT backpressure、同拍 token rearm、descriptor
conservation 和 malformed/out-of-range index fail-close。

当前数字仍是 exact-payload analytic frontend boundary。indexed producer/event
memory、SRAM/端口交付、四路 weight response、M169 arithmetic、accumulator context、
BN2/residual、complete FC2、power 和 system speedup 均未包含，不能作为 physical
speedup 或论文 headline。`docs/359` 未修改。
