# M510 r1 独立静态打铁评审

日期：2026-08-27  
结论：`NO_GO__R1_SUPERSEDED__DO_NOT_EXECUTE_PRODUCTION_AUDIT`  
评分：**68/100**  
生产 audit 执行：**否**

## 裁决

r1 存在 1 个 P0：它没有将 H67 实际模型类和 `ATLIF → ConvTranspose2d`
顺序纳入 exact-SHA/fail-closed 证明链。因此 r1 不得运行，不得产生
`PASS_CONFIRMED` 收据，也不授权 exact trace 或 RTL。

## P0｜实际 MS decoder 类链未被固定

冻结 config 的 `model.name` 是 `MS_SpikingformerFlowNet_en4`。实际类链为：

`MS_SpikingformerFlowNet_en4`
→ `MS_Spikingformer_MultiResUNet`
→ `transpose_type = MS_SpikingTransposeDecoderLayer`
→ `x = sn(x); x = deconv(x)`。

这个顺序是冻结 decoder ATLIF active count 能够作为遗漏
`ConvTranspose2d` source count 的必要前提。但 r1：

- 只在 `SNN_models.py` 中检查通用 `UpsampleLayer` 选择 token；
- 只检查基类 `SpikingTransposeDecoderLayer` 的 K3/S2/P1/output-padding1 构造；
- 没有固定 `Spiking_STSwinNet.py` SHA（当前观测为
  `b8d969f9b91c292197dbe47c7b9a11803f10b7c604daaf911ed4bb5d00999b71`）；
- 没有断言 config 模型名、FlowNet/unet 绑定、MS transpose 类和
  `sn-before-deconv` 顺序。

所以 r1 的数字虽然与当前源码结果一致，但“这些 active 是遗漏反卷积的
source”并未被 r1 自身 fail-closed 证明。

必修：将 `Spiking_STSwinNet.py` 加入冻结输入，并对上述四级类链和顺序做显式
断言，然后另起 r2 重审。

## 条件通过的部分

1. **K3/S2/P1/output-padding1 几何正确。** 对每一维
   `o=2i-1+k`，`i=0` 仅裁掉 `k=0`；最后一个 input 坐标仍全部落在
   `0..2H-1`。因此只有 top/left 裁剪，二维容量为 `4/6/9` tap。
2. **S100 aggregate 装填数学可复现。** 独立计算得四层总下界
   `1,637,926,293,504`、总上界 `1,761,318,549,504` product/S100；
   dense exact 为 `78,848,509,440` product/frame。
3. **没有把 aggregate 界冒充 per-sample trace。** analyzer/contract/docs 都明确
   先除以 100 再报 per-frame 均值，并禁止 exact-coordinate/cycle/RTL 声称。
4. **旧分母降级正确。** `620,302,905 cycle/frame` 被重标为
   `included-scope 96-lane activity-weighted envelope`，不再称 strict full-network；
   以它直接得到的全网 Amdahl 结论必须重算。

## P1

- analyzer 只记录传入 contract 的 SHA，没有外部固定 expected contract SHA；
  r2 应配 exact-SHA one-shot runner，并禁止 canonical output 路径逃出
  `results/` 根。
- `not output.exists()` 到 `os.replace()` 之间有极小 TOCTOU 窗口；受控
  one-shot 运行下不是数值阻塞，但严格 no-clobber 应用独占锁或等价机制。

## 重审门

r2 只有在以下条件同时满足时才可 `GO`：

1. 新 analyzer 和 contract 以 exact SHA 相互绑定；
2. 固定并检查 `Spiking_STSwinNet.py` 及 MS decoder 顺序；
3. 保留 aggregate/per-sample 边界和 included-scope 降级；
4. 本评审的生产文件与 `docs/359` 继续保持冻结 SHA。
