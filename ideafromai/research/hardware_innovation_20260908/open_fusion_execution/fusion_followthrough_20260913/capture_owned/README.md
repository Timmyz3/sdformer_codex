# matched dense四个独立推理帧捕获

**已完成并取回本机。** [index.json](index.json)为`complete=true`；四帧共775,680个实际S48及对应源门全部0差，首帧12个原整数/门字段与原`aee_sum`精确一致。部署32个常量数组完全一致。corner/interior源形状分别为(10,96,9,9)/(10,96,11,11)。本机复核见[local_verification.json](local_verification.json)，捕获日志见[capture.log](capture.log)。A800四次前向约10秒、峰值分配21,864,402,432B；这是捕获耗时，不是候选加速结果。

入口`capture_four.py`只执行既有matched dense stage320的四次完整前向，0训练。固定顺序与时间戳来自`BASE/motion/capture/frames.json`的Zurich0001–0004；每帧重置网络状态，旧native ep34的门/flow不作输入。

```bash
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /root/private_data/work/hardware_innovation_20260908/env312/bin/python3.12 -u \
  capture_four.py --root /root/private_data/work/hardware_innovation_20260908
```

`index.json.complete`是完成判据；未完成时不能将部分数组当作四帧已捕获。索引的`frames[].capture`给出四个NPZ相对路径，并记录时间戳、来源、原首帧核对和每帧检查。

数据沿用`SmallCapture`两halo格式：`window_geometry_json`记录source/gate/output坐标，`corner_`和`interior_`下有`I24`、`sn1_gate`、preview Z/raw/BN1、`sn2_gate`、`updated_I24`、`proj_gate`、`continuous_q24`、native projection/BN输出。新增`source_S48`与同halo I24同形状(T10,C96,H,W)，为int64容器中的signed48实际门前dot。

S来自`LiteralForward`调用`helper.observe('accumulator_ranges','As_I_Q24',value)`时的真实GPU张量；观察发生于原`write24`之前，不改变算子或顺序。CPU用实际I24与As独立重算S和门，**仅作捕获后的验证**。root执行证书必须由实际Machine源指令产生S，不能用此gold供数。

`deployed_constants.npz`为本父`matched_training/dense/stage320/deployed_constants.npz`，`live_parameters.npz`由既有WindowCapture导出实际preview/native参数；`BN_parameters.npz`、`deployment_metadata.json`保留原单遍动态BN合同。首帧对已有matched dense两halo所有整数/门端点及原`aee_sum`精确核对；只报告这四次forward，未新增825或独立质量总体。

执行使用已授权sd5ai A800，GPU预检空闲。只通过父代理已建立的SSH ControlMaster认证；脚本、日志及此目录不存凭据。没有训练、RTL、EDA或生产修改。

## 固定跨帧机会统计（离线gold）

[opportunity.json](opportunity.json)逐行列出interior源窗口前2像素、全部24个H8组的3次相邻转移，共72项。参考严格为前一捕获帧，并非实际缓存驻留模拟。使用实际S48和整数源程序cutoff，令m为含端点的安全整数余量（`>=`门：真时`S-K`、假时`K-1-S`），`L=sum(abs(As[t]))`，组半径取80项`floor(m/L)`的最小值；{64,256,1024}是一个固定三级编码。7680个选定S/门与源程序复核0差。

72项中exact-I24、gate80全相同、固定三级半径命中、未量化最大半径命中均为**0**。实际组`max|ΔI24|`为56,050–144,526（中位90,786.5），参考组最大安全半径为15–4,459（中位374.5）；三级码0/64/256/1024分别有6/20/33/13项。本样本未显示三级量化是漏掉机会的原因。

逐lane的gate10相同为**239/576=41.49%**，三次转移分别48.44%/37.50%/38.54%；但逐lane未量化半径命中仍为0/576，且每个H8至少一lane翻门。因而组最坏lane确实损失输出复用机会，但不能独自解释该各向同性证书的失败。此处半径仅为该L1余量证书允许的最大值（未利用signed24输入域截断），不是所有证书上界；gate80相同率才是本样本整H8精确源输出复用的oracle上界。统计不收费、不执行Machine、不供控制器使用，不外推为家族否定。
