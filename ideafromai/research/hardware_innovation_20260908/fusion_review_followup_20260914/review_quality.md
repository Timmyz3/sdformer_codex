本轮逐段审阅上一批有损分组、单rank、原型、时间保持的SV/CPU/GPU实现及原模型挂接；随后独立重编译并复跑，**没有发现改变旧AEE或主要周期结论的数值错误**。这不等于完成全网定点或ASIC验证。

本轮实际新增验证是 [audit_quality/run_review.py](audit_quality/run_review.py)：real_6、zero、poison_corner三个不同义务，全部9模式、有/无背压、冷/暖，共108命令；raw/J/I24各414720值通过，108条所有整数计数逐项复现旧记录。原396命令没有冒充本轮新命令。前向完整读过 `evaluate_sparse.py`、`model_access.py`、`reference.py`、`calibrate.py`、`lossy_r8.sv`、父模型loader及任务metric。

另由 [recompute_quality.py](audit_quality/recompute_quality.py) 直接逐帧读取旧valid结果与paired825，而非采用旧汇总数字：825帧、18序列、48152523有效像素的文件身份/有效像素逐一吻合。frame-mean分别为group-drop 1.351391218079484、rank-drop 1.331155075046491、group-hold 1.352029903373364、rank-hold 1.326947034648740；同组NB0 1.447936665574317，R8+I24 1.327635022607994。pixel-mean也独立复算吻合；四臂每帧I24及identity饱和计数均为零。本轮没有重新跑这四个825评价，证据是旧逐帧结果的独立复算。

重点判断如下。

- `{0,θ}`/权重吸收身份保留。实际门输入被检查为0/1，Q1为完整96×3×3，非零门只选择加权贡献。signed协议没有被误写为逐事件连续幅值MAC。
- GPU的每位置T10 reference复位及更新与CPU相符；group/rank保持用的是上次已近似状态，非偷偷更新到精确值。原型函数有损，原始z仍需完整形成。训练336窗来自thun_00_a_0002；验证没有用于重新拟合阈值或codebook。
- 消费者的J由实际FP32 identity按RNE转换；整数式为 `p*a + ((j+b)<<20)`，再RNE26、sat24并以F14进入真正下游。负数floor余数的tie-even处理正确；当前参数及p边界在signed64内。后置hook覆盖的是完整residual输出，helper读取检查对应I24；不存在TB把I24直接喂给producer。
- 原始BN/output计算在GPU被覆盖，故评价的是**已定义的R8部署学生**，不是原ep34浮点函数的位等价。后续网络仍有浮点执行及既定coarse-head选择；`fullnet_bittrue=false`标签正确。
- group相比rank周期少1.20%、group-hold相比rank-hold少1.74%，是**不同精度函数之间的成本点**。不能据这两点声称同AEE Pareto更优。25%训练阈值预算不等于相同验证质量；共享消费者并不能消除这种算法差别。
- 1.47的prototype十帧失败只停当前冻结codebook/one-residual端点，不证明全部LUT/原型方法无用。group四臂虽然通过用户当前“优于NB0”门，也没有强新颖性或完整网络硬件加速证据。

当前没有必要重复825或重新训练来核对这些已有数值。新增的Kronecker函数已单独启动新评价，不能继承这里的质量。需要后续论文比较时，rank与group应先在同任务质量边界比较；若新增电路只有约1–2%服务差且增加选择器，不应给它恢复标题身份。
