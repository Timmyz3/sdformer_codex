# arXiv:2405.16790 · SCSim

- uid/来源: `ARX-013`｜arxiv_2405.16790+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`）
- 题名: SCSim: A Realistic Spike Cameras Simulator
- 精读深度: 方法级（仅方法摘录窗口）+依据：§III 仿真框架 Fig.2；噪声无关积分阈 ϕ；电路级噪声/SNEE；Rand scenes+标签；完整噪声统计表可能截断

## 可继承 A
尖峰相机电路级噪声建模 + 渲染→积分重置→尖峰流流水 + SNEE 噪声评估——尖峰传感前端数据合同与仿真供数对照（借入≠X）。

## 强对照 B
SPCS 无噪声/弱噪声；NeuSpike/SpikingSIM 仅暗电流简化；纯事件相机差分采样；无标签生成的裸渲染。

## 可差分 X线索
SCSim/尖峰相机仿真≠lifting 源字 X；数据集/前端旁路，勿搬重建 PSNR 当净服务%。

## 与 F1–F7 / Stage B 关系
传感/数据旁路；弱挂表示选择。不抢 Stage B。f_candidates空。

## 不可搬用边界
RHDD/WGSE 微调效果≠valid825；仅摘录窗；电路噪声参数≠本地事件阈值声明。

## 可复用 idea 点
- 积分达阈 ϕ+重置作尖峰相机像素合同
- SNEE 联噪声与尖峰流统计作前端标定旁证
- Rand scenes+多后端渲染作高速标签生成模板
- 与事件仿真器分簇勿混差分采样 | 负结果只停 SCSim 数据替换

## 杀门建议
仿真替换无主网增益或挤占 Stage B → 保持旁路。
