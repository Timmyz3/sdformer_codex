# arXiv:2412.09209 · eCARLA-scenes

- uid/来源: `ARX-010`｜arxiv_2412.09209+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`；PDF §3–4 窗补齐）
- 题名: eCARLA-scenes: A synthetically generated dataset for event-based optical flow prediction
- 精读深度: 方法级（数据集/流水窗+PDF补齐）+依据：CARLA EBC+多城镇/天气；场景录制与回放改传感器位姿；预同步光流/灰度；eWiz 工具库；完整标定表可能截断

## 可继承 A
CARLA 驾驶向合成事件光流数据集 + 场景回放改传感器流水 + eWiz 编训工具——事件光流数据合同与仿真供数对照（借入≠X）。

## 强对照 B
MVSEC/DSEC 真值同步噪声/抖动；ESIM/MDR/BlinkFlow 随机传感器轨迹非车载；Blender 重光照门槛高。

## 可差分 X线索
eCARLA/合成数据≠lifting X；数据旁路，勿搬合成 EPE 当本地合同。

## 与 F1–F7 / Stage B 关系
F2弱相关（事件时间窗/光流数据）。数据旁路。不抢 Stage B。f_candidates含F2。

## 不可搬用边界
合成→真实仍有域差；CARLA 光学≠芯片；仅摘录+§3–4；禁止虚报已读完整标定。

## 可复用 idea 点
- 车载运动模式（进退转弯摇摆）作光流数据合同
- 场景回放改传感器位姿作低成本扩展模板
- 预同步光流+灰度作多模态监督旁证
- eWiz 损失/指标库作训评纪律 | 负结果只停该合成集替换

## 杀门建议
合成微调无真实增益或挤占 Stage B → 保持旁路。
