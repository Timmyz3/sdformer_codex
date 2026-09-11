# arXiv:2602.23204 · Motion-aware Event Suppression

- uid/来源: `MAIN-R232`｜arxiv_2602.23204+本地excerpt（`p0_excerpt_batches/batch_06.json`）
- 题名: Motion-aware Event Suppression for Event Cameras
- 精读深度: 方法级（仅方法摘录窗口）+依据：§III-A Anticipatory Motion Suppression＝当前 IMO/ego 二值掩码分割 + 预测光流 warp 掩码做前瞻抑制；贡献列 ATC 交叉注意多任务编解码、多视野预测、下游 token pruning/VO；摘录止于相关工作/方法开头，§III-C/D 网络与损失细部不全

## 可继承 A
任务驱动的「保留/抑制」事件掩码 + 短视野（≤100ms）运动预测前瞻过滤——作输入侧活动删减与多消费者供数前剪枝对照（借入≠X）。

## 强对照 B
稠密 3D/SLAM 再建分割；手调生物启发 OMS/阈值滤波；只做 IMO 分割不做事件抑制；无前瞻的事后滤波。

## 可差分 X线索
事件抑制/token pruning ≠ lifting 物理源字删字 X；可作 F1「删活动」上游合同，但差分须落到 r1 执行对象，而非 EVIMO IoU/173Hz 叙事。

## 与 F1–F7 / Stage B 关系
F1/F2/F7 弱–中相关（删事件、短视野共同完成、运动打包）；旁路前端。不抢 Stage B。

## 不可搬用边界
EVIMO IoU/+67%、173Hz、<1GB、ViT +83% FPS、ATE −13% 勿写成 same-port%；仅方法摘录窗口（ATC/损失截断）。

## 可复用 idea 点
- 「分割掩码 × 流预测 warp」作前瞻抑制两步合同
- motion-guided token pruning 作 F1/F7 下游消费者减负模板
- 动态事件 <5% 极端不平衡 → 杀门须报保留集召回，不只报吞吐
- 与 EventShiftFlow/光流旁路成对阅读
- 负结果只停「抑制前端挂本地主网」布局

## 杀门建议
抑制后 AEE 变差或保留集漏关键关键事件 → 停该前端挂载；不因此停 F1 删字家族。
