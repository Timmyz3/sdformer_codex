# arXiv:2412.11284 · Learning Normal Flow

- uid/来源: `MAIN-R233`｜arxiv_2412.11284+本地excerpt（`p0_excerpt_batches/batch_04_residual.json`）
- 题名: Learning Normal Flow Directly From Event Neighborhoods
- 精读深度: 方法级（仅方法摘录窗口）+依据：§3.1 每事件邻域 N(ek)→法向流；§3.2 VecKM 局部编码（无显式采样分组）；§3.3 径向+角向 motion-field 损失（用光流 GT 导法向流）；§3.4 旋转/缩放/采样增强；§3.5 旋转等变集成估不确定度；摘录止于 UQ 集成平均

## 可继承 A
点式局部邻域法向流 + 不确定度过滤——作「局部几何可估分量 vs 全光流」的运动合同与不可靠预测门控对照（借入≠X）。

## 强对照 B
平面拟合小邻域法向流；体素/帧式稠密光流网；无 UQ 的全量 per-event 输出灌下游；依赖专用 SNN 硬件才可部署的叙事。

## 可差分 X线索
法向流/VecKM ≠ lifting X；可作 F2 弱相关（局部完成即出预测），但不进主岛。

## 与 F1–F7 / Stage B 关系
F2 弱相关；运动前端旁路。不抢 Stage B。

## 不可搬用边界
自建 GT 光流监督与 egomotion 求解≠AEE 合同；仅方法摘录窗口（§3.6 求解器后半可能截断）。

## 可复用 idea 点
- 径向约束圆直径=GT 光流 + 角向防零解 的损失模板
- 旋转等变集成 σ 作「不可靠源抑制」门控（近 F1 保留集）
- 归一化相机坐标训练的迁移纪律
- 负结果只停「法向流替换本地光流出口」

## 杀门建议
UQ 过滤后下游无增益或召回崩 → 停该门控挂载。
