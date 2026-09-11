# arXiv:2508.12637 · HOMI

- uid/来源: `MAIN-R237`｜arxiv_2508.12637+本地excerpt（`p0_excerpt_batches/batch_05.json`）
- 题名: HOMI: Ultra-Fast EdgeAI platform for Event Cameras
- 精读深度: 方法级（仅方法摘录窗口）+依据：§III 多时钟域（传感器/预处理 5ns / RAMAN 7.8ns）；EVT 3.0 向量化解码动机；事件基/时间基控制与双缓冲；预处理 ALU+shift-scale→u8；部署 RAMAN（ASE 运行时激活稀疏+权稀疏+PE 阵列）；摘录止于地址生成/降采样叙述中段

## 可继承 A
端到端事件相机 EdgeAI：原生 EVT 解码→可切换事件/时间累积表示→稀疏感知 CNN 加速器（RAMAN ASE）流水；多时钟避免丢事件——系统级供数/表示/稀疏执行分母对照（借入≠X）。

## 强对照 B
USB AER 带宽瓶颈管线；仅直方图丢时间；重 ETS 指数时间面（高 LUT）；无稀疏引擎的稠密 CNN FPGA；靠丢事件保实时的单时钟设计。

## 可差分 X线索
HOMI/RAMAN 平台≠ lifting 结构化执行对象 X；本地不把边缘手势/分类 fps 写成 Stage B 净服务。

## 与 F1–F7 / Stage B 关系
旁路（事件前端+CNN 边缘平台）；与 SSER/Spiking Patches 等表示卡并列。不抢 Stage B。

## 不可搬用边界
IMX636+Ultrascale fps/资源 ≠ θg/T10 合同；RAMAN 引用稀疏引擎细节以摘录为准；方法窗在预处理中段截断，完整训练/应用结果可能缺失。

## 可复用 idea 点
- EVT 3.0 bank 向量化「同 bank 打包少传」作源字/广播域压缩工程对照（F7 弱）
- 预处理快钟 vs 加速器慢钟→背压/不丢事件的多域合同
- 事件基（变时长）vs 时间基（定 fps）双控制作输入合同旋钮
- ASE 运行时激活稀疏作 ordinary 稀疏跳过强对照，不升标题
- 负结果只停「HOMI 管线替换本地前端」布局

## 杀门建议
对主岛无差分或仅系统集成无算法/执行新合同 → 保持平台旁路，不进融合标题。
