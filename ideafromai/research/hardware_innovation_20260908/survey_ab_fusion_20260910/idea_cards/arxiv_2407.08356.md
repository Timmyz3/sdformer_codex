# arXiv:2407.08356 · Event-based vision on FPGAs (survey)

- uid/来源: `ARX-012` / `MAIN-R276`（同文别名）｜arxiv_2407.08356+本地excerpt（`p0_excerpt_batches_gap/gap_04.json`；PDF §III 窗补齐）
- 题名: Event-based vision on FPGAs -- a survey
- 精读深度: 方法级（综述方法窗+PDF §III 补齐）+依据：2012–2024 FPGA/SoC 事件处理分类（滤波/光流/立体/检测跟踪/SNN-AI/机器人）；指标口径 MEPS/延迟/器件；非单算法实现

## 可继承 A
FPGA/SoC 事件处理图谱：滤波→光流/立体→经典与 SNN-AI 检测跟踪——边侧事件加速对照轴与空白区定位（借入≠X；综述）。

## 强对照 B
仅 GPU/CPU 事件流水；单一任务点论文无跨域对照；忽略功耗/延迟联合口径。

## 可差分 X线索
综述≠lifting X；只作地图与缺口，不导出单芯片标题差分。

## 与 F1–F7 / Stage B 关系
地图/缺口旁证；弱挂硬件执行岛。不抢 Stage B。f_candidates空。

## 不可搬用边界
各文 MEPS/mW 不可直接并表外推本地；预印本未纳入；协作署名噪声；仅摘录+§III 窗。

## 可复用 idea 点
- 按应用轴（滤波/光流/立体/AI/机器人）分簇对照 FPGA 事件工作
- MEPS+延迟+器件三联作汇报口径模板
- 知识缺口清单作第二队列选题过滤器
- 负结果只停「用综述数字当本地合同」

## 杀门建议
若仅引用综述数字无法落到同端口合同 → 停当证据，保留地图。
