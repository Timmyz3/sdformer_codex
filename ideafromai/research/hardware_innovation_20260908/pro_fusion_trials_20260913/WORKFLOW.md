# 方法说明

本阶段沿用 scientific-brainstorming 的明确假设、强反例、交叉审阅方法，并使用此前 deep-research 流程形成的原文来源。输入由用户和Pro/Grok候选锚定，不声称盲发散或human-first程序完全实现；没有自动矩阵选赢家。分解、时间表示和文献审查由三个独立子任务处理，root实现打包，再交叉阅读代码/结果。

实际硬件验证另列工具和数值范围：Python准备真实输入/配置，SystemVerilog执行数据路，C++ TB负责输入、背压和golden检查。方法打分与硬件测量分开，来源完整程度与移植完整程度分开。

Scientific Agent Skills 的程序来源：Kassis等（2026），[Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065)。引用方法来源不等于本研究已取得论文接收证据。
