三项不同有损接口已完成完整K864→R8→N96→FP32 identity/I24的小fixture RTL，以及各自diverse10。四个候选/强控制的官方825全部完成，四臂均通过同环境NB0严格门。AS2固定K4+1工作点十帧未过，净周期收益小于约定10%扩展门，停止本点。没有模型训练、格式/比例扫描、生产修改或EDA。

| 执行模式 | 暖计算总拍 | 冷配置+计算 | 比exact暖拍减少 | diverse10 AEE | valid825 AEE |
|---|---:|---:|---:|---:|---|
| 0 exact | 107044 | 134644 | 0.000% | 1.390331540132 | 1.327635022608（同环境封存值，未重跑） |
| 1 groupdrop | 97682 | 125282 | 8.746% | 1.393107727514 | 1.351391218079（过门） |
| 2 rankdrop | 98870 | 126470 | 7.636% | 1.379365260456 | 1.331155075046（过门） |
| 3 K4prototype_residual | 98813 | 126413 | 7.689% | 1.471894678700 | 未运行 |
| 4 zeroproto_rank | 93185 | 120785 | 12.947% | 1.714547411881 | 未运行 |
| 5 temporalgroup_fullrefresh | 100916 | 128516 | 5.725% | 1.386801509411 | 1.352029903373（过门） |
| 6 temporalrank_fullrefresh | 107612 | 135212 | -0.531% | 1.345721957989 | 1.326947034649（过门） |
| 7 temporalrank_signed13delta | 102596 | 130196 | 4.155% | 1.345721957989 | 1.326947034649（过门） |
| 8 temporalgroup_signed13delta | 100808 | 128408 | 5.826% | 1.386801509411 | 1.352029903373（过门） |

结果入口：[AS1组关停](AS1_GROUP.md)、[AS2原型残差](AS2_PROTOTYPE.md)、[AS3时间保持](AS3_TEMPORAL.md)、[周期与位宽](RESOURCE_CONTRACT.md)、[独立公式](EXECUTION_LEDGER.md)、[primary来源](SOURCES.md)。

本地复现使用Python3.12：`/opt/anaconda3/bin/python3.12 prepare.py`，`/opt/anaconda3/bin/python3.12 run.py`，`/opt/anaconda3/bin/python3.12 verify.py`。SV已提交独立文件；make_rtl.py随后strengthen.py可重建当前核心，make_wrapper.py重建共同消费者wrapper。NPZ大源保留本地，常量hex和frozen_parameters.json/parameters.json含整数实值。GPU入口evaluate_sparse.py，沿既有A800 env312、同合法模型及数据协议；不需要重建venv。

本目录已完成并冻结。四臂825、逐帧配对和进程退出检查见[最终质量报告](QUALITY_REPORT.md)、[quality_checks.json](quality_checks.json)、[final_process_check.json](final_process_check.json)。原监视器的PID命名空间错误已安全纠正，重复137帧mode6前缀未采用、不增加试验数。作者自评与保守误差界见[AUTHOR_REVIEW.md](AUTHOR_REVIEW.md)。不再启动本目录的新训练、质量或RTL任务。
