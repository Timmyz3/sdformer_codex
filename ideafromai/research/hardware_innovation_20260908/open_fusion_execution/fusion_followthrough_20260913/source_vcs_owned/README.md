# 共同源 RTL 的 VCS 固定五父验证

**VCS 已实际完成20个固定测试点，所有门、RF写回和计数精确通过。** 共同 `temporal_source.sv` 原样用于新 dense、contiguous34、lifting40，以及两项 dense/lifting40 源程序；每父为 corner/interior × ready/stress。共执行9,601,442周期，检查4,329,264个RF向量（34,634,112个signed48 lane）和1,939,200个门位，0diff。每测试点的周期、SR64/SW64次数、读写stall、RF写回数也与既有Verilator结果逐字段一致。

这是同一个源核的跨仿真器功能/握手/周期验证。没有修改原RTL或生产树，没有DC/PT/FM、综合、布局布线、PPA、频率或完整网络延迟结论；不能将VCS与Verilator的运行时间当作硬件加速比。源输入仍是既存真实I24 halo，结果不继承到preview/Conv2/PED全链。

## 工具与隔离

- VCS：`/opt/synopsys/vcs/V-2023.12-SP1/bin/vcs -full64`，实际编译器 `VCS V-2023.12-SP1_Full64`，build `Mar 04 2024 20:18:51`。
- `27030@ic.ismd-nemo` 的license server和snpslmd均UP；探测时VCSCompiler/VCSRuntime相关feature有99 issued、0 in use。实际编译和20次仿真均已成功，不仅是license状态查询。
- 全部Synopsys编译/仿真持有 `/tmp/date_dual_synopsys_same_uid_eda_queue.lock` 排他锁，顺序运行，结束后已释放；未使用licqueue抢占或重复失败战役。
- 前置默认 `vcs -ID` 曾选择不存在的32位linux编译器；明确 `-full64` 后版本核验成功。该平台探测记录保存在 `tool_probe.json`。**实际编译一次成功，20个仿真均首次通过**；没有访问或重试旧failed目录。
- 现有Verilator4.028仅用于新SV TB语法lint；剩余提示只有其不支持SV延时仿真的STMTDLY。本轮动态执行证据来自VCS。

## 原始输入与转换合同

原DUT与C++驱动分别为 [temporal_source.sv](../../stage_20260912/hardware/rtl_source/temporal_source.sv) 和 [tb.cpp](../../stage_20260912/hardware/rtl_source/tb.cpp)。本目录的 `tb_source_vcs.sv` 直接实例化原DUT，未复制、重写或修改其数据路径。

| 本轮父名 | 既有程序/金值目录 | 程序条数 |
|---|---|---:|
| dense | `breadth_20260912/source_execution/dense` | 275 |
| contiguous34 | `breadth_20260912/source_execution/contiguous34` | 127 |
| lifting40 | `breadth_20260912/source_execution/lifting40` | 224 |
| dense_two_term | `breadth_20260912/source_constant_probe/dense` | 144 |
| lifting40_two_term | `breadth_20260912/source_constant_probe/lifting40` | 138 |

前三父为既有matched320源，后两父是既有固定≤2 signed-power系数投影，没有新拟合或训练。所有父沿用 `stage_20260912/hardware/source_rtl_inputs` 的 ordinary 输入：corner为9×9×12=972个H8 tile，interior为11×11×12=1452个H8 tile。这里的ordinary表示共同输入捕获来源，不表示给lifting使用ordinary门金值；各父的门金值分别读取各自既有 `corner_gates.bin/interior_gates.bin`。

`prepare.py` 只做格式转换与往返验证：原program.txt的13个字段编码成128bit ROM字，负threshold保留signed48；原little-endian int32容器中的signed24输入转成32bit hex；各父原uint16门金值转成16bit hex。输入、程序、金值均未重新生成数值或重新编译图。完整来源、20个case参数与原Verilator预期字段保存在 `manifest.json`，转换结果在 `fixtures/`。

## SV TB 的比较与周期定义

TB移植原C++参考解释器与驱动顺序：96×8参考RF、signed48范围检查、floor quotient＋ties-to-even＋sat24、原门比较、10bit词收集。它先核解释器门与各父原捕获金值，再逐条核DUT `wb_valid/wb_dst/wb_data` 的8个signed48结果，最后核两个实际SW64门写。被阻塞写必须保持valid/address/data稳定。

ROM配置使用原cfg128接口。每tile输入按T10、288字节stride放入TB供数内存；SR64请求返回真实输入字节。ready模式请求/写一直ready，读响应延后一计数周期；stress模式为 `rd_ready=(cycle%32<24)`、`wr_ready=(cycle%32<28)`，响应延迟 `1+cycle%3`。这些规则与原C++ TB相同，没有改DUT以迎合结果。

周期计数排除reset和program加载，包括每tile的start边沿、请求/响应/等待、算术流水及最后门写，跨tile连续计数。SV的手工时钟延时用于事件推进，不能据其仿真时间推断工作频率。逐RF写回采用四态精确比较，可拒绝X或数值差异。

## 实际周期结果

| 父 | corner ready | corner stress | interior ready | interior stress |
|---|---:|---:|---:|---:|
| dense | 477252 | 528758 | 712932 | 789878 |
| contiguous34 | 294516 | 342138 | 439956 | 511098 |
| lifting40 | 433512 | 482113 | 647592 | 720193 |
| dense_two_term | 310068 | 373218 | 463188 | 557538 |
| lifting40_two_term | 281880 | 326626 | 421080 | 487906 |

表中20值与原Verilator结果完全相同；不表示VCS新增了源算法加速。父间服务差异仍对应各自既有程序，不改变此前数值/质量评价口径。

## 可复查文件与执行入口

- `results.json`：实际VCS测量、原Verilator字段、逐项比较、完整命令和每case日志路径，`complete=true,VCS_completed=true`。
- `summary.json` / `summary.csv`：20点总表及合计。
- `compile.log` / `compile_driver.log`：唯一一次VCS编译日志；`runs/<case>/simulation.log`：20次实际仿真；`run.log`：顺序与mutex获得/释放记录。
- `probe_tools.py` / `tool_probe.json`：工具与license核验，包含默认32位探测及明确64位成功信息。
- `prepare.py` / `tb_source_vcs.sv` / `run_vcs.py`：转换器、SV TB、互斥编译/执行器。所有新文件均在本隔离目录，原RTL、原program/input/gold/results未改。

本次顺序为 `prepare.py` → Verilator语法lint → `run_vcs.py` → `summarize.py`，Python使用 `/opt/anaconda3/bin/python3.12` 和 `PYTHONDONTWRITEBYTECODE=1`。执行器在已有 `results.json` 时拒绝自动重跑；每case的具体VCS命令已写入结果，复查不需要再次消耗license或改写本次记录。
