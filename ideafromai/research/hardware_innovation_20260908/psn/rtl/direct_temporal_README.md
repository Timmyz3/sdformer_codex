# 整数时间类别源编码叶

`direct_temporal_code.sv` 实现一个标准源编码叶：10 个 signed12 输入，经三个整数阈值判决和 8 项映射，输出 3-bit 类别。一个 Acc24 加减链执行静态 CSD 指令，也负责三次 `−τ` 初始化；没有三个并列乘法器。输入样本 `t` 位于 `in_values[12*t +: 12]`。`τ` 是判决参数，输出的连续 `θ` 幅值由后继码本消费者保留。

已完成 Verilator 4.028 对照。3840 个真实捕获向量及 24 个定向向量，分别运行无停顿与背压模式，共 7728 次，零错误。只在开头复位一次，随后 128 次配置切换均不复位；覆盖 INT8 −128、精确阈值相等、常量判决、空程序行、输入停顿和输出背压。共实际经历 7904 拍输出背压、34 拍空闲输入停顿。结果见 `direct_temporal_result.json` 和逐组 `direct_temporal_results.tsv`。

参考生成器 `prepare_direct_temporal_cases.py` 用 **NumPy int64 普通点积** 计算类别，与 GPU 捕获的类别和原始三判决地址逐项核对，3840 项全部一致。C++ 对照再独立执行普通 int64 点积。CSD 编译过程中，每加入一个移加项就更新累计输入系数，以完整 signed12 输入域求该前缀的精确区间，包含初始 `−τ`；所有实际及定向程序都在 Acc24 范围内。这是整数范围检查与 Verilator 功能对照，不是 Formality 或 ASIC PPA。

| 模块 | 三行 CSD 项数 | 实测无停顿服务拍/向量 | 配置写入拍 |
|---|---|---:|---:|
| S2 B0 | 26 / 22 / 24 | 78 | 83 |
| S2 B1 | 25 / 29 / 25 | 85 | 90 |
| S2 B2 | 31 / 27 / 29 | 93 | 98 |
| S2 B3 | 23 / 24 / 27 | 80 | 85 |
| S2 B4 | 31 / 27 / 27 | 91 | 96 |
| S2 B5 | 22 / 27 / 30 | 85 | 90 |

每个向量指一个空间位置、一个通道的完整 T10 输入。无停顿连续服务为 `CSD 项数 + 6` 拍，包括输入接收、三次初始化、码表读取、输出握手；输入握手到输出握手的时间差为 `CSD 项数 + 5` 拍。程序与配置可跨向量复用，配置写入为程序项数加 11 拍。三个常量判决的控制路径仍使用 6 拍/向量。

RTL 声明的存储为：128×8-bit 指令 1024 bit，三行起址/长度/τ/标志 117 bit，码表 24 bit，共 1165 bit 配置；输入、累加器、指令及控制状态共 176 bit。另有一个加减链、0–7 位移位选择和输入选择逻辑。这些是未映射 RTL 的位数，不是 SRAM 宏或面积结果。

本叶接收已经量化并排列好的 120-bit 输入，未包含 FP32→signed12 量化、T10 输入生产/读取、层级广播、GP 权重与状态服务。上述拍数不能当作 GP 完整层或整网性能。输出背压时类别保持稳定；配置只允许在 `busy=0` 时写入。

复跑：

```bash
/opt/anaconda3/bin/python3.12 prepare_direct_temporal_cases.py
verilator --lint-only --sv --top-module direct_temporal_code -Wall direct_temporal_code.sv
verilator --cc --exe --sv --top-module direct_temporal_code --Mdir direct_temporal_obj -Wall -CFLAGS '-std=c++17 -O2' direct_temporal_code.sv direct_temporal_scoreboard.cpp
make -C direct_temporal_obj -f Vdirect_temporal_code.mk -j4
./direct_temporal_obj/Vdirect_temporal_code direct_temporal_cases.bin
```
