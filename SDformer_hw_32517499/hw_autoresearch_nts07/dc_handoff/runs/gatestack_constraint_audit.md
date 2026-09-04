# GateStack DC约束与环境静态审计

- 静态约束状态：通过
- RTL filelist：28 个文件
- dc_shell：缺失
- fm_shell：缺失
- vcd2saif：缺失
- yosys：可用

## 检查项

- 主时钟端口为clk_core：通过
- 500MHz探索周期：通过
- setup不确定度：通过
- hold不确定度：通过
- 输入延迟：通过
- 输出延迟：通过
- 输入转换：通过
- 输出负载：通过
- 最大扇出：通过
- 同步复位未误设false-path：通过
- filelist非空：通过
- filelist全部存在：通过
- 顶层在filelist：通过

## 结论边界

本审计不替代DC的 `check_design`、`check_timing`、无约束路径、目标库映射、SAIF注释覆盖率和Formality。当前环境只能完成交付准备与开放结构综合。
