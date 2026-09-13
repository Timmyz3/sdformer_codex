# One-axis F(2,3) challenger

固定接口已封口：[REPORT.md](REPORT.md)、[资源合同](resource_contract.json)。

- 原八真实tile+五功能tile，52runs/199,680输出全绿。
- 同模块强direct178,948拍，one_axis305,875拍；不跨资源比较native mode6。
- signed18系数真实紧排/两行CR组装，输入与逆变换、纵向第二次V读取均付费。
- `one_axis_tile.sv` / `tb.cpp` / `prepare.py` / `run.py`：完整C96/N96/T10实施。
- `verify.py` / `checks.json`：恒等式、字布局与旧direct不弱化核对。
- 仅tile范围，未声称整帧；不再追加布局实验。

独立审阅：[REVIEW_ONE_AXIS.md](../novelty/REVIEW_ONE_AXIS.md)，无阻断。静态字节覆盖与动态半组对齐覆盖分别报告。
