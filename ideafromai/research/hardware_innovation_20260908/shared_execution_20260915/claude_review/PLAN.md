# T5 有界独立复核

Claude目录只读；本目录保存静态结论、最小自有激励与未改DUT/TB的Verilator复现。无EDA、训练、GPU、生产/Git或旧remote会话。

核查范围：BF指数的signed24定义、全零/最小负数/负幂、区间证书与锁定冻结、阈值方向及48bit保存、元数据/位平面/BN阈值来源、25拍分母、外推杀门与新颖性。数值域问题区分已存四trace是否实际触发；不以合成反例否定其正gamma结果，不以局部供数比代替完整服务比。

最小测试：自有合法signed24/signed16、PN由A真实生成、signed48阈值，四模式复用原SV/CPP编译，独立dot/区间模型核判决及供数。另单列旧模型允许49bit而DUT只存48bit的反例、负gamma转换反例，以及“改为最短signed指数后e=0可含−1”的接口陷阱（后者不是原maxabs指数下的bug）。只复核存档四trace结果/阈值，不重算其大型FC1捕获。

可再生的 legal/、negative_gamma/、threshold49/、minimal_exp_zero/ 与 obj/ 不必纳入版本库。在本目录执行以下命令重建原封不动的 Claude DUT/TB；工具版本为 Verilator 4.028，Python 需 numpy，原捕获及原四 trace 存档须仍在 BASE 中。

```bash
task_base=/home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908
verilator -Wall -Wno-fatal --cc --exe --top-module cert_gate_core --Mdir obj "$task_base/claude_fusion_trials_20260914/t5_rtl/cert_gate_core.sv" "$task_base/claude_fusion_trials_20260914/t5_rtl/tb_main.cpp" -CFLAGS '-O2 -std=c++14' > build.log 2>&1
make -C obj -f Vcert_gate_core.mk -j2 >> build.log 2>&1
/opt/anaconda3/bin/python3.12 check_t5.py > test.log 2>&1
/opt/anaconda3/bin/python3.12 check_t6_envelope.py
```

父任务新增 cert_transport 的结果齐后，可执行 `/opt/anaconda3/bin/python3.12 check_cert_transport_billing.py`。它只读既有 CSV/summary，不编译或运行 RTL，并拒绝旧25拍FX收据。
