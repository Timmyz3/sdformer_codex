# R19 CPU 准备交接

最终执行准备采用 `IMPLEMENTATION_FREEZE_V2.json` 与 `orchestration_cpu_v2.json`，绑定三个新源、5 项合成 CPU 检查及只读原 V2/V1 primitive。独立预审要求补核实际 `sys.modules` 的共享数学模块路径/SHA及导入函数对象身份，第二版只补这一来源检查并测试同名错误缓存拒绝，没有数学变化。首次候选源在 `pre_review_snapshot_v1/`、原 V1 freeze/CPU 保留，仅供追溯，不用于新执行。

实际原 MLP/PSN 小通道 fixture 已走 native、FF、CC、CC repeat、singleton 五次局部前向，真目标行数分别 12000、12000、1400、1400、2120；78BN观察及异常 finally 清理通过。没有完整模型、GT、质量或 GPU 执行。

远端准备仅需要根代理在独立审后生成 `inputs_v1.json`、`source_freeze_v1.json`，指向原 R13 依赖及 R16 V2 primitive，保留实际字节 SHA。`source_freeze_v1.json` schema 是 `flow_r19_compact_graph_source_freeze_v1`，`files` 为 path/SHA 列表，至少包含实际 runner、primitive 及其 `compact_current_mlp.py` 依赖；完整新源、CPU、协议、独立审与旧依赖应一起冻结。

```bash
python -B run_compact_graph.py \
  --inputs /absolute/path/inputs_v1.json --inputs-sha256 ACTUAL_INPUTS_SHA \
  --freeze /absolute/path/source_freeze_v1.json --freeze-sha256 ACTUAL_FREEZE_SHA \
  --plan compact122 --output /absolute/fresh/run_v1 --wall-minutes 20 \
  --preflight-output /absolute/fresh/preflight_v1.json
```

上述默认仅 CPU preflight，不创建 `run_v1`。根代理之后用同参数加 `--execute` 才会在原 GPU0 全局锁内加载 CUDA。独立 admission schema `flow_r19_independent_admission_v1`，status `ADMIT`，permitted_plans `['compact122']`，须绑定 runner_sha256、core_sha256、BN_mode `NATIVE_DENSE_CURRENT`。input schema `flow_r19_compact_graph_inputs_v1`；其他 path/SHA asset 键沿用 R17，primitive 改指原 `native_dense_bn_v2.py`。

输出 schema `flow_r19_compact_graph_v1`；graph_id 在顶层及 binding 为 `r19_W271482_all78current_s2b2_R16V2_compact`。`forward_records` 共122，`quality_endpoint` 真值120，两个 CC_repeat false；`endpoint_arm` 将重复明确指回CC。成功状态 `COMPACT_GRAPH122_COMPLETE`。10个数值检查=8个FF/native旧门+2个CC逐字节重放。只有独立实际审通过才能读取新GT，不能把重复计成新的质量样本。

所有端点四 heads；native/FF以及首CC/两repeat保留前缀/branch/PSN，compact同时保存packed原PSN和展开PSN。预计122NPZ/488head/98capture/586 raw arrays。失败原样归档 `INCOMPLETE_OR_NUMERICAL_FAILURE`，不续跑补齐或读前缀质量。R16的compact-to-expanded失败仍保留，本轮没有expanded参考，不宣称等价；最终质量属于新图自身。
