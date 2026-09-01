# M1713｜ep34 S2 FC/patch 零成本乐观上限快杀

状态：`PASS_ZERO_COST_UPPER_BOUND__FC2_AND_ALL_STANDALONE_MODULES_NO_GO__FC1_PATCH_CONDITIONAL__NO_PERFORMANCE`

## 结论

以 ep34 live93 封存 capture 的 40 个样本为唯一数据源，以 **exact C2 zero-source skip 后仍存在的非零 source-product work** 为分母，再给 S2 一个不可实现的免费条件：候选对象剩余工作可被 100% 删除，且 metadata、误差、端口、bank/burst、流水、terminal 和 commit 全部不收费。

| 对象 | FC/patch 目标工作份额 | 完全免费删除的上限 | 达到 1.15× 至少要丢掉对象剩余工作 | 判定 |
|---|---:|---:|---:|---|
| patch embed | 57.2523% | 2.339307× | 22.7825% | 仅保留跨层 family 测试 |
| FC1 | 33.0056% | 1.492662× | 39.5190% | 仅保留跨层 family 测试 |
| FC2 | 9.7421% | 1.107936× | 133.8875% | **直接 NO-GO** |

32 个单独模块的乐观上限全部低于 1.15×，其中最高只有 **1.123992×**。因此 S2 不得以单层机制推进；FC2 即使把所有现存非零工作免费删光也过不了门。若后续 reduced-binary capture 成功，S2 只允许在跨层 patch family 或跨层 FC1 family 上做 paired AEE + same-resource replay。

## 数据完整性

- 封存身份：M1458/M1434 ep34 live93，checkpoint `4bbaf7fc...`。
- 人群：4 条 DSEC sequence × 10 sample，共 40 sample。
- 目标记录：patch 320、FC1 480、FC2 480；模块数分别 8、12、12。
- `execution_trace.json` 与 `unified_ordered_records.jsonl` 按 `(global_sample_id, name)` 交叉核对，activity 一致、nonfinite=0。
- FC/patch retained value payload 为 **0/1280**；所以本收据不虚构 epsilon drop rate，也不包含 AEE。
- exact C2 基线产品工作为 `1,164,297,535,416`，理想 K8×96 issue 代理为 `1,516,012,416` cycle-equivalent。

## 口径与红线

工作量定义为 `sum(input_active × dense_macs / input_elements)`；`input_active` 已是 exact zero-source suppression 后的非零 signed source。分母只含 FC1+FC2+patch，主动排除了其他全网工作，因此比系统口径更乐观。

这些比值是 **zero-cost Amdahl upper bound**，不是周期仿真、局部加速或系统加速。`paired_aee=false`、`cycles=false`、`traffic=false`、`energy=false`、`speedup=false`、`rtl=false`、`vcs=false`、`eda=false`。不得写入 DATE 性能主表。

可复核入口：

```bash
python3 hw_autoresearch_nts07/system_simulator/scripts/analyze_m1713_ep34_s2_fc_patch_zero_cost_upper_bound_fastkill.py --json
```
