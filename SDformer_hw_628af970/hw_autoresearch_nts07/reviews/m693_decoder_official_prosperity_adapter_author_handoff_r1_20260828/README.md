# M693 official decoder adapter 作者静态交付

## 结论

M693 已按 M689 设计实现为新增文件，并冻结 M692 `GO_M672_STATIC_ADAPTER_INPUT_ONLY`、M686 canonical manifest 及顶层/runtime/weights 双封。纯静态单元测试与一个非方形小形状 polyphase 整数 miter 通过 `27/27`。

本交付没有运行完整 M672 workload，没有导入或执行官方 `Simulator.run_fc`，没有产生周期、倍率或 canonical M693 result。正式 240 个 D0/D2/D3 direct calls、80 个 D1 diagnostic calls及 80 个 D0 N128 miter calls，必须等 fresh static hammer 给出精确状态 `GO_M693_FULL_OFFICIAL_CPU_REPLAY__P0_0_P1_0`。

## 冻结文件

- runner: `hw_autoresearch_nts07/scripts/run_m693_h67_ep35_decoder_official_prosperity_iso_workload.py`
  - SHA256 `9b1f0adf72db9ddf496a753f10574e3983feb5e452abb8a81f7ba79c172fe64e`
- contract: `hw_autoresearch_nts07/contracts/m693_h67_ep35_decoder_official_prosperity_iso_workload_contract_r1_20260828.json`
  - SHA256 `e675a1a8d156d270ee729e1671d4a5b606626ee2957eacf89f478b3e937c4e98`
- tests: `hw_autoresearch_nts07/system_simulator/tests/test_m693_decoder_official_prosperity_adapter.py`
  - SHA256 `d7a1c7ccc37690d75d2ef22901dced8c0ed00e9b21036ed8c7ac3bacf3057ad8`

## 实现边界

- exact official subset 只有 D0/D2/D3，30 records，每条 4 phases；
- D1 的 exact scaled-binary mask 只是 opportunity diagnostic；因 folded miter 非 bit-exact，`exact_decoder_complete` 周期/倍率强制为 null；
- 每个 sample/module/phase/mode 直接跑真实 N；只有 D0 可额外做 direct-N384 vs N128×3 全计数器 miter；
- operator 名禁止 `_fc` 后缀，避免官方隐式 Conv2d/img2col DRAM 分支；
- 官方 product/bit 分母是同一调用集合的 ratio-of-summed cycles；
- phase 求和只能称 support-tile aggregation，不是 monolithic ConvTranspose2d latency、ours 或 system speedup；
- runner 需要 fresh review outer-seal SHA 环境变量、显式 CLI flag，且 review 必须反向绑定 runner/contract/test/M686/M692 SHA；
- 输出只能经 fresh staging、双封、atomic rename 发布；post-publish 验证失败立即 quarantine。

## 测试

```text
PYTHONDONTWRITEBYTECODE=1 /opt/anaconda3/envs/pytorch310/bin/python -m pytest -q \
  hw_autoresearch_nts07/system_simulator/tests/test_m693_decoder_official_prosperity_adapter.py
...........................                                              [100%]
27 passed in 1.63s
```

测试覆盖：strict JSON、path/symlink、顶层/nested seal 身份、M692/M686 40-cell payload 全量预检、popcount/tail、D1 负路径、direct-N 准入范围、ratio-of-sums、future authorization 反向 SHA 绑定、atomic non-overwrite 及小形状 exact polyphase miter。

## Claim boundary

`author_static_only=true`；`mapper_production_executed=false`；`official_simulator_run=false`；`cycles=false`；`speedup=false`；`ours=false`；`full_decoder_latency=false`；`system_speedup=false`；`energy=false`；`ppa=false`；`date_headline=false`。

M618/M672/M686/docs359 均未修改；docs359 SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
