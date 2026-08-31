# M460R3 独立打铁评审

结论：**90/100，P0=0、P1=3、P2=2，GO_REMOTE_PREFLIGHT_ONLY**。

这只授权将 exact double-sealed bundle 放到冻结远端路径后执行 `--preflight-no-launch` 和只读 idle check。**不授权 GPU capture、训练或任何性能/精度主张。** 本评审没有 SSH、没有触碰 GPU，也没有修改 subject 与 docs/359。

## 独立通过项

- 三个 subject 根 SHA 与任务给定值逐字一致；outer-seal 文件认证 29-leaf launch manifest，29/29 叶哈希通过，且 outer seal 不在自身 manifest 内，无循环信任。
- runner 默认 `--preflight-no-launch`。缺失/错误 expected seal 均以 exit 2 在故意设置的无效 Python 路径被调用前失败。
- 正确 seal 在本地默认 no-launch 路径通过，并明确输出 `gpu_touched=false`、`remote_contacted=false`、`capture_launched=false`。
- 作者十类攻击复跑 10/10；评审另写 fake module、手写循环 reference 与不同十类攻击，再过 10/10。12 FFN/60 hooks 数值 mismatch=0。
- 12 个 NPZ 字段与 dtype 对上冻结合同；合同与 runner 的四个 receipt 字段一致。
- subject 与独立 hammer 均在 CPython 3.6.8 实跑通过。
- 本地验证冻结 Git commit/tree、21 个 critical file、checkpoint、30 个 S10 输入；host/repo/Python 路径均已冻结。
- M159 重新计算：FFN 份额 `205384111 / 620302905 = 0.33110293268737795`；达到理想系统 1.15/1.20/1.30× 所需整 FFN skip 分别为 39.3940% / 50.3368% / 69.6971%。

## GPU capture 前必须关闭的 P1

1. 远端 Python 包身份未冻结：需在 sealed remote preflight 中记录并封存 Python、PyTorch/CUDA、torchvision、NumPy、PyYAML、spikingjelly、timm、einops 的版本/构建身份，并只做 profile/import-origin 检查。
2. 未排除 untracked import shadow：需拒绝 import-visible roots 下的 untracked 文件，或断言并哈希所有实际 resolved module origins。
3. post-capture summary 的 advisory receipt 名称是 `summary_sha256` 等三项，而合同/runner 是 `launch_outer_seal_sha256` 与三个 `capture_*` 字段；capture 前必须统一并重新封合同/manifest/outer seal，再做 delta hammer。

P2 是冻结合同的 pre-seal 布尔仍为 false，以及 queue receipt 自身不在 payload double seal 内；二者不阻塞只读 remote preflight，但应在后续 reviewer receipt 收口。

本里程碑没有 oracle 数据、ΔAEE、可执行跳算、周期、能量、PPA、系统倍速或 headline 准入。
