# M460 / G8 H67 FFN residual 流式 capture preinput

结论：`PASS_M460_PREINPUT_AND_CPU_MICRO__REMOTE_NOT_LAUNCHED`。

本里程碑只冻结并本地验证入口；没有连接 A800、没有执行远程 idle snapshot、没有启动 GPU、没有训练。

## 真实 residual 边界

H67 使用 `MS_Spiking_Mlp`。12 个 FFN 的真实顺序是：

```text
sn1 -> drop1(p=0) -> fc1 -> BN1(no_running/current-batch)
    -> sn2 -> drop2(p=0) -> fc2 -> BN2(no_running/current-batch)
```

完整 `.mlp` 返回值才是 `F(x)`：它位于 fc2 和动态 BN2 之后、父 Swin block 执行 `ADD sew: y=x+F(x)` 之前。fc2 hook 仅保留 pre-BN2 L1 对照，不能用于 whole-token skip 判据。

模块名通过固定 stage/block 集合唯一枚举：

```text
sttmultires_unet.encoders.swin3d.layers.{stage}.swin_blocks.{block}.mlp
blocks = [2, 2, 6, 2]
```

每个 FFN 装 full pre、sn1、sn2、fc2、full output 五个 hook，总计 12 个模块、60 个 hook；真实运行还会核对四 stage 的精确 `[T,N,H,W,C]` 几何、BN running buffer 已禁用、每 sample 每模块恰好调用一次。

## tau 与输出

```text
rho = ||F_token||1 / max(||x_token||1, 2^-24)
```

- tau=0：只有所有 post-BN2 `F(x)` channel 数值精确为 0 且相关张量 finite 才计入。
- tau>0：主计数严格使用 `rho < tau`；`rho == tau` 与 `rho <= tau` 分开输出。
- 每次 FFN 调用仅把 channel-axis reductions 写为一个压缩 NPZ；不落完整 activation/residual tensor。
- `f_l1/f_l2_sq/f_linf` 提供局部注入误差项。网络输出界仍需独立的 downstream `L_tail`；本入口不把局部范数冒充 AEE 上界。

## 本地验证

CPU micro 构造全部 12 个同名 FFN、安装 60 个 hook 并完成 12 次调用。独立逐 token/逐 channel reference 对 `x` 范数、sn1/sn2 nnz、pre-BN2 L1、post-BN2 exact-zero/L1/L2/Linf、finite、rho 和 strict tau count 比对为 0 mismatch；同时构造了 pre-BN2 与 post-BN2 不同的数值，防止边界退化。

```text
PASS_M460_CPU_MICRO_12_FFN_HOOK_AND_REFERENCE
independent_reference_mismatches=0
```

exact-SHA dry-run 也通过，且 `gpu_touched=false`、`automatic_launch=false`。

## 未来远程条件

runner 默认只做 exact-SHA + CPU micro + 四次相隔 10 秒的连续 idle guard，随后退出且不启动。idle 要求 GPU compute context 为 0，且没有其他 train/eval/valid/profile Python 进程。

只有人工显式设置下列变量才会在四次 guard 与第二次 exact-SHA 检查后运行 capture：

```bash
cd /root/private_data/work/sdformer_codex/SDformer
M460_EXPLICIT_REMOTE_LAUNCH=1 \
  ./hw_autoresearch_nts07/system_handoff/run_m460_h67_g8_ffn_token_residual_s10_when_gpu_idle_20260826.sh
```

输出先进入唯一 `.partial.<pid>`，manifest 与 10 sample / 12 FFN / 120 record / 5,580,000 token 全部通过后才原子发布。

当前不得引用 skip-rate、Delta-AEE、cycle speedup、energy、PPA、system speedup 或 headline 数字。
