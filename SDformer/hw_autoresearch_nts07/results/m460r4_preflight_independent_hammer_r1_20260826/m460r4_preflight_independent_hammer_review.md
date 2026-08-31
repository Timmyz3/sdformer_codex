# M460R4 sealed remote preflight 独立打铁评审

结论：**94/100，P0=0、P1=0、P2=3**。M460R3 的三个 capture blocker 已在这个封存 preflight 里闭环。我仅放行下一步“构建新的封存 launch，执行一次 G8 S10 opportunity capture”；不放行训练、性能/精度结论或 headline。

## 独立通过项

- launch outer seal→14 个 leaf，remote-result outer seal→3 个 payload，全部 exact SHA 闭环。
- 17 个 contract identity、Git commit/tree、tracked runtime roots clean predicate 相互一致。
- Python/conda history、7 个 package、CUDA/cuDNN/driver 二进制和 29 个 import origin 与 freeze 精确相等。
- isolated sys.path 不含原远程脏树；shadow 逻辑能拒绝 exact module、namespace init 和 top-level critical package 注入。
- 30 个 S10 数据文件均为非软链接 regular file，bytes/SHA 全对；4 个 idle snapshot 间隔 10/10/11 s，GPU context 和 ML process 均为 0。
- 四个 capture receipt 字段名在 contract/capture advisory/runner 间统一；R4 runner 不暴露 capture mode。
- 独立攻击 16/16 通过；docs/359 SHA 未变。

## P2 与单次 capture 条件

1. 封存 receipt 只保存 untracked 数量和 rejected 空列表，没有保存全部 accepted 路径。因此单次 capture 启动时必须立刻重跑 exact inventory/import/shadow/data 预检，不得只信历史 receipt。
2. `clean` 只表示冻结 HEAD/tree + tracked runtime roots 无 diff；11 个 launch overlay 文件是 untracked 但已由 manifest 认证，不能写成全局 `git status` 为空。
3. preflight 用 `nvidia-smi` 做了只读 telemetry，也为身份校验按字节哈希 checkpoint。可证的是“无 CUDA context、无模型构造、无 checkpoint 反序列化、无 capture/training”，不要写成字面的 GPU/checkpoint 从未读取。

下一步必须新建而不是修改 R4 合同，同时绑定本评审的 launch/result outer-seal-file SHA；用全新原子输出目录只跑一次冻结 S10，结果和 launch receipt 双封。该 capture 仍只是 opportunity/oracle 证据，不自动准入 skip、ΔAEE、cycle、energy、PPA 或 system speedup。
