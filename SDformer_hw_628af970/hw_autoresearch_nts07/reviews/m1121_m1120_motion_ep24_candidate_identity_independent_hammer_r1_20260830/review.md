# M1121 independent hammer — M1120 motion ep24 candidate identity

结论：**PASS candidate identity；仍不是 final checkpoint，也没有硬件 replay 准入。**

通过已有 SSH control socket 进行了只读重算，没有下载 checkpoint：

- `checkpoint_epoch24.pth`：`225,504,447 B`，mtime `2026-08-30 05:04:00 +0800`，SHA256 `1e55900cd0bb4e411d09a5e4cd7bd56c08c60874a1e4868f6494d18b3e691e32`；SHA 前后两次 stat 相同。
- 配置 SHA256：`c7b5b994cb9f9a43478f3cb7c09e52a7aecf529fcd6a590f982a291e9eeed955`。
- 独立解析 `train.log`：epoch23 validation loss `0.8988656344867888`；epoch24 为 `0.8975050449371338`；epoch24 validation stats 为 `15.40 s / 0.7335 s / 2.7268 sample/s`；epoch25 已在第 6637 行启动。

日志在训练期间继续追加，因此 hammer 观察到的 log size/mtime 高于 M1120 作者观察值；历史 epoch23/24 行和 checkpoint identity 没有变化。

本地 contract 与 receipt 双封、严格成员覆盖通过。duplicate key、NaN，以及 final-checkpoint、valid825、hardware replay、intermediate replay authority、hardware/system speedup、accuracy、checkpoint-downloaded 八类 claim 升格共 10/10 全部拒绝。

本 hammer 没有远端写、GPU 干预、checkpoint 下载、hardware replay 或源文件修改。ep24 只能称“目前观察到的最佳已保存 candidate”；必须等预声明的最终保存点完成，再冻结 final identity、跑 valid825，并重绑全部 checkpoint-dependent 硬件证据。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
