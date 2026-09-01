# M1692｜TSBG authority-shape repair successor 作者回执

## 状态

**SOURCE-ONLY PASS 100/100**。M1692 只新增可消费的 M1693/M1694 authority shape；不修改 M1668、原 M1669 或纠错回执，不授权 M1694、远端 launch、capture、GPU、attempt 或 retry。

## 修复内容

- M1693 review 使用 validator 实际读取的 `score` 字段、12-key exact identity 和 4-key exact authorization。
- M1694 release 固定新的 result/attempt/work/failure namespace，并 exact 绑定 `ssh.sd5ai.scnet.cn:10037`、user `root`、repo `/root/private_data/work/sdformer_codex/SDformer`。
- child interpreter 同时验证固定路径和 release 中的 SHA/regular-file identity。
- capture 仍经 M1668→M1647→M1624：runtime handoff、current checkpoint/config/profile entity、parent/child `build_runtime`、GPU lease、O_EXCL attempt、one child/one capture/no retry 未放宽。
- 新结果固定生成 `m1692_clean_child_receipt.json`，其 schema/status/identity 包含 source、contract、release、runtime、checkpoint/config/profile 与 M1669 correction；必须经新的不同作者 result hammer 才能进入 DSE/AEE。

## 回归

CPython 3.6.8 与 3.12.13 输出逐字节一致，各 21/21。覆盖 exact review/release 正例、11 个 review/remote/interpreter/release/order 负例、parent/child runtime 顺序、lower GPU/O_EXCL 顺序、context 恢复和 capture-receipt evaluator identity。

本回执未连接远端、未读取 checkpoint、未运行 capture/GPU/EDA、未写 production attempt。下一步只允许不同作者 M1693 审阅。
