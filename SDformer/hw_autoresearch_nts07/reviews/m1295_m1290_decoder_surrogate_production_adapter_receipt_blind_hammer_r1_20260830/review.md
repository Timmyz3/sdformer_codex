# M1295：M1290 decoder surrogate production adapter 独立盲打

结论：**PASS，100/100，P0/P1/P2 = 0/0/0**。

本轮只审阅冻结的 M1290 source/test/contract，并在临时目录构造独立 synthetic result 与 hammer。没有读取 M1290 作者回执作为证据，没有打开 live prefix、canonical M1111DR2 结果或 canonical M1291 hammer，也没有运行真实 calibration、EDA、GPU 或远端任务。

## 通过项

- result 顶层严格限定为三个文件，nested seal 的 manifest、outer seal 与三个成员摘要全部闭合；future hammer 另有独立 manifest/outer，并绑定 result 的 manifest、outer 和三个成员摘要。
- 120 行投影严格闭合 3 条 sequence、30 个 sample、4 个 module、每层 30 个不同 observation；每行三个 digest、六类 kind summary、逐层固定 commit byte 均进入投影。
- 强制 `group_count <= active_source_terms <= 8 * group_count`，且 descriptor/weight/psum/compute/write 计数、traffic conservation、transaction ordinal 和 cycle interval 一起检查。
- fixture 只接受 exact Boolean `True`；`0`、`1`、`None`、`False`、字符串均拒绝。合法 fixture 仍强制 `analytical_cycle_annex=false`。
- production API 为零参数；调用者不能注入路径、SHA、PASS 字符串或布尔授权。

## 攻击重放

- 裸 SHA/PASS 伪 authority：拒绝。
- fully coordinated group/term/traffic/cycle 篡改并重封 result：由既有 different-author hammer linkage 拒绝。
- 巨大 commit 篡改并生成 fresh hammer：由 D0–D3 固定 commit 常量拒绝。
- term/group 越界并生成 fresh hammer：由整数界拒绝。
- 同层 observation 坍缩并生成 fresh hammer：由每层 30 个不同观测门拒绝。
- speedup claim promotion 并生成 fresh hammer：由逐行 claim boundary 拒绝。

## 准入边界

本回执只证明 M1290 source repair 经独立盲打通过。它不证明生产 calibration 已运行，不准入 analytical cycle annex、speedup、system speedup 或 paper PPA。生产准入仍必须打开并核验 canonical M1111DR2 result 与 future M1291 different-author hammer。

