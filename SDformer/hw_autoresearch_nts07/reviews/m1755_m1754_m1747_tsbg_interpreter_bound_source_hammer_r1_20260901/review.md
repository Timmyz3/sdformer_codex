# M1755 independent source hammer: M1754 interpreter-bound TSBG runner

结论：**PASS，100/100，P0/P1/P2 = 0/0/0。允许作者创建 M1756 一次性 release；本评审不授权执行。**

M1754 没有导入或修改 M1747。它先核验 M1747/M1748/M1749、已消耗失败回执以及未来 M1755/M1756 权限，再核验固定解释器及 Python/torch/numpy 版本，然后检查 M1747 result/work 为空，原子创建 wrapper attempt，最后用 `os.execve` 执行 exact-SHA M1747 `--run-analysis`。

## 已消费失败

失败回执三重身份完整且双封通过。唯一 M1749 调用由 `/usr/bin/python3` 3.12.3 发起，在 `checkpoint_fc1_betas` 导入 torch 时得到 `ModuleNotFoundError`，随后 fail-closed。六帧 traceback 逐项一致；checkpoint beta extraction、payload replay、result publication 分别为 0/0/0，M1747 result/work 均不存在，且 `automatic_retry=false`。

## 固定解释器与执行边界

M1754 静态绑定 `/opt/conda/envs/sdformerflow/bin/python3.10`、SHA256 `89520a3f...42aa0`、Python 3.10.20、torch 2.2.2+cu121、numpy 1.26.4。M1755 按合同没有访问远端，也没有启动该解释器；生产时 M1754 会在检查 namespace 和消费 attempt **之前**现场校验路径、regular-file SHA 和三项版本，任何漂移均 fail-closed。

结果目录内部仍由 unchanged M1747/M1749 生成；外部运行预算由未来 M1756 限定为一次 M1754 wrapper attempt。attempt 内的 launch receipt 在 exec 前绑定 M1755 review 与 M1756 release SHA。两层边界没有被混写成新的分析算法或论文结果。

## 独立打铁

独立 hammer 在 Python 3.6 与 3.12 各通过 35 项检查，62 个重封负向 mutation 全部拒绝，覆盖 review/release 的全部 identity 与 budget 字段、schema/status、重复 JSON key、未重封 release、symlink member、authority/preflight/namespace/attempt 顺序、重复 attempt 和 exact exec 参数。动态路径全部使用临时 namespace 和替代 `execve`；未访问网络/远端，未运行 capture、analysis、GPU 或 EDA，未写 production namespace。

## 权限

- 允许：创建严格绑定本 review 双封的 M1756 release。
- 禁止：本 review 直接授权 wrapper、analysis、capture、GPU/EDA 或 paper claim。
- M1756 必须保持 `wrapper_runs=1`、`interpreter_preflights=1`、`execs=1`、`analysis_runs=1`、`capture_verifications=1`、`result_publications=1`、`automatic_retry=false`，其余运行预算为 0。
