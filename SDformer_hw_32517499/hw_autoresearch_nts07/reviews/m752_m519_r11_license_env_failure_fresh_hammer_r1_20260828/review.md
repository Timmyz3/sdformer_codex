# M752：M519 R11 license 环境失败独立打铁

结论：**PASS failure audit，100/100，P0/P1/P2=0**。R11 已经消费唯一 attempt，且必须永久保持 `FAILED_OR_INCOMPLETE_DO_NOT_CITE`；只允许另建 additive R12，不允许重跑、改名或覆盖 R11。

## 已证实的事实

- R11 quarantine 和 attempt receipt 均通过双封印。quarantine manifest/outer-seal SHA 分别为 `2f49a486...` / `e618f3ae...`。
- attempt 于 20:06:35 在第一次 K1 DC launch 前按合同消费；K1 preflight 的 outer seal 与 attempt receipt 精确相等。
- 只存在 `k1`。`k8`、`k1x8` 及其 preflight 目录全部不存在；canonical 与临时 work 目录不存在。
- K1 child 与 runner 均已结束。child/runner rc 都是 255；runtime monitor rc=0、resource latch=0，失败不是资源或碰撞所致。
- `dc.log` 的完整内容只有 `Fatal: Design Compiler is not enabled. (DCSH-1)`；没有 Tcl terminal、面积或时序证据。

## license 因果判断

直接故障是 DC 在执行 Tcl 前没有启用 Design Compiler。环境遗漏是**高置信主因，但不是被现场日志唯一证明的排他根因**：M748 发布的 exact `env -i` 命令只保留 PATH/LANG/LC_ALL 和两个 SHA pin；R11 runner 对 `SNPSLMD_LICENSE_FILE` / `LM_LICENSE_FILE` 的引用数为 0，所以 child 必然没有从调用环境继承这两个变量。当前交互环境为 `SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo`、`LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat`，并且 13 份既有 Synopsys `FM_INFO/env` 同时记录了这两个值。

本次 attempt 没有封存 child environment 或 license-server/feature availability probe。因此不能排除故障时刻 server 不可达、feature 暂时耗尽或本地 license 无效；R12 不能只补两项 env 后盲跑，必须先补 license preflight。

## 最小 additive R12

1. 使用全新的 R12 runner、contract、candidate、release、canonical 与 attempt 身份；精确绑定 R11 quarantine、R11 attempt receipt、本 M752 review 及各自双封印。R11 保持不可变。
2. future `env -i` 明确加入：
   - `SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo`
   - `LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat`
3. contract/admission 使用 closed keys 固定上述两个字符串，固定 `/opt/synopsys/Synopsys.dat` SHA `fc6e1face2ac...`，并由 runner 在 attempt 之前逐项验证。禁止从交互 shell 隐式继承或接受未知 license key。
4. 在资源 preflight 之后、attempt 发布之前加入 double-sealed license preflight：固定 `/opt/synopsys/scl/2025.03/linux64/bin/lmutil` SHA `e7e056cce4de...`，分别检查 `Design-Compiler` 与 `DC-Ultra` 的 server/feature 状态，保存原始 stdout/stderr/rc 和解析回执；任何不确定、不可达或无 feature 结果都不消费 attempt。
5. license status 不是 reservation，runner 的第一次真实 DC launch 仍可能遇到竞争；此时必须像 R11 一样 quarantine，不能重试或把失败当 PPA。不要加入无限等待或未封存的 `SNPSLMD_QUEUE` 行为。
6. 保持 R11 的 RTL/Tcl/filelist/SDC/DB、K1→K8→K1x8 顺序、资源/碰撞门和 PPA pass gates不变。R12 只修 license discovery/preflight 与新身份。

本评审没有运行 runner、DC、VCS、Formality、PT/PTPX 或 remote，也没有修改 R11 quarantine、attempt、runner 或 release。只运行了文件/进程核对和 `lmutil -v` 的无 checkout 版本身份查询；没有联系 license server。
