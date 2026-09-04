# M920｜M919/M912 C1 metadata-pipeline VCS source fresh hammer

Verdict: **PASS100（P0=0，P1=0）**。本次是 fresh、source-only 独立审计；没有运行 runner、VCS、simv、DC、PT、Formality、ICC2 或许可证查询，没有创建 attempt/result，也没有自行生成 M921 release。

## M918 四项缺口已关闭

1. runner 在所有 launch gate 之前 exact-SHA 校验现存 `docs/359_DATE终局冻结_20260813.md`，SHA 为 `dedde7ce...bdfc4`；不存在的旧路径不再出现。
2. attempt 创建前先递归验证固定 M920 目录的 manifest 与 outer seal，严格要求本 review 的 PASS100/status/source identities；随后必须验证独立双封 M921 release，并要求 release 精确绑定 M920 review、manifest 与 outer-seal 三个 SHA。M921 当前按设计不存在，因此当前包仍不能启动。
3. collision scanner 排除精确 ancestry 后，同时按 `/proc/comm` 与 NUL-separated argv token basename 检查 `vcs/vcs1/simv/dc_shell/pt_shell/fm_shell/icc2_shell/common_shell_exec/common_shell_exe`；不再漏掉 Synopsys `common_shell_exec -shell dc_shell` 形态，也不以 runner 文件名子串自匹配。
4. 新 SVA r2 只相对冻结 M912 SVA 增加 PF pop input、严格 `{pop,row}` 后继断言及对应 bind：`pf_pop < candidate_pop`，或 pop 相等且 `pf_consumer < candidate_consumer`。弱的 consumer-inequality property 已删除。

## 功能冻结重验

- M912 RTL SHA 仍为 `eef2...e53`，冻结 r2 SHA 仍为 `7260...0e1`，冻结 M863 TB SHA 仍为 `7835...e9d`。
- M912 与 r2 的 59 项顶层 port tuple 完全相同。
- metadata 边界仍为 active 55 bit、next 53 bit、PF 13 bit、debug 9 bit，共 130 bit，低于 512-FF 门。
- 没有新 1824-bit registered psum payload；1152-bit registered payload 仍只有 `slot0_data_q/slot1_data_q`。
- 平衡 selector key 仍为 `{invalid,original_popcount,row_id}`；六级 pairwise min 与全局 min 等价，equal-pop tie 保持低 row-id。
- cleanroom oracle 仍显式收费 `+2 cycle/task`；`active_ctx_primed` 和有 next-context 时的零 inter-row bubble 门保留。
- TB 保留 14 项 normal minima、P2 `1/2`、held-final、六类攻击各一次、九项延迟 debug oracle 与唯一 coverage/held/PASS token。
- compile 身份仍为 foundry `UNIT_DELAY`，没有 `+notimingcheck` 或 `+no_notifier`；runner 仅含一次 VCS compile 与一次 `simv -no_save`。
- M919 attempt、result 均不存在；M912 predecessor attempt/result 也未被本 hammer 修改。

## 授权边界

M920 只允许主代理据此创建独立的、双封的 M921 release。当前 `vcs_launch_authorized=false`；只有 M921 精确绑定下列 review/manifest/outer hashes 且 runner 再次验证全链后，才可消费一次 M919 VCS attempt。

Claim boundary：`functional_vcs_verified=false`、`timing_verified=false`、`cycles_measured=false`、`speedup=false`、`ppa=false`、`energy=false`、`paper_citable=false`。
