# M856 — M851/C2 R24 recursive-seal source fresh hammer

## 裁决

**88/100，FAIL：P0=1，P1=0，P2=0。M851/R24 不得进入 true release，更不得运行 VCS。**

R24 新增的 recursive exact seal verifier 本身通过了完整出版流水和 16 类负向边界：15 个规则文件可以复制、双封、递归精确校验、`RENAME_NOREPLACE` 发布并在 canonical 端复验；文件/目录 symlink、源路径与发布路径 TOCTOU、缺失/额外/深度漂移、payload/manifest/outer 变异及目的碰撞全部 fail closed。

但是正式 runner 和 whitelist 的 receipt 名称不一致。runner 在第 298 行生成 `m851_c2_r24_recursive_seal_vcs_receipt_r1.json`，而 `r848.WHITELIST` 仍要求 `m848_c2_r23_whitelist_vcs_receipt_r1.json`；R24 的 `RESULT_MEMBERS` 又直接继承该旧 whitelist。正式流程到第 309 行 `stage-result-whitelist` 时会因缺少旧 receipt 确定性失败。作者测试用旧 whitelist 构造 synthetic work，因此正常路径的绿灯并不代表 runner 真正产出的 work 能出版。

## P0 独立复现

我构造了 runner 实际会留下的 15 个控制/日志文件：14 个公共文件，加 `m851_c2_r24_recursive_seal_vcs_receipt_r1.json`。调用正式 `stage_result_whitelist` 后稳定拒绝；同一函数期待的第 3 个成员仍为 M848 receipt。代码定位：

- `dc_handoff/scripts/run_vcs_m851_c2_r24_recursive_seal_exact_sha.sh:298`：写 M851 receipt；
- 同 runner `:309`：调用继承的 stage；
- `verif_m848/m848_c2_r23_whitelist_guard.py:65`：白名单写死 M848 receipt；
- `verif_m851/m851_c2_r24_recursive_seal_guard.py:46`：`RESULT_MEMBERS` 直接继承旧 whitelist。

若当前身份被 release，一次性 attempt 会在两次 VCS 都成功后再次消费，并在 `RESULT_STAGE_SEAL` 失败；不会产生 canonical result。该问题与 R22 的工具 symlink 或 M850 的 flat verifier 问题不同，是 R24 自身的真实 work/测试夹具人口不一致。

## 已通过的边界

- 独立 17 项矩阵全部得到预期：1 项完整流水 PASS，16 项攻击或 runner-shape 缺陷 REJECT。
- recursive verifier 精确要求 15 payload、2 seal 和 `attack/`、`equalbw/` 两个隐含目录；额外空目录也拒绝。
- stage copy 对每级目录/文件使用 `O_NOFOLLOW`，源 dev/inode/size/SHA 前后稳定；源路径替换被拒绝。
- 发布前替换已验证 source pathname 会在 canonical postverify 的 identity 比较处被拒绝；已有 destination 不覆盖。
- 继承的 flat verifier 字节未改，且仍拒绝嵌套 exact population。
- R23→R24 的 `compile_and_run` 与 attack/equal-bandwidth 命令/PASS gate 字节一致；五组周期仍为 `51/53, 131/133, 486/499, 1231/1246, 14/14`。
- M803 三个 RTL、SVA、两个 TB、两个 VCS filelist 和 DC filelist 全部匹配合同 SHA。
- 作者 7/7 测试通过，但存在上述 fixture coverage gap。
- source dry-run 返回 86，许可证查询、VCS compile、simv、attempt、result、quarantine 均为 0；正式 M851 人口前后均为 0。
- `docs/359` SHA 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 最小后继修复

建立新的 R25 身份，不修改 M803 RTL/SVA/TB/filelists/命令/seed/周期。R25 必须定义自己的精确 15-key whitelist，把唯一的 receipt 成员替换成 M851/R25 对应名称，并让 staging 与 recursive `RESULT_MEMBERS` 共用同一个常量；禁止继续调用硬编码 R23 whitelist 的 staging。测试必须从 runner 的真实输出集合构造 work，并验证 receipt 的 filename、schema 与 status 三者一致。刷新全部受影响 SHA 后重新做 source hammer。当前 M851 身份不能补写 release，也不能原地运行。

本审阅没有查询许可证，没有运行 VCS/simv/DC/任何 EDA，没有创建正式 attempt/result。
