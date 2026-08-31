# M523 r2/r3 独立静态打铁评审

## 结论

**GO，99/100，P0/P1/P2 = 0/0/0。** 授权 SHA-256 为 `60f85b104839cd8d340ba7592147117fc930d46742784c644a385a2ed470ece6` 的 r2 runner 做且仅做一次 Synopsys VCS directed functional 运行。本评审未运行 VCS/DC/Formality/PT/PTPX 或开源 RTL 工具，未改作者冻结文件或 `docs/359`。

## 身份与修复核验

- runner 是 mode `0755` 的 regular non-symlink，SHA 与 request/author handoff 一致。RTL、SVA 和 filelist 分别为 `ad6def...a0920`、`913605...77f7c`、`f2cc54...f8d0`，与 exit-31 失败运行字节一致。
- TB 唯一的 r3 改动是在 accepted-work drain loop 后、终局检查前增加第 358 行 `@(negedge clk_core);`。仅删除该行即恢复旧 SHA `1769ced4...bc31`；新 SHA 为 `3b468b72...ff5cc`。
- contract r3 已将 r2 过时的 `full8/tails1=3/3` 修正为 `4/1`，绑定 TB r3 和 failure-review outer seal。`descriptor_only=true`，direct C2/performance/energy/area/timing/PPA/system/headline 全部保持 false。
- author handoff、r1 static review 和 exit-31 failure review 的 inner/outer seals 全部通过，三个 review root 均无 symlink；`docs/359` SHA 仍为 `dedde7ce...dfc4`。

## 机械证据

- `bash -n` 通过；7 段 embedded Python 在实际 Python 3.6.8 全部编译通过，无 `Path.is_relative_to`。request 和 contract 均通过 strict-finite JSON 解析。
- 独立整数 oracle 得到 event fanout `4/6/6/9/9/9`、43 taps、bundle `8/2/6/8/1/8/8/2`、full8/tails1 `4/1`、cross-event `2`、phase `6/10/10/17`，与 TB/contract/pass regex 逐项一致。
- 独立穷举 8,352 个合法 FIFO head/count/pop/push 转移：容量、ring pointer 不变式、未弹出项与新写项不重叠全通过。
- r2 canonical/attempt/work/quarantine 在评审时均不存在。runner 只使用 r2 结果身份，无 r1 attempt 复用或 quarantine 升格。

## Runner 门禁

caller self-SHA/exit-10 是第一门；两组独立 review 必须 exact double-sealed。resource gate 在 attempt 前，attempt 在 VCS identity/compile 前。identity 和 compile 都使用 exact Full64 binary SHA `0735e4...96287`。compile/sim warning-error token、唯一 PASS regex 与 10 个 nonzero cover（包括 `cp_fault_drain_complete`）全部 fail closed。receipt 严格 finite；VCS 产物必须恰好只有两个已封存的 in-tree symlink。staging 在 rename 前验证，canonical 在 atomic `mv -T` 后再验证；失败保留 attempt 并转入 quarantine。

## 授权边界

本 GO 仅授权一次 M523 descriptor-only directed VCS。即使 VCS 通过，也不能据此声称 direct M218/C2 集成、decoder 加速、能量、面积、时序、PPA、系统倍速或 DATE headline。成功后还需不同的 receipt-blind 评审者验证 sealed result。

启动必须同时传入：

```bash
M523_R2_EXPECTED_RUNNER_SHA256=60f85b104839cd8d340ba7592147117fc930d46742784c644a385a2ed470ece6 \
M523_R2_EXPECTED_STATIC_REVIEW_OUTER_SEAL_SHA256=<this review outer-seal-file SHA256> \
M523_R2_EXPECTED_FAILURE_REVIEW_OUTER_SEAL_SHA256=b3c2ec802dc053c84e1154369ee23045724f303585a9ebea3260022b6b96b0ad \
  dc_handoff/scripts/run_vcs_m523_c2d_k8_polyphase_tap_bundler_r2_exact_sha.sh
```

## 残余假设

本授权基于评审时 r2 结果路径全部不存在，要求唯一启动前无并发修改。这是执行环境假设，不是已准入的性能结论。
