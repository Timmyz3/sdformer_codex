# M476 dual-slot parent queue 独立 hammer（2026-08-26）

## 裁定

**62/100，REVISE / NO-GO。发现 1 个 P0，禁止启动或引用 same-constraint 3.0 ns DC compare。**

Producer 的 exact-SHA VCS 结果本身可复现：manifest/outer seal、12 个输入、runner、VCS 工具、M475 hammer、M473 边界和 docs/359 全部通过；正式日志的14项 PASS count与9/9非空 cover也正确，fresh exact-SHA replay完全一致。但是 directed suite 漏掉了一个会产生 silent stale-parent corruption 的合法并发场景，因此 producer receipt 中的 `m476_micro_functional=true` 被本次独立评审推翻。

## P0：stalled final 与同地址 prefetch

当 final issue 因 `psum_write_ready=0` 被 stall 时：

1. `scratch_write_enable=0`；
2. 只要 queue 有容量，`prefetch_ready=1`；
3. 同地址 prefetch 因 `forward_match_w` 依赖 `scratch_write_enable` 而不匹配；
4. RTL 将它作为普通 `scratch_read_enable` 接受，读出提交前旧值；
5. 下一拍 final 可以写入新值，但已经接受的 response 把旧值用同一 parent ID 入队。

独立 Synopsys VCS 负测在冻结 RTL SHA `c5aa9d0c...` 上复现：

```text
REPRODUCED M476 STALE SAME-ADDRESS PREFETCH old=5 new=1 occupancy=1
```

这不是性能保守，而是 silent numerical corruption：后续 issue 的 slot0 ID 检查会通过，却消费旧数据。负测 testbench、compile log、sim log 和 receipt 已随本 review 封存。

Producer 的 `cp_output_stall` 与 `cp_forward` 分别非空，但没有交叉；原 TB 的 output-stall 段只调用 `issue_once`，没有并发 prefetch。因此9个 cover无法排除此P0。

## Producer 正式证据仍通过的部分

- VCS V-2023.12-SP1，top 为 `tb_m476_dual_slot_parent_queue_pipeline`。
- PASS counts：issues 6、rows 5、forward 1、reads/responses 4/4、dual-enqueue 1、full/full-consume 2/2、stalls 9、b2b 2、exact/partial 2/2、ID/overflow attack 1/1。
- 9个 SVA cover 全非空：forward 1、read/response 4/4、dual enqueue 1、queue full 2、full-consume 2、back-to-back 2、output stall 3、overflow block 1。
- 无 assertion/error/fatal；fresh exact-SHA replay count/cover相同。
- TB 的 ready edge 后在第一个 negedge 撤销 issue/prefetch valid，没有把同一个正常 issue 重复提交。5个 scratch与5个 psum结果各逐 lane 检查96 lanes。

这些只能说明已测试轨迹通过，不能覆盖缺失交叉。

## Queue 状态机审计

独立抽象状态机穷举了 reservation 规则的5个可达 `(occupancy, read_pending)` 形态：

```text
(0,0), (0,1), (1,0), (1,1), (2,0)
```

对16个合法 action transition 穷举 `pop / current response / new read / forward`：没有发现 `occupancy+pending>2`、slot overwrite 或 FIFO order loss。正确顺序是先pop旧head，再入队更早的macro response，最后入队本拍forward；full状态也确实不借同拍consume credit。

因此当前P0不是双槽容量公式本身，而是**尚未提交的同地址写与prefetch handshake之间缺少coherence interlock**。

## P1

1. `ap_queue_bound` 只检查两个valid位计算出的occupancy，近乎结构恒真；没有绑定 `occupancy+read_pending<=2`、enqueue不覆盖、compact next-state和ID/data FIFO order。
2. directed matrix仍缺 response+new-read、pop+forward、pop+response，以及ID/overflow fault时queue preservation等交叉。建议把独立16-transition表转为formal/directed matrix。
3. parent issue只能匹配slot0；slot1有目标ID时会fault。若这是ordered scheduler策略，必须进入full-controller合同，并把 `ap_parent_issue_has_head` 加强为accepted parent ID等于slot0 ID。

## 修复门

不能用“scheduler不会这样发”悄悄绕过，因为prefetch和issue是独立ready/valid接口，当前模块明确接受了该事务。可接受的修复必须做到：matching final仍valid但未commit时，同地址prefetch不得被当作普通read接受；要么hold/fail-close，要么建立显式待提交forward reservation。修复若重新引入issue→prefetch-ready路径，也必须在后续DC比较中如实体现。

修复后必须：

1. 冻结新RTL SHA；
2. 新增 `output stall × same-address prefetch` assertion、cover和data postcondition；
3. 重跑exact-SHA VCS与独立hammer；
4. 只有新hammer为GO，才允许same-constraint 3ns DC compare。

M473 performance、system speedup、DATE headline在任何情况下仍为false；M476修复通过也不会自动推翻 `PASS_M473_CPU_DSE_NO_GO`。

## 复核

```bash
python3 results/m476_independent_hammer_review_r1_20260826/audit_m476_independent.py \
  --root .
```

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
