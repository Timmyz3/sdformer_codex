# M2067 FC2 exact-continuation — header handshake 诊断状态

**时间**: 2026-09-04（用户时区 UTC+8）  
**范围**: 只读诊断；**未**获准改 RTL/TB/wrapper/scripts/repo（本地或远端）。  
**Quarantine**: `m2067_ep34_fc2_exact_continuation_vcs_r1_20260903.failed_or_incomplete.quarantine`  
**状态**: `FAILED_DO_NOT_CITE_NO_RETRY` — **禁止**重跑该 identity/path。

---

## 1. 证据摘要

| 项 | 值 |
|---|---|
| Repo HEAD | `f35a9e3` |
| Fatal | `tb_m2067_ep34_fc2_exact_continuation_s960.sv:412` `send_header_both` |
| 仿真时刻 | `3016500 ps`（≈1000 个 3ns 周期后的 posedge，与 TB 1000-cycle wait 一致） |
| 首刺激 | `run_alias_attack(96,1,0)` → 第一帧 `send_header_both(96,0,0,2,0,1,0,0)`（G96 chunk0，几何/旗标/序列在纸面上合法） |
| Fixture slot0 | `source_groups=96`, `chunks=2`（stats.memh LSB 解析已核对） |
| 编译 | quarantine `vcs_compile.log` 成功；仅 KUAI/`context` 警告 |

Wrapper 接受条件（未改）：

```text
chunk_accept = chunk_valid && chunk_ready
            && header_geometry_legal && header_flags_legal && header_sequence_legal
```

非法且 `valid&&ready` → `FAULT`（wrapper ~470–472），之后 `ready` 保持低 → TB 超时。

---

## 2. 根因（有证据）

**根因：TB 在 posedge 上对单周期 combo `chunk_accept` 的采样竞态，而非首包几何非法。**

机制：

1. DUT `always_ff` 在 posedge 见 `chunk_accept==1` → `wrapper_state_q <= W_RESET_ASSERT`（NBA）。
2. NBA 后 combo 重算：`ready=0` → **`chunk_accept` 脉冲在同一拍内消失**。
3. TB `send_header_both` 在 `@(posedge clk_core)` 之后用阻塞读 `if (base.chunk_accept)`；若进程排在 DUT NBA+combo 之后（VCS 上易发生），**永远采不到 accept**。
4. `chunk_valid` 因未见到 accept 而不清；后续 `ready=0`，循环空转到 1000 → line 412 fatal。
5. 时刻 `3016500 ps` = 复位后约 1000 个周期，与「第一帧 header 从未被 TB 记为 accept」完全吻合。

**本地只读/scratch 烟雾（非 repo 写入）已复现该竞态：**

- 最小 stub 镜像 wrapper 合法性 + FAULT/RESET 状态机。
- 在 posedge 后 `#1` 晚采样：`accept=0 ready=0 state=W_RESET_ASSERT`，同时 **sticky always_ff 捕获到 accept=1**。
- 结论标签：`ROOT_CAUSE_CONFIRMED: late sample misses accept pulse; sticky caught it`。

排除/降级假设：

| 假设 | 结论 |
|---|---|
| 首包 geometry/flags/sequence 非法 | 低：字段 `(96,0,2,0,first)` 对 SOURCE_GROUPS=48 合法；若真非法会进 FAULT，但超时形态与「accept 脉冲被错过、DUT 已离开 W_HEADER」一致 |
| 复位前 header / inner 未 ready | 低：TB 先 `while (ready)`；超时在 accept 环而非 ready 环 |
| stats/fixture 漂移 | 否：slot0 groups=96 chunks=2；metadata drift fatal 未触发 |
| 需削弱 legality 才能过 | **否** — 协议 bug 在 TB 采样，不应改 fail-closed 检查 |

---

## 3. 建议最小补丁（仅报告中的 diff 文本 — **未落地**）

**只改 TB** `tb_m2018/tb_m2067_ep34_fc2_exact_continuation_s960.sv`。  
**不改** wrapper legality、不弱化 FAULT、不改 RTL。

要点：

1. 增加与 DUT 同沿的 **sticky accept latch**（`always_ff` 在 `chunk_valid && chunk_accept` 时置位）。
2. `send_header_both`：valid 拉低时先驱稳 header 字段一整拍，再拉高 valid；等待 **sticky**，不要晚采样 combo `chunk_accept`。
3. 超时/`protocol_error` 时 `$display` 字段 + ready/accept/sticky/alias（便于区分真 FAULT vs 采样问题）。
4. 可选同源修复：`load_descriptor_both`、`send_invalid_alias_header_both`（非法路径先 settle 再 valid，避免合法性毛刺）。

### 提议 diff（示意，非已应用文件）

```diff
--- a/tb_m2018/tb_m2067_ep34_fc2_exact_continuation_s960.sv
+++ b/tb_m2018/tb_m2067_ep34_fc2_exact_continuation_s960.sv
@@ - around side interfaces -
     m2067_side_if base();
     m2067_side_if tsbg();
+    // Sticky catchers: chunk_accept is a 1-cycle combo pulse cleared when
+    // wrapper leaves W_HEADER; post-NBA TB samples can miss it (VCS @3016500ps).
+    logic header_accept_base_sticky, header_accept_tsbg_sticky;
+    logic load_accept_base_sticky, load_accept_tsbg_sticky;
+    logic header_accept_clear=0, load_accept_clear=0;
+    always_ff @(posedge clk_core) begin
+        if (rst_core || header_accept_clear) begin
+            header_accept_base_sticky <= 0;
+            header_accept_tsbg_sticky <= 0;
+        end else begin
+            if (chunk_valid_base && base.chunk_accept)
+                header_accept_base_sticky <= 1;
+            if (chunk_valid_tsbg && tsbg.chunk_accept)
+                header_accept_tsbg_sticky <= 1;
+        end
+        if (rst_core || load_accept_clear) begin
+            load_accept_base_sticky <= 0;
+            load_accept_tsbg_sticky <= 0;
+        end else begin
+            if (load_valid_base && base.load_accept)
+                load_accept_base_sticky <= 1;
+            if (load_valid_tsbg && tsbg.load_accept)
+                load_accept_tsbg_sticky <= 1;
+        end
+    end

@@ - send_header_both -
-            while (!(base.chunk_ready && tsbg.chunk_ready)) @(negedge clk_core);
-            /* drive fields + valid=1 */
-            for (...) begin
-                @(posedge clk_core);
-                if (base.chunk_accept) begin base_seen=1; chunk_valid_base<=0; end
-                if (tsbg.chunk_accept) begin tsbg_seen=1; chunk_valid_tsbg<=0; end
-            end
-            if (!(base_seen&&tsbg_seen))
-                $fatal(1, "M2067 legal header timeout/reject");
+            /* clear sticky; wait ready; drive fields with valid=0; @(negedge);
+               then valid=1; wait header_accept_*_sticky; deassert valid;
+               on protocol_error/timeout $display fields+sticky+ready */
```

（完整任务体替换见会话内已拟文案；**需用户明确批准后再写入 server worktree**。）

---

## 4. 新 identity 计划（勿重试 quarantine）

沿 `m2068 → m2070 → m2072(r3)` 命名：

| M# | 角色 | 建议名 |
|---|---|---|
| **M2073** | 源合同 r4（handshake TB 修复 + 重钉 hash） | `contracts/m2073_m2067_ep34_fc2_exact_continuation_vcs_source_contract_r4_20260904.json`（或 `m2073_..._handshake_repair_...`） |
| **M2074** | 独立 source hammer | `reviews/m2074_m2073_m2067_ep34_fc2_exact_continuation_vcs_source_r4_hammer_r1_20260904/` |

纪律：

- **永不**重跑 `..._vcs_r1_20260903` / quarantine identity。
- Hammer **PASS 之前** 不启 960-slot 生产 VCS。
- 不重试 M2063 power；不声称 `paper_citable`。
- 生产 one-shot 仅在 M2074 PASS 后、**新** runner/attempt 名下（新 result 目录，非 quarantine 路径）。
- Wrapper legality **保持** fail-closed。

---

## 5. 本地烟雾是否可行（只读）

| 方式 | 可行？ | 说明 |
|---|---|---|
| 最小 iverilog stub（合法性+FAULT+TB 采样） | **是** | 已在 `/workspace/m2067_smoke/` scratch 复现晚采样丢脉冲；**不**改 repo |
| 全量 TB+wrapper+M2018 iverilog | 困难 | filelist 含 VCS/大 RTL；非必要 |
| Server 单 slot VCS | 需新 identity + 批准 | 修复落地后用**新** work 路径；禁止 quarantine 名 |
| Parser `--static` | 只读可用 | 合同/TB hash 变更后需新合同针脚 |

---

## 6. 已做 / 未做 / 合规说明

**已做（诊断）：**

- 读 quarantine `failure.json`、`slot_0000.log`、`attempt.json`、compile log 尾部。
- 读 TB `send_header_both`、wrapper header/FAULT、contract r2、M2072 hammer 综述。
- Scratch iverilog 竞态复现（`/workspace/m2067_smoke/`）。

**未做（按新规则停止）：**

- **未**向 ismd-nemo server worktree 写入/scp TB 或 RTL。
- **未**启生产 960 VCS；**未**重试 quarantine。
- **未**创建 M2073/M2074 正式合同/hammer 目录（需批准后）。

**注意（代理曾短暂越界，需用户知晓）：**

- 在收到「禁止未批准改文件」之前，曾在 **box** 上生成 scratch `tb_fixed.sv`，并可能 `cp` 覆盖过  
  `/workspace/sdformer_codex/.../tb_m2067_ep34_fc2_exact_continuation_s960.sv`。  
- **Server 远端 worktree 不应被本代理改过。**  
- 恢复 box mirror / 清理 scratch **也需要你的明确批准**（当前 STOP 下不再写 repo 路径）。  
- 请以 server `HEAD` + `sha256sum` 的 TB 为权威；box mirror 若被改过，勿当 source of truth。

---

## 7. 下一步（待你批准）

1. 批准后：仅改 server TB（sticky + settle），可选同步 box。  
2. 冻 M2073 r4 合同（新 TB sha）+ M2074 独立 hammer。  
3. Hammer PASS 后：新 identity 单 slot 烟雾 → 再考虑 960 one-shot。  
4. 在此之前：**不** launch 生产 VCS。

**Hammer 结果**: 未跑（合规停止；infra 存在：parser `--static` + 既有 m2072 模式可复用）。  
**生产 VCS**: 未启动。  
**paper_citable**: 否。
