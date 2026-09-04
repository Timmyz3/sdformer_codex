# 另一台服务器上跑新思（本机不跑）

本机没有、也不需要 `dc_shell` / `fm_shell` / `pt_shell`。  
本目录的目标是：拷到有 Synopsys 和目标库的服务器后，按下面顺序能直接跑。

结果一律标 **pre-macro 逻辑 DC**，除非服务器提供了 SRAM/RF `.db` 和 adapter，并设 `PPA_ADMISSION=1`。  
不是 encoder PPA，不是签核，不是本机 Yosys/OpenROAD 数字。

## 0. 先解包

保持仓库相对布局。工作目录必须是 `hw_autoresearch_nts07/`：
复制 `.tar` 和同名 `.tar.sha256` 后，先 `cd` 到两者所在目录，再运行
`sha256sum -c <archive>.tar.sha256`，校验通过再解包。

```text
hw_autoresearch_nts07/
  dc_handoff/          # 脚本、SDC、filelist、wrapper、活动 VCD
  rtl_h67/ rtl_ttx/ rtl_qfit/ rtl_local5/
  docs/359_DATE终局冻结_20260813.md
```

```bash
cd hw_autoresearch_nts07
python3 dc_handoff/scripts/audit_date_dual_handoff.py \
  --root . \
  --output dc_handoff/runs/date_dual_handoff_audit_server.json
python3 scripts/audit_three_line_predc_gate.py \
  --root . \
  --output results/grok_codex_collab/three_line_predc_gate_server.json
```

审计必须 PASS。`docs/359` SHA 必须仍是
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 1. 服务器必须自备（本包不提供）

| 输入 | 用途 |
|---|---|
| `LIB_DB` 标准单元 `.db` | DC / Formality / PT |
| `OPERATING_CONDITION` | 与库角一致 |
| `CLOCK_PERIOD_NS` | 默认探索 3.0，可改 |
| 可选 `MACRO_DBS` + `EXPECTED_MACRO_REFS` | 只有这时才能 `PPA_ADMISSION=1` |
| `vcd2saif` | 把本机 VCD 转 SAIF（本机不转） |

没有宏就不要设 `PPA_ADMISSION=1`。当前顶层 `MEMORY_IMPL=0` / `ACC_MEMORY_IMPL=0`。

## 2. 三个论文组件顶层 + 1RW 敏感度

| DESIGN_NAME | 活动 VCD + 合同 | SAIF_INSTANCE |
|---|---|---|
| `h67_fixed2s_mssb5_dc_top` | `runs/motion_fixed_dc_activity_population138_fair/` | `TOP/tb_h67_motion_dc_activity/g_fixed/dut` |
| `h67_rqtb2s_mssb5_dc_top` | `runs/motion_rqtb_dc_activity_population138_fair/` | `TOP/tb_h67_motion_dc_activity/g_rqtb/dut` |
| `local5_unified_out2_dc_top` | `runs/local5_dc_activity_full_population100/` | `TOP/tb_qfit_local5_score_projection_postg0/g_dc_wrapper/dut` |
| `local5_unified_out2_1rw_dc_top` | `runs/local5_1rw_activity_population100_full/` | `TOP/tb_qfit_local5_score_projection_postg0/g_dc_wrapper/dut` |

`local5_unified_out2_1rw_dc_top` 是同端口敏感度，不是第三主列。VCD 在
`runs/local5_1rw_activity_population100_full/`。

Motion 论文活动必须对上 `112589/94891`、slot `62100/34099`、equal `28001`。  
Local5 主顶层 busy `155791`。不要用带 `obj/` 的编译目录当输入。

## 3. 每个顶层：DC → Formality → 转 SAIF → PTPX / PTSTA

以 RQTB 为例。`LIB_DB` 换成服务器路径。

```bash
export LIB_DB=/path/to/ss_corner.db
export OPERATING_CONDITION=ss_0p9v_125c
export CLOCK_PERIOD_NS=3.0
export DESIGN_NAME=h67_rqtb2s_mssb5_dc_top
export DC_RUN_DIR=dc_handoff/runs/h67_rqtb2s_mssb5_dc_top

dc_handoff/run_dc.sh
dc_handoff/run_formality.sh

# 本机 VCD -> 服务器 SAIF
vcd2saif \
  -input dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/h67_rqtb2s_mssb5_dc_top.vcd \
  -output $DC_RUN_DIR/h67_rqtb2s_mssb5_dc_top.saif

python3 dc_handoff/scripts/make_saif_manifest.py \
  --root . \
  --activity-contract dc_handoff/runs/motion_rqtb_dc_activity_population138_fair/activity_contract.json \
  --saif $DC_RUN_DIR/h67_rqtb2s_mssb5_dc_top.saif \
  --output $DC_RUN_DIR/h67_rqtb2s_mssb5_dc_top_saif_manifest.json

SAIF_FILE=$DC_RUN_DIR/h67_rqtb2s_mssb5_dc_top.saif \
SAIF_INSTANCE=TOP/tb_h67_motion_dc_activity/g_rqtb/dut \
SAIF_MANIFEST=$DC_RUN_DIR/h67_rqtb2s_mssb5_dc_top_saif_manifest.json \
CORNER_ROLE=power \
MIN_SAIF_COVERAGE_PCT=95.0 \
dc_handoff/run_ptpx.sh

LIB_DB=/path/to/setup_corner.db \
OPERATING_CONDITION=setup_corner_name \
CORNER_ROLE=setup \
PT_RUN_DIR=$DC_RUN_DIR/pt_setup \
dc_handoff/run_ptsta.sh

LIB_DB=/path/to/hold_corner.db \
OPERATING_CONDITION=hold_corner_name \
CORNER_ROLE=hold \
PT_RUN_DIR=$DC_RUN_DIR/pt_hold \
dc_handoff/run_ptsta.sh
```

Fixed2S、Local5 必须同步改 `DESIGN_NAME`、`DC_RUN_DIR`、活动目录和
`SAIF_INSTANCE`；禁止复用前一个设计的 `DC_RUN_DIR`。  
三个顶层都要跑；Motion 必须成对报 Fixed 与 RQTB。
每个顶层的 setup/hold 使用独立 `PT_RUN_DIR`，禁止后一个角覆盖前一个。
PTSTA 不依赖 PTPX；只要 DC 网表/SDC 和该角库已准备好就可独立运行。
最终准入项见 `dc_handoff/PPA_REVIEW_CHECKLIST.md`。

`docs/419_Local5当前生产前端跨Head_OUT32闭合_20260815.md` 是生产前端到
3-head OUT32 的 synthetic 组件证据，不是第四个 PPA 顶层，也不用它生成
论文功耗列。

有宏时再加：

```bash
export PPA_ADMISSION=1
export MACRO_DBS=/path/to/sram.db:/path/to/rf.db
export EXPECTED_MACRO_REFS=macro_a,macro_b
```

没有 adapter / 宏名未知：不要猜，保持 pre-macro。

## 4. 失败必须是真失败

- 缺 `dc_shell` / 库：脚本非零退出，不许造假报告。
- Formality 只认 `reports/formality_status.txt == PASS`。
- 无 SAIF 时 DC 不写默认翻转率功耗。
- `audit_synopsys_postrun.py` 查工件和工具 Error；WNS/TNS 仍要人看报告。

## 5. 本机打包

```bash
cd hw_autoresearch_nts07
bash dc_handoff/scripts/pack_synopsys_handoff.sh
```

生成 `dc_handoff/packs/date_dual_synopsys_handoff_<date>.tar`。  
不含 Verilator `obj/`，含论文活动 VCD 和合同。
