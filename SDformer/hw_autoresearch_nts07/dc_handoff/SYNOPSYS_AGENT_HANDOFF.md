# 给新思服务器 agent 的接手说明

日期：2026-08-17。本机没有、也不会跑 `dc_shell` / `fm_shell` / `pt_shell` / `vcd2saif`。
你的任务是：先读懂双线硬件身份，再在有库和许可证的机器上跑 DC → Formality → vcd2saif → PTPX → PTSTA。

不要写 H81 RTL。不要改 `docs/359`。不要把 Yosys/OpenROAD/Nangate45 数字写成 ASIC PPA。

## 0. 不要整仓盲 clone 来跑新思

Git 仓库是：

```text
https://github.com/Timmyz3/sdformer_codex.git
分支：autoresearch/neuron-ops-20260507
```

**不要**为了读文档去 clone 整个历史当输入。仓库历史里曾经误提交过 1GB 级 VCD 和大量 Verilator `*.gch` / `obj/`。HEAD 已把这些从当前树拿掉，但 **普通 clone 仍可能拉历史大文件**。用 `--filter=blob:none --sparse`。

新思流程的**完整可跑输入**是 Git LFS 上的交接包（活动 VCD + RTL 快照 + 脚本），不是全仓 `build_*`。

## 1. 你需要拿什么

### A. 必须：交接包（Git LFS，731MB）

```bash
git clone --filter=blob:none --sparse --branch autoresearch/neuron-ops-20260507 \
  https://github.com/Timmyz3/sdformer_codex.git
cd sdformer_codex
git sparse-checkout set \
  SDformer/hw_autoresearch_nts07/dc_handoff \
  SDformer/hw_autoresearch_nts07/rtl_h67 \
  SDformer/hw_autoresearch_nts07/rtl_ttx \
  SDformer/hw_autoresearch_nts07/rtl_qfit \
  SDformer/hw_autoresearch_nts07/rtl_local5 \
  SDformer/hw_autoresearch_nts07/tb_h67 \
  SDformer/hw_autoresearch_nts07/tb_qfit \
  SDformer/hw_autoresearch_nts07/tb_local5 \
  SDformer/hw_autoresearch_nts07/tb_ttx \
  SDformer/hw_autoresearch_nts07/sim_h67 \
  SDformer/hw_autoresearch_nts07/sim_qfit \
  SDformer/hw_autoresearch_nts07/sim_new_arch \
  SDformer/hw_autoresearch_nts07/sim_ttx \
  SDformer/hw_autoresearch_nts07/verif_h67 \
  SDformer/hw_autoresearch_nts07/verif_qfit \
  SDformer/hw_autoresearch_nts07/docs \
  SDformer/hw_autoresearch_nts07/scripts \
  SDformer/neuron_autoresearch
git lfs install
git lfs pull --include="SDformer/hw_autoresearch_nts07/dc_handoff/packs/date_dual_synopsys_handoff_20260814T194436Z.tar"
```

路径：

```text
SDformer/hw_autoresearch_nts07/dc_handoff/packs/date_dual_synopsys_handoff_20260814T194436Z.tar
SDformer/hw_autoresearch_nts07/dc_handoff/packs/date_dual_synopsys_handoff_20260814T194436Z.tar.sha256
```

SHA256 必须是：

```text
ff986c74070e39f2effe24494f911490dbc896036b798599ebf525779a1f6ebc
```

可选第三份：`dc_handoff/packs/server_run_four_tops.sh`。

包前缀是 `hw_autoresearch_nts07/`。包内活动仍是 Local5 **ep29** 论文合同，不是 ep44 sidecar。H81 不在包里。

稀疏检出必须包含 `tb_*` / `sim_*` / `verif_*` 源码和已进 git 的向量，方便对照、Formality 调试和必要时重仿。论文 PPA 仍用包内已封 VCD 做 `vcd2saif`，不要为交差去重出一套活动。

不要再全量 clone。不要检出：`build_*`、`results/`、`neuron_experiments/**/results/`、任何 `*.pth`。若干超大向量（约 20MB+ memh/txt）已从 git 拿掉，包内已有 DC 活动 VCD 和对应合同。

## 2. 先读这些，再碰 dc_shell

按这个顺序读，读完再跑：

1. `dc_handoff/packs/COPY_THIS.md`（拷包规则）
2. `dc_handoff/SERVER_RUN.md`（本文件之外的逐步命令）
3. `dc_handoff/PPA_REVIEW_CHECKLIST.md`（什么叫 PASS）
4. `docs/359_DATE终局冻结_20260813.md`（冻结表；SHA 必须仍是 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`）
5. `docs/418_本机不跑新思_服务器交接包_20260815.md` 与 `docs/422_用户澄清本机不跑新思_可拷走剩余项_20260815.md`
6. `docs/421_三线硬件preDC最终交接与剩余门_20260815.md`
7. `docs/433_H81_G0与Local5创新筛选收口_20260816.md`（H81 不是投稿对象；MVSEC 四序列门已 FAIL）

主线是 **H67 Motion + Local5**，不是 H81。

| 顶层 | 身份 | 论文周期 | 备注 |
|---|---|---:|---|
| `h67_fixed2s_mssb5_dc_top` | H67 ep35 | 112589 | 必须和 RQTB 成对报 |
| `h67_rqtb2s_mssb5_dc_top` | 同上 | 94891 | 1.1865×，不能搬给 H81 |
| `local5_unified_out2_dc_top` | Local5 组件 OUT_DIM=2 | 155791 | 主列 |
| `local5_unified_out2_1rw_dc_top` | 同端口敏感度 | 170269 | 不是第三主列 |

全是 **READY_PREMACRO**。没有目标 SRAM/RF `.db` 就不要设 `PPA_ADMISSION=1`。结果只能标 pre-macro 逻辑 DC/STA/PTPX，不是 encoder PPA，不是流片签核。

禁止写进论文主表的数字见 `docs/359` 和 433 的禁表。Yosys/ABC/Nangate45/OpenSTA 不是 ASIC PPA。

## 3. 服务器上怎么跑

```bash
# 在 tar 与 sha 所在目录
sha256sum -c date_dual_synopsys_handoff_20260814T194436Z.tar.sha256
tar -xf date_dual_synopsys_handoff_20260814T194436Z.tar
cd hw_autoresearch_nts07

python3 dc_handoff/scripts/audit_date_dual_handoff.py \
  --root . \
  --output dc_handoff/runs/date_dual_handoff_audit_server.json
python3 scripts/audit_three_line_predc_gate.py \
  --root . \
  --output results/grok_codex_collab/three_line_predc_gate_server.json
```

两道门都必须 PASS。然后设服务器自己的库：

```bash
export LIB_DB=/path/to/ss_corner.db
export OPERATING_CONDITION=ss_0p9v_125c
export CLOCK_PERIOD_NS=3.0
```

四个顶层各自独立 `DESIGN_NAME` 和 `DC_RUN_DIR`，禁止复用上一个设计的目录。每个顶层：

1. `dc_handoff/run_dc.sh`
2. `dc_handoff/run_formality.sh`（`formality_status.txt` 必须精确为 `PASS`）
3. 本机 VCD 在服务器上 `vcd2saif`（本机不转）
4. `make_saif_manifest.py` 后 `run_ptpx.sh`
5. setup / hold 各用独立 `PT_RUN_DIR` 跑 `run_ptsta.sh`

活动合同、`SAIF_INSTANCE`、测量窗见 `SERVER_RUN.md`。Motion 活动必须对上 `112589/94891`。Local5 主顶层 busy `155791`。不要用带 `obj/` 的编译目录当输入。

也可把 `server_run_four_tops.sh` 放到 `dc_handoff/scripts/` 后按四顶层自动跑。

## 4. 你不要做的事

- 不要跑 H81 DC，不要写 H81 RTL。
- 不要重打包、不要改 359、不要改公平 LFSR `16'h1d3f`。
- 不要把 ep44 AEE 1.2819 / hardware-order 1.2804 写进冻结表。
- 不要把 1.770×（OUT_DIM=2 tile）写成 encoder。
- 不要抢算法 GPU，不要在本仓库里再发明新 DATE 对象。

跑完后按 `PPA_REVIEW_CHECKLIST.md` 回传四个顶层的 DC / Formality / PTPX / PTSTA 报告路径和 WNS/面积/功耗，并标明 **pre-macro**。
