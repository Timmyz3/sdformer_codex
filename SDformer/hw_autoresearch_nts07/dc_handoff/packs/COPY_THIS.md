# 拷到另一台新思服务器：只拿这两个文件

本机没有、也不会跑 `dc_shell` / `fm_shell` / `pt_shell` / `vcd2saif`。  
新思流程只在你有库和许可证的那台机器上做。

## 拷走

不要用旧包。当前唯一有效包是 Git LFS 上的：

```text
SDformer/hw_autoresearch_nts07/dc_handoff/packs/date_dual_synopsys_handoff_20260814T194436Z.tar
SDformer/hw_autoresearch_nts07/dc_handoff/packs/date_dual_synopsys_handoff_20260814T194436Z.tar.sha256
```

clone 后必须 `git lfs pull`，否则 tar 只是指针。同目录更早的 `170941Z`…`193554Z` 不在 git 里，作废。

SHA256 必须是：

```text
ff986c74070e39f2effe24494f911490dbc896036b798599ebf525779a1f6ebc
```

同目录里更早的 `170941Z` / `172732Z` / `172904Z` / `192921Z` / `193554Z` 都作废。

可选第三份：`server_run_four_tops.sh`（本目录也有一份）。解包后放到
`hw_autoresearch_nts07/dc_handoff/scripts/`，可按四顶层自动跑。

## 在新思服务器上

先校验，再解包。`sha256sum -c` 必须在 **tar 和 sha 文件所在目录** 执行：

```bash
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

两道门都必须 PASS。然后打开包内 `dc_handoff/SERVER_RUN.md`：
DC → Formality → `vcd2saif` → PTPX → setup/hold PTSTA。

没有目标 SRAM/RF `.db` 时不要设 `PPA_ADMISSION=1`。结果只能标
**pre-macro 逻辑 DC/STA/PTPX**，不是流片签核，也不是 encoder PPA。

准入清单：包内 `dc_handoff/PPA_REVIEW_CHECKLIST.md`。
