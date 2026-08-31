# M829｜M826/C2 R20 final-launch hammer

## Verdict

`PASS100_M826_R20_FINAL_LAUNCH__ONE_VCS_ATTEMPT_AUTHORIZED`，100/100，P0/P1/P2=`0/0/0`。

本评审只授权 caller 在 live preflight 下执行一次固定 M826 R20 Synopsys VCS+simv runner。本 reviewer 没有运行 VCS、simv、license query、HDL compile、Icarus、Verilator、DC、Formality、PT、PTPX、CPU/GPU workload、remote/network job，也没有创建 formal attempt、result 或 failure quarantine。

## Authority closure

- release SHA 为 `52606ff5e04e1b65c29b429b92c32c6022bcbb188d3d5737ccefd4a9581c830f`，双封 live；runner/contract/candidate 分别固定为 `4e64356a...f0a9b`、`70cc165f...c1c1b`、`88993646...99ce`。
- release 顶层 `authorization` 是冻结 guard 所需的 6 键 release schema，逐键、逐值、逐 Python 类型正确；它不是漏洞，也不应被错误要求改成 15 键。
- 真正的 executable final gate 是 release 的 `final_hammer_authorization_exact` 与本 review 的 `authorization`。两者均为同一个 15 键 typed closed set；guard 对 future review 直接调用 `require_exact_typed_mapping()`。
- 实测 exact positive chain 通过；5 个规定错值、15 个缺键、15 个 bool/int 混淆、extra key，以及 3 个 duplicate-key 与 NaN/Infinity/-Infinity 共 6 个 strict-JSON 攻击全部拒绝。
- M823 揭示的“矛盾 deny-launch authorization 仍放行”P1 已由 M826 exact typed closure 和 M827 PASS100 关闭。

## Frozen source and atomic boundary

- request、M828、M827、M823 目录双封重算通过；release、runner、contract、candidate 文件双封重算通过；contract 40 个 source SHA 全部 live。
- M803 RTL/SVA/TB 与三份 filelist 未改；filelist 无重复、缺失或 symlink。
- 五组 exact cycle 仍为 `51/53, 131/133, 486/499, 1231/1246, 14/14`。numeric/tuple/weight 必须为 0，request/result/raw stall、full8、K1x8 full issue、两侧 out-of-order cover 必须非零，runner 未弱化。
- Python 3.6.8 与 3.12.13 均通过 compile、atomic 12/12、final-authorization 8/8、wrong-SHA rc3 和 source dry-run rc86；function closure 通过，undefined-function mutation 被拒。
- fresh CLI replay 得到 `attempt_consumed=false,false,true,true`。pre-existing exact collision 保持 source/destination 双侧不改；postrename exact 与 damaged destination 均按 durable move 保守记为 consumed。四份临时 receipt 均双封。

## Exact caller command

先把本目录 `SHA256SUMS.seal.sha256` 文件本身的 SHA256 代入 `<M829_OUTER_SEAL_FILE_SHA256>`，然后只可执行一次：

```bash
cd /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07
env -i PATH=/usr/bin:/bin LANG=C LC_ALL=C \
  VCS_HOME=/opt/synopsys/vcs/V-2023.12-SP1 \
  VCS_ARCH_OVERRIDE=linux \
  SNPSLMD_LICENSE_FILE=27030@ic.ismd-nemo \
  LM_LICENSE_FILE=/opt/synopsys/Synopsys.dat \
  M826_R20_EXPECTED_VCS_RUNNER_SHA256=4e64356a50e1c7bb409ba1d05fea57f505899b01d5cbe5c899077f25c1af0a9b \
  M826_R20_EXPECTED_FINAL_HAMMER_OUTER_SEAL_SHA256=<M829_OUTER_SEAL_FILE_SHA256> \
  /bin/bash -p /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/dc_handoff/scripts/run_vcs_m826_c2_r20_atomic_exact_sha.sh
```

runner 必须自行重做 live collision/resource/license/SHA gate；任何失败都只允许进入 sealed non-paper quarantine。完成后仍需 fresh result hammer，才能引用功能、周期、PPA、能量或论文指标。

## Claim boundary

当前仅完成 final-launch admission。VCS/RTL validation、DC/timing/PPA、cycle/speedup、energy、system/headline/paper claim 全部仍为 false。`docs/359` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
