# M838：M837/C2 R22 identity-compat fresh source hammer

## 裁决

**PASS 100/100，P0/P1/P2 = 0/0/0。** R22 的 additive identity wrapper 已闭合 M834 R21 的精确 PASS100 status 与四键 target，旧 M826/M833 status、三键、逐键缺失和额外键全部 fail-closed；M832 已判 spent 的 M826 release 不能复用。

本轮只授权另一位作者制作一次 `M839/M837 R22` true release。它不授权本 reviewer 制作 release/final hammer，也不授权 VCS、simv、license query、EDA、attempt、result 或 failure quarantine。

## 独立证据

- request、author handoff、runner、guard、contract、candidate、M823/M832/M834 authority 的适用双封全部重算通过；contract 的 35 个 source member 均为 exact regular non-symlink，SHA 全部相符。
- Python 3.6.8 与 3.12.13 各自通过 atomic 12/12、final authorization 8/8、Unicode 5/5、R22 identity 11/11、source closure 与 synthetic positive launch chain。
- 精确 M834 R21 status 与四键 target 通过；旧 M826 status、错误 M833 status、三键、每个缺键和额外键均被拒绝。M832 的 `m826_release_reusable=false`、`m826_attempt_consumed=false` 已被 source contract 精确绑定。
- 额外补测 release/final 链 9 个攻击：旧/错 release status、release auth 缺键/额外键/bool-int，错 final status、final auth 缺键/额外键/bool-int，全部被拒绝。
- actual runner 在 outer `LANG=C, LC_ALL=C` 下 wrong-SHA 返回 3，positive source dry-run 返回 86；事件仅到 live VCS/license boundary sentinel，VCS/license/simv/attempt/result/quarantine 均为 0。
- 12 个 guard Python child 与 1 个 inline writer 仅在 child scope 使用 `C.UTF-8`；无 global locale export。`license_gate` 与 `compile_and_run` 相对 M833 byte-identical，VCS/simv 保持 outer C。
- M803 RTL/SVA/TB/filelists、五档周期 `51/53,131/133,486/499,1231/1246,14/14`、四 receipt `false,false,true,true` 与 15-key exact typed authorization 均保持冻结。

完整机械回执见 `mechanical_checks.txt`。

## 边界与下一步

当前仍是 source-only：`vcs_validated=false`、`rtl_validated=false`、`dc_validated=false`、`paper_citable=false`。下一步只能由 fresh release author 精确绑定本 review 的 outer seal，生成一次 M839 true release；随后仍须独立 final launch hammer，才可能授权一次 live VCS。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
