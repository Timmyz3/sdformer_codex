# M828 / M826 C2 R20 one-way true-release author handoff

M827 已以 PASS100、P0/P1/P2=`0/0/0` 授权编写一份 true release。M828 已生成并双封固定路径 release：

`contracts/m826_c2_r20_atomic_vcs_launch_admission_r1_20260829.json`

release SHA256 为 `52606ff5e04e1b65c29b429b92c32c6022bcbb188d3d5737ccefd4a9581c830f`，outer-seal-file SHA256 为 `a698613567a287a39393b92ec9ae2aa2bd8283285ea4628a899377ec909ce0dc`。

## 权限边界

M826 冻结 guard 要求 release 的 `authorization` 是 6 键 release-schema 对象；该对象保持完全兼容。真正进入 final hammer 的执行授权另外固定为 `final_hammer_authorization_exact`，它是 M827 已穷举证明的 15 键 typed closed set。M829 review 必须逐键、逐值、逐 Python 类型重复这 15 键，并由 guard 的 `require_exact_typed_mapping()` 直接验证。

本 release 在 M829 独立 PASS100 及 caller 精确 pin M829 outer-seal 前无效。M828 没有运行 VCS、simv、license query、Icarus、Verilator、DC、Formality、PT、PTPX，也没有创建 formal attempt/result/failure quarantine。

## 绑定内容

- runner、guard、source contract、candidate 及各自 SHA/双封；
- M827 PASS100 review 和双封；M823 negative review 及被 M826/M827 关闭的唯一 P1；
- M803 RTL/SVA/TB 和三份 filelist；
- K8/K1x8 五组 exact 周期 `51/53,131/133,486/499,1231/1246,14/14`；
- numeric/tuple/weight=0，stall/full8/K1x8-issue/out-of-order 非零；
- attempt receipt `false/false/true/true`、collision no-clobber 和 postrename damaged conservative-consumed；
- `docs/359 = dedde7ce...`。

## 下一步

只能由另一位 receipt-blind reviewer 按 M829 request 评审 release 与整条 chain。只有固定路径 M829 review PASS100、P0/P1/P2=`0/0/0` 并完成双封后，caller 才能把其 outer-seal-file SHA 放入 `M826_R20_EXPECTED_FINAL_HAMMER_OUTER_SEAL_SHA256`，执行唯一一次无参数 runner。M828 作者不得自做 M829 final hammer。
