# M1092 M1090/M1091 observation source 独立打铁

结论：**STOP，78/100，P0/P1/P2 = 1/1/0；不授权 M1091。** 本审计未导入或执行 runner，没有运行 DC/VCS/GPU/remote，也没有消费 M1091 attempt。

## 成立的部分

- M1090 wrapper 的 22 路 `obs_*` 全部有驱动，均来自现有 functional/debug 信号；没有 observation 接回 M1058 implementation 或功能输出。
- TB 在 post-header 128-cycle 窗内逐拍打印 header/raw/busy/fault、service/adapter/bank counters；22 路 observation 均有 `$isunknown` fatal，另有 header 16-cycle、raw 32-cycle 和 1000 ns watchdog。
- runner 只描述一次 fresh DC、一次 mapped compile、一次 128-cycle case0；attempt 在 DC 前消费；旧 result/attempt/work 与被占 lock 都 fail-closed，失败进入 quarantine。
- 无 SAIF/toggle dump，无 `+vcs+initreg`；M1080 `DO_NOT_RETRY` 固定；未来结果明确是 diagnostic-only、`paper_citable=false`。
- 旧 C2 M1089 草稿源已清空；唯一 `m1089` 是 checkpoint 重绑定 review，本次未触碰。

## P0：GO 与 release 是 caller 自签

M1091 从环境读取：

```text
M1091_EXPECTED_RELEASE_OUTER_SHA256
M1091_EXPECTED_RUNNER_SHA256
M1091_EXPECTED_M1092_OUTER_SHA256
```

但没有独立冻结的 caller/trust root 规定这些值必须是什么。因此两种攻击可通过当前 `static_gate`：

1. 修改 contract/source，重算 contract 双封；修改 release 中的 contract outer/runner hash，重算 release 双封；把新 release outer 与 runner hash 放入环境。
2. 构造一个带所需 GO status 和 `one_m1091_attempt=true` 的假 M1092，重算 manifest/outer，把假 outer 放入环境。

这与“manifest 自洽”不同：runner 缺少的是独立信任根，无法判断 GO 是否真的来自本次独立打铁。故当前绝不能执行 M1091。

## P1：外部工具/库只有路径，没有身份

dc_shell、VCS、slow/fast DB、cell-model 是绝对版本路径，但没有 SHA pin，也没有拒绝 direct/resolved symlink。项目源文件的 hash/symlink 边界是合格的；外部工具边界尚未闭合。

## 最小修复

M1090/M1091 与本 STOP 全部冻结，不得原地修、不消费 attempt。新 namespace 需要：

1. 加入不能由同一调用环境自行选择的 sealed caller/launch trust root；重签全部 caller-visible hashes 后仍必须拒绝伪造 contract/release/runner/GO。
2. 对 dc_shell、VCS、slow/fast DB、cell model 做 exact identity pin，并拒绝 symlink。
3. 保留现有 22 路纯扇出、128-cycle 首 X、单 DC/单 case、attempt-before-EDA、quarantine、无 SAIF/initreg、diagnostic-only 边界。
4. 新 source 仍需不同作者打铁；通过后才可授权 root 唯一一次新 namespace attempt。

`docs/359` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
