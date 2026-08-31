# M1071 independent release hammer

结论：**STOP，不授权 M1070。**

sidecar successor 的静态语义正确：只接受 exact `contracts/<basename>` 与 `contracts/<basename>.sha256`；M1069 的 basename 不兼容也已独立重放。但更早的生产工具身份门不可达：M1070 `expect_sha()` 无条件拒绝 symlink，冻结的 `/opt/synopsys/syn/V-2023.12-SP3/bin/dc_shell` 实际为 `dc_shell -> snps_shell`。隔离执行在 `SOURCE_PREFLIGHT/static_identity_gate` 报 identity drift，真实 attempt、DC、VCS 均未触发。

必须新增 additive successor，显式钉住 symlink 路径、exact `readlink` 目标和 resolved payload SHA；不得改写 M1070。随后需要新的独立 hammer。
