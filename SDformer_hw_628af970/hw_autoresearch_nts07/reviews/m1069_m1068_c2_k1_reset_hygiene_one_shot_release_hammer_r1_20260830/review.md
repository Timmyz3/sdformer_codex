# M1069 independent release hammer

结论：**STOP，不授权 M1068。**

P0 是验证器与冻结证据格式不兼容：M1068 `verify_sidecar()` 要求 sidecar 第二列严格等于 basename；两份已钉住的 M1058 sidecar 第二列均为 `contracts/...` 相对路径，outer sidecar 同样为 `contracts/...sha256`。因此 runner 在 `SOURCE_PREFLIGHT/static_identity_gate` 即失败，attempt、DC 和 mapped VCS 均不可达。

建议新建加法 successor runner，使验证器只接受被显式钉住的 exact relative-name 或 basename 两种格式；不得改写旧 M1058 封存证据，并需新的独立 hammer。
