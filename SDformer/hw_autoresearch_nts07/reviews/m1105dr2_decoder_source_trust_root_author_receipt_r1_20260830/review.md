# M1105Dr2 decoder source trust-root repair author receipt

结论：**r2 source trust root 修复通过作者预检；production 继续 STOP，下一步必须由不同作者 hammer。**

M1106D 发现的 caller-selectable `repo/contract/output` 已从 r2 移除。source 从自身固定位置导出 canonical repository、hardware root 与 M699 payload，只接受零参数；canonical contract 的 file/sidecar/outer SHA 被 source 硬编码。contract 的 136 个 leaf 以固定 digest 全覆盖，并对 trust、输入、population、D1、资源、地址、dependency/time、release 和 claim boundary 逐字段投影。

source SHA 不写回 predecessor contract，避免 source↔contract hash 循环；本 sealed author receipt 同时绑定 source SHA `b2d8ef41...a5c4` 与 contract triple `cdbae036...80a4 / 37cdc8aa...b4fe / 4f95a616...193d`。下一位独立 hammer 必须同时 pin 该 receipt outer、source 与 contract triple。

作者预检在伪造 repo/contract/output/expected-SHA 环境变量存在时重新执行 canonical preflight，结果仍为 120 calls、261,090,000 packed bytes、30 个 D1 theta miter 0 mismatch。18 类合同字段篡改和 6 类 caller path/argv 攻击全部拒绝。M700 不准入，final-checkpoint 变化仍要求完整 rebind，`production_run_allowed=false`。

本轮没有创建 runner、attempt 或 production result，没有枚举 production transaction，没有运行 EDA/RTL。没有新增可引用周期、traffic、speedup 或 PPA。
