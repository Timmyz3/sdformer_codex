# M1117r3：M1112r3 acyclic engine 独立打铁

## 裁决

**PASS：仅授权异作者编写并封存 zero-argument launcher。** 未运行或创建 launcher、attempt、VCS、DC、simv。

## 无 SHA 固定点

- Launch receipt 的精确字段只绑定已存在的 engine、contract、author receipt、M1116 和 M1117r3 authority，不包含未来 M1118r3 outer。
- M1118r3 outer 在授权执行时由 `verify_flat_self_consistent()` 从 regular、无 symlink、exact-member seal 自洽发现。
- M1118r3 review 必须逐项绑定 launcher SHA、launch-receipt outer、engine SHA、contract outer、author-receipt outer、M1116 outer 和 M1117r3 outer。
- 依赖方向为 `M1117r3 → receipt → M1118r3 review → self-consistent outer discovery`，不存在 `receipt → future outer → review → receipt` 回边。

合法临时链通过。15 类攻击全部被拒，包括 future-outer placeholder、caller environment 转发、伪 outer、自洽重封但更改 authority bytes、七个 review identity pin 分别漂移、旧 hammer 替换、launcher symlink、receipt sidecar symlink、future manifest symlink。

## 保持的硬件与信任机制

- M1116 circularity STOP 和 M1114r2 source hammer 均按固定 outer 完整验证。
- 13 个 async shadow counter、337 bit、22-signal bitmap 和 observation 无功能反馈保持。
- 337-bit reset provenance 合法结构通过；fake/direct/constant/multilevel/reconvergent/set-only/336-bit 攻击均拒绝。
- Live exact-member seal 继续拒绝额外成员和 manifest symlink。
- M1112r3 launcher、launch receipt、attempt、result 均不存在；attempt consume 调用只有一次，automatic retry 为 false。

下一阶段只可由另一作者创建固定、零参数、caller-environment 不转发的 launcher 与 source receipt。仍需 M1118r3 独立 launch hammer 后才可能授权一次生产 attempt。当前无 mapped functionality、性能、能量或论文 PPA 准入。
