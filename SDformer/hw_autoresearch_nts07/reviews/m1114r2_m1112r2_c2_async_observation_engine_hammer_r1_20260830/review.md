# M1114r2：M1112r2 C2 async-observation source/engine 独立打铁

## 裁决

**PASS：只授权异作者进入 zero-argument launcher 的作者阶段。**

本轮只执行 Python source/static 和临时目录变异检查；没有调用 VCS、DC、simv、launcher，没有生成 launch receipt，也没有消费 attempt。

## M1113 两个 P0 的闭合

### 337-bit reset provenance

独立构造的 337-bit mapped 结构中，以下三种合法形态通过：

- `rst_core → INVD1 → CDN`
- `rst_core → INVD1 → CN`
- `rst_core → CKND1 → CDN`

以下 13 类攻击全部被拒：无关 fake reset、常零 clear、常一 clear、`rst_core` 直接接 active-low clear、两级反相、重汇合逻辑、buffer 代替 inverter、set-only cell、D/clock-only cell、336-bit census、双驱动、inverter 多余 pin、active set 未固定为 inactive high。

因此 gate 现在同时证明数量、清零语义、复位极性、单级允许反相器、唯一驱动和 canonical `rst_core` 来源，而非仅匹配 `.CDN/.CN` 字符串。

### Live seal boundary

合法 double seal 与 exact-flat seal 通过。以下 8 类攻击全部被拒：live primary/sidecar/outer symlink、flat member/manifest/manifest-outer symlink、未列 extra member、manifest 所列成员缺失。

M1113 STOP、M1112r2 作者回执及全部 contract-pinned live source 均按 regular/no-symlink/固定 SHA 核验。唯一历史例外仍限定为 M1080 quarantine，实测 symlink census 为 1，目标位于封存目录内、解析后为 regular file，followed bytes 与 manifest 一致。

## 保持不变的行为边界

- 13 个 service/adapter async shadow counter，共 337 bit。
- frozen synchronous debug 仅终止到 unused sink，不参与 observation。
- shadow/observation 不反馈进功能锥。
- 22 个同拍 unknown predicate，first-X 与后续 union 保留，完整 128-cycle 窗口末才 fail closed。
- M1113 的 STOP 状态和 outer seal 被新 contract、engine 与本回执共同绑定。

## 下一步权限

本回执只允许另一作者创建并封存固定 zero-argument launcher 与 launch receipt。launcher 必须固定 Python、清空非白名单 caller environment、不给调用者参数/路径/hash 选择权，并继续保持 attempt namespace 不存在。

launcher 作者阶段完成后，仍需 M1115r2 独立 launch hammer。只有 M1115r2 明确给出 one-attempt GO，才可执行一次 fresh DC + mapped-VCS。当前没有 mapped functionality、性能、功耗、系统倍速或论文 PPA 准入。
