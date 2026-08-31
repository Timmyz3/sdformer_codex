# M607 fresh independent static-hammer request

请由非 M606 作者独立审阅 M606/M593 parent-scratch component-energy exact runner。不得运行 formal analyzer，不得生成 result/attempt，不得起草 M608 授权。

必须逐项攻击：

1. 构造完整双封存但 RUN_COMPLETE 失败、identity/member map 缺失或 schema 错误的伪结果，确认 verifier 拒绝。
2. 对 post-publish、attempt-seal、consume、post-consume 四处分别故障注入，确认 canonical result/attempt/consumed/staging 全部消失并进入一个可验证 quarantine。
3. 攻击 adapter staging、result publish、attempt consume 的 symlink/dangling-symlink/existing-coordinate 竞态，确认真正 no-replace。
4. 改写 authorization、runner、adapter、source contract、result 或 consumed attempt，确认对应 terminal rehash 拒绝。
5. 核对 M597 完整 schema、固定 frozen-input identity、scope/macro、两行来源/方程、CSV/JSON、精确完成 token 与 terminal receipt 的 exact member map。

只有 score=100 且 P0=P1=0，才可在 review 中写 `true_launch_admission_authoring_allowed=true`。作者不得自评。
