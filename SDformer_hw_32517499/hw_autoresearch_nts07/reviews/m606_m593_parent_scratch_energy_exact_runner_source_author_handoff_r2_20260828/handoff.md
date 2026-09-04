# M606 author handoff｜M593 parent-scratch energy exact runner r2 repair

状态：**SOURCE CANDIDATE ONLY；launch_now=false；release=false；NO FORMAL RUN。**

本作者只修复 M604 的执行器缺陷，不对自己的修复打分。fresh M607 必须由独立 reviewer 完成，且只有 `100/100, P0=0, P1=0` 才允许另行起草 M608 true-launch admission。

## 修复边界

- 数值/业务模型仍是 exact-SHA 冻结的 M597 r2；M606 adapter 只替换发布边缘。
- verifier 绑定完整 schema、固定输入 identity、scope、宏参数、每行来源与能量方程、CSV/JSON 一致性、精确 RUN_COMPLETE token。
- adapter staging、result、attempt consume 均使用 `renameat2(RENAME_NOREPLACE)`，并先做 `lexists/lstat` 检查。
- attempt 建立后的任何失败，包括 post-publish、attempt seal、consume 和 post-consume rehash，必须把 canonical result/attempt/consumed/staging 一并移入双封存 quarantine。
- publish 与 consume 后重新哈希 authorization、源码、输入、result 和 consumed attempt。

## 已做但不构成正式运行的检查

- bash syntax、Python 3.6 `py_compile`（pyc 只写临时目录）。
- source preflight/self-test。
- 缺授权 `--execute` fail-close，退出 70，canonical result/attempt/consumed 均不存在。
- 临时目录故障注入：no-replace 碰撞拒绝；伪 `RUN_COMPLETE=FAIL_NOT_COMPLETE` 结果拒绝；post-publish 故障把四个模拟 canonical 坐标全部隔离并双封存。

## 禁止

- 本 handoff 不是运行授权，不得生成 M606 formal result/attempt。
- 不得把尚未运行的 `38.2283%` component-energy diagnostic 写成 admitted paper data。
- 不得修改 docs/359；其 SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
