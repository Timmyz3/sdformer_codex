# M1622｜M1621 C2 VCS runner 窄修不同作者 hammer

日期：2026-09-01

状态：`PASS_M1622_M1621_M1613_C2_REGISTERED_FAULT_RUNNER_HAMMER__AUTHORIZE_ONE_FUTURE_VCS_ATTEMPT`

评分：98/100；P0=0，P1=0，P2=1。

## 裁决

M1621 通过不同作者、纯静态审阅。它只修复了 M1613 的 pre-attempt 路径阻塞：把会被 `expect_sha` 拒绝的 `/usr/bin/python3.6` symlink 换成其指向的普通可执行文件 `/usr/libexec/platform-python3.6`。两者解析后 SHA256 均为 `9c9502e...`，Python 内容未变。

除注释、Python 路径和必需的 M1621/M1622/M1623 控制平面身份外，新 runner 归一化后与 M1613 逐字节相同。M1609 RTL、M1613 TB/filelist、top、seed 1613、VCS/simv 命令、PASS token、M1613 result/attempt namespace、一次 compile+一次 simv、无自动重试和原子不覆盖发布均冻结。

## 反向变异结果

独立 hammer 在 CPython 3.6 和 3.10 下均拒绝 43/43 组攻击：Python/RTL/TB/filelist 身份漂移；review/release 路径、状态或 SHA 绑定绕过；compile/simv 预算扩张；seed、PASS token、namespace 漂移；attempt 后移、自动重试、非原子发布；same-UID 冲突或 ancestry 筛选被破坏；性能声称注入；以及 sealed tree 中的 extra flat、nested、symlink 和重复 manifest 成员。Runner `bash -n` 也通过。

本轮没有启动 VCS、simv 或任何 EDA，也没有创建 M1613 attempt/result 或 M1623 release。

## 一次性权限

当前只授权根代理另行创建并双封 M1623 release。M1623 必须精确绑定：

- runner SHA `11da68ff4eb9da70c83b56ae7dd2dbff26f125833224beb08f165fe97a0ea30b`；
- M1613 source contract SHA `248c9065d81608a8fc2aacdd8539a3287462653e411ee545a8f320a98a8a5f8d`；
- 本 `review.json` SHA；
- 权限恰为一次 VCS compile、一次 seed-1613 simv、零其他 EDA、零自动重试。

## P2 与结果红线

Runner 自身的 review 目录 seal 检查只校验 manifest 所列成员，不会主动拒绝未列出的 flat/nested 成员。这不改变被 M1623 绑定的 `review.json` 裁决，所以列为 P2；M1622 自身已按 exact topology 封存，未来 result hammer 必须拒绝 extra、nested 或 symlink 结果成员。

未来 VCS PASS 也只能证明 compactor-local registered-fault 行为，不证明外层 error OR-chain、周期、面积、时序、功耗或论文性能。`docs/359` 未修改，SHA 仍为 `dedde7ce...`。
