# M617 r5 energy runner author handoff

状态：candidate only；M620 fresh PASS required；formal analyzer runs=0。

M616-P0-01 已进入可执行谓词：r5 精确核验 M615 true-release 与 M616 FAIL evidence 的完整 ID、SHA 和双封；未来唯一 M621 authorization 还必须携带实际 M620 PASS review 的 review/manifest/outer SHA，缺失、FAIL 或身份不一致均 fail-close。

M616-P0-02 已进入状态机：analyzer 前 exclusive 创建 attempt、写入并双封 receipt，再以 `RENAME_NOREPLACE` 发布到永久 consumed。crash 若发生在 rename 前，attempt 本身阻断重试；rename 后 success/failure/signal 均不移动、不删除、不改写 consumed。任何 qfinal、result/attempt/consumed、result/runtime/adapter/qraw/qstage 均按 `lexists` 阻断。

作者只运行 synthetic/static 测试：永久 consumed 二次尝试拒绝、no-replace 碰撞、qfinal 可见性、dangling symlink 拒绝，以及固定谱系与当前坐标 absence preflight。未调用 formal analyzer、`--execute`、GPU、EDA 或 remote。

冻结 M612 未修改；candidate `launch_now=false`、`release=false`。正式运行现在不被授权。
