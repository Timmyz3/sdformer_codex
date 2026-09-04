# M1645 actual-prefix runner SOURCE ONLY 作者回执

结论：`PASS_AUTHOR_M1645_ACTUAL_PREFIX_RUNNER_SOURCE__M1646_DIFFERENT_AUTHOR_REVIEW_REQUIRED__NO_EXECUTION`，作者分 98/100。本里程碑只完成可审阅的 actual-prefix runner 源码、测试与合同，没有打开 ep34 payload，没有执行 prefix，也没有创建 attempt/result/release。

源码将人口锁定为 final `motion_ep34_live93`、D0/call0/module0/timestep0、destination 0..41、output block 0..3，三个 non-product 配置顺序不可变。每个配置各自创建 exact M1539 reference scheduler、exact M1610 compact engine、持久 mirrored weight cache 和 exact M1638 configuration-bound session。每个 accepted request 和每个 destination state 都必须进入 M1638 miter，最后只接受三个 distinct session 的 exact-order bundle。

作者检查发现一个不能隐藏的对象差：M1610 的 `parse_synthetic_identifier` 为旧 synthetic module3 设计，commit 坐标会固定成 module 3，而本 prefix 是 D0/module0。M1645 不改 M1610 scheduler，也不放宽 M1638；它新增严格 D0 ID 语法/坐标编码器，仍使用 M1610 的 exact packed schema，并在测试中显式证明 legacy=module3、actual=module0。

除逐 request 周期、port calendar 和 outstanding 对比外，M1645 还独立重建 packed address/commit digest 并与 compact 内部 digest 比较；M1539/M1610 cache 则在 miss/slot/content/age/tick/state digest 上锁步。RSS 同时采集 baseline/current/HWM，限制 absolute <2 GiB 且 HWM-baseline <512 MiB，每个 destination 检查。

CPython 3.6 和 3.10 均为 10/10 PASS，`--describe` 与 payload-free `--preflight` 跨解释器 byte-identical。合成测试只验证协议和 miter，其 cycles/bytes 明确为 non-paper synthetic 数字；未来 actual 输出仍必须标记 `independent_hammer_pending=true`。

下一步只授权另一作者执行 M1646 源码独立评审。M1646 之前禁止 payload、L2 execution、L3/full decoder、production、GPU/EDA 和任何论文数字。
