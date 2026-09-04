# M1615｜M1610 decoder compact L0/L1 独立复核

日期：2026-09-01（Asia/Shanghai）  
裁决：**PASS，但只准进入“另写 L2 source/contract”这一个下一步；M1615 不授权打开真实 ep34 payload。**

## 复核结论

M1610 在其明示边界内成立：它是冻结 M1539 scheduler/cache 的**表示替换**，不是新调度器，也不是性能结果。24 个 port calendar entry、129 个 outstanding return slot、9-entry weight cache、8 组 address/bank scratch 均为固定容量；三个配置保持不变，`PRODUCT_CAPTURE_TYPED_K8` 继续拒绝。

双 Python 均通过编译、作者测试和独立 hammer：CPython 3.10.16 与 3.6.8。作者测试为 12 个 case/config row；独立 hammer 另用了 16 个确定性 8x2x2 pattern，在三个配置上完成 48 次 miter、共 7,722 个请求。逐请求 issue/return/dependency/port-ready、prefix/final cycle 与 count、per-kind bytes、packed address、packed commit、port calendar、active outstanding multiset 全部 exact；BIT equal-service 与 BIT typed 的 compute population 相等，三配置 packed commit stream 相等。

压力覆盖不是只看常量：external queue 实际达到 16-entry 满容量并触发 1 次 full wait；共享 1RW 路径触发 18 次 serialization；9-entry cache 发生 8 次 eviction 后仍与 M1539 的 key/slot/victim/age/tick exact。第 9 个 address scratch entry、禁用配置和 production release 均被拒绝。

静态检查覆盖 19 个 compact hot-path 函数，未发现按请求增长的 dict/set/list comprehension、append/pop/insert、JSON/string-format digest 或 Python `hash()`。系统调用审计只看到 M1539 source、docs/359 与 M1572 seal/review 等授权元数据；没有打开 capture、NPZ、checkpoint 或结果 payload。

## Claim 边界

- 没有读取真实 ep34 payload；没有跑 L2 actual prefix 或 L3 full diagnostic。
- 没有跑 pilot、production、EDA 或 GPU。
- 没有产生可用于 Table-A 的 cycle、traffic、speedup、energy 或 PPA。
- packed digest 是对 M1539 row adapter 的 exact binary-stream miter；它不冒充 legacy JSON transaction digest。
- M1610 目前只闭合 numeric scheduler 与 cache 的 L0/L1，不等于已实现完整 compact decoder generator。

## 唯一下一步授权

仅授权作者新增一个独立命名的 **L2 canonical-prefix source 与 contract**，绑定本次审阅的 M1610 SHA，并保证 canonical prefix 中间 destination 的 cache/calendar 历史不被跳过。该 L2 source 必须再次独立审阅后，才可获得一次真实 prefix 执行授权；L3、D0/call0 full diagnostic、pilot、production 与论文性能数字继续关闭。

作者三个文件和 docs/359 均未修改。
