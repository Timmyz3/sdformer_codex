# M1834｜M1832 C2 Formality + 双角 PT 源独立打铁

## 裁决

**FAIL CLOSED，84/100，P0=0，P1=2，P2=1。** M1834 不授权 M1832 attempt、Formality、PrimeTime、license query 或 M1836 release。本次只读审阅没有运行任何 EDA/仿真器，也没有创建 attempt/result/release，未修改作者源、M1811/M1830、canonical 或 docs/359。

M1832 的主体方向是对的：K8=`ARCH_MODE=0`、K1×8=`ARCH_MODE=1`，两轴各自使用独立 mapped V/SDC/SVF；成功路径意图恰好运行 2 次 Formality 和 2 次独立 PrimeTime，无第三轴、无自动重试。PT 使用 `ssg0p9v125c` slow-max、`ffg1p05vm40c` fast-min、`set_min_library` 与 OCV；setup/hold 负 slack 会原样写入结果，不做例外、ECO、hold repair 或夹零。

但 source identity 和 release authority 尚未闭合，不能发放一次性执行许可。

## P1-1：13 个 live RTL 没有在 launch 时逐项锁 SHA

runner 只锁了 13-row filelist 的 SHA，并检查行数/唯一性；它没有把 filelist 解析出的 13 个 live RTL 与 M1830 已封存的 `source_identity` SHA map 逐项比较。Formality Tcl 随后从工作树重新解析路径并读取 live RTL。

当前 13 个源的 SHA 与 M1830 **此刻一致**，但这不是 launch-time fail-closed：在 M1832/M1834/M1836 封存之后，只要修改某个 RTL 而不改 filelist，future run 就会验证另一个 reference identity。即使改动形式等价，Formality 仍可能 PASS，因此“等价”不能替代“源身份精确”。

修复要求：另起 superseding source identity，把 13 个 `path -> SHA256` 写入 contract/runner，attempt 之前逐项 `exact_regular()`，并写入 sealed input identity；checker/tests 必须攻击单文件内容漂移。

## P1-2：M1836 无法绑定 M1834 的完整双封

runner 从 caller env 读取 M1834 的 review/manifest/outer 三个 SHA 并验证目录，但它要求 M1836 的 `identity` **只包含** `m1834_source_review_sha256`。由于 release identity 做 exact-dict 比较，未来 M1836 即使想加入 manifest/outer 两个 pin 也会被拒绝。

这使 release 不是完整授权源：manifest/outer 由 launch caller 临时选择，而非由 release 封存。`review.json` 本体虽有 exact pin，完整独立审阅目录的成员集合却没有由 release 授权。

修复要求：superseding runner/release schema 必须在 release identity 中同时要求 `review SHA + manifest SHA + outer seal file SHA`，并攻击缺失、跨角色串线、swap 和 caller-only substitution。

## P2：PT raw-result 报告门仍需收紧

PT Tcl 会生成 exceptions/design/wire-load 等报告，但 `verify_pt()` 的 required set 没有要求这三份；它也只解析 setup/hold finite slack，没有语义解析 check_timing、analysis coverage 与 constraint violators。由于 raw result 仍标为 pending independent result review，这项可以留到 result hammer 阻断，不单独构成当前 P1；但任何后续审阅都必须拒绝缺失/空报告，并如实报告 unconstrained、untested 和 constraint violations。

## 保留的正确边界

- M1811/M1830、两轴 mapped V/SDC/SVF、工具与库身份都已 exact-pin；六条 artifact path 互异。
- Formality 每轴 fresh elaborate 参数化 reference，并消费该轴独立 SVF/netlist；非等价会 fail closed。
- PT setup/hold 负 slack 不阻止 raw artifact 发布，但 `closure=false`，不得隐藏或改网表。
- M1832 目前没有 attempt/result，M1836 release 也不存在。
- 不得由本源包推导 equivalence、PT closure、power、energy、cycle/system speedup、paper PPA 或 headline。

下一步不是运行 M1832，而是由不同作者另起 superseding source identity，修完两项 P1 后再做新的 severity-zero 独立源打铁。
