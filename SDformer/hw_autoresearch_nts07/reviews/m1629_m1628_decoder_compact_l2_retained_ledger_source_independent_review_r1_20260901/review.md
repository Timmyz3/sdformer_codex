# M1629｜M1628 decoder compact L2 survivor source 独立评审

日期：2026-09-01

状态：`NO_GO_M1628_ACTUAL_L2_RUNNER_SOURCE__ONE_P1_SESSION_CONFIGURATION_BINDING_GAP__SUCCESSOR_REPAIR_ONLY`

评分：86/100；P0=0，P1=1，P2=1。不能进入 actual runner-source。

## 已确认修复

M1620 的 8 类 survivor mutation 在独立锤中全部被拒：较早 max-return 丢失、跨 destination future-return 消失、psum-ready 回退、cache 清空/无活动改内容、module/timestep/destination/output-block 越界、外报 kind/byte 账本、伪造 address/commit digest、手写 finish dictionary。

accepted request 已成为 count、bytes、transaction-address digest、commit digest、commit population、port calendar、active outstanding 和 psum-ready 的内部权威。cache predecessor 与 accepted-weight-request digest 也已绑定。genuine receipt 的 clone、tag mutation、重复 session、配置换序、跨配置 commit-stream 不一致和已消费 bundle replay 均被拒。

作者测试在创建本 review 路径前于 CPython 3.6/3.10 各 18/18 PASS。独立锤在两版本下输出 byte-identical。全程只使用 synthetic state/request；没有打开 ep34 payload，没有 actual L2/L3、attempt、release、GPU 或 EDA。

## 新 P1：session 配置可在 finish 前重标

`CanonicalPrefixMiter.configuration` 在完成全部 request/state 后仍可直接改写，而隐藏的 `issued` registry 只记 `session_identity -> owner`，没有记 session 创建时的 immutable configuration。

独立攻击创建了三个不同 owner 和三个不同随机 session identity，但三者实际都完整执行 `DENSE_TYPED_K8`。完成后、调用 `finish()` 前，将第二、第三个 owner 分别重标为 `BIT_EQUAL_SERVICE_K1X8` 和 `BIT_TYPED_K8`。三个 HMAC receipt 都是真实对象、身份不同、报告顺序正确、commit digest 相同；`validate_bundle` 最终返回 `True`。三个 receipt 的 dense-only coverage 甚至全部为 true，却没有被配置策略拒绝。

因此当前接口证明了“3 个 fresh receipt”，但没有证明“3 个 frozen configuration 各自执行了一次”。这正落在 M1620 第三个 P1 的核心边界，不能授权 actual runner-source。

修复应在 session 创建时把 `(exact owner, initial configuration)` 存入隐藏 registry，并在每个 request/state、finish 和 authenticated payload 处核对；bundle 还应要求配置精确 coverage policy：dense 行为只允许第一配置为 true，两个 bit 配置必须为 false。successor 双运行时测试必须加入本 relabel 攻击。

## P2

RSS 仍是 reference/compact 分别 bounded validation，而不是 `PREFIX_EXACT_FIELDS` 的 pairwise equality。该边界已在 M1620 披露，暂不阻塞下一版 source 修复，但未来 runner/result receipt 必须明确，不能把 RSS 相等写成功能 exactness。

## 裁决边界

只允许另起名字编写 source-only successor，修复 session initial-configuration binding。禁止 M1633 release、actual runner source、payload、L2/L3、pilot/production、周期/流量/加速/能量/Table-A 与论文结果。successor 必须再过一次 P0=0/P1=0 的不同作者评审。
