# M1577｜M1573 decoder fresh-worker gate successor 独立 hammer

日期：2026-09-01（Asia/Shanghai）  
裁决：`NO_GO_M1577_M1573_ONE_SHOT_RUNNER_AUTHORING__FRESHNESS_RSS_AND_RESULT_BINDING_NOT_ENFORCED`

M1573 的固定身份和作者外封均正确。CPython 3.10.18 与 3.6.8 均重放
作者 9/9 测试；clean synthetic miter 保持 configuration、resource digest、
cycles、request/kind/byte counts、address digest 和 commit digest exact。对这些
字段逐项注入差异也都会 fail closed。严格 8 GiB current/peak RSS 的等号边界、
product 配置、production release，以及 actual/pilot/M1570 retry CLI 均拒绝。

但 fresh-worker 主合同没有被实现成可验证的门。hammer 用不访问生产 payload
的内存 witness 替换 actual-call seam 后，在同一解释器连续执行两个配置均成功，
两份结果仍自报 `fresh_exec_required=true`。另一个 witness 完全不调用 dual-RSS
gate，M1573 仍接受 `gate_calls=0` 的结果。第三个 witness 返回 forbidden product
configuration 和伪造 resource digest，也被 wrapper 原样接收。这三个反例在两套
Python 完全复现；hammer 没有打开 ep34 payload，更没有执行 actual pilot。

因此当前源码只能证明 clean pinned synthetic 路径的 gate replacement 不改变
投影，不能证明 future actual worker 是 fresh、至少执行过一次 RSS gate，或返回
结果仍绑定冻结 hardware projection。`fresh_exec_required` 现在只是标签，不是证据。

最小 successor 应把 exact upstream actual-call 与预期投影绑定进 clean-import
closure；在首次 entry 前消费私有 per-process one-shot token；拒绝第二次同进程
entry；要求 `gate_calls>0`；并逐字段验证请求配置、resource digest 与完整 frozen
hardware projection。修复后必须再次由不同作者双运行时 hammer。M1577 只授权该
source-only 修复，不授权 runner、actual pilot、M1570 retry、GPU/RTL/EDA 或论文数字。
