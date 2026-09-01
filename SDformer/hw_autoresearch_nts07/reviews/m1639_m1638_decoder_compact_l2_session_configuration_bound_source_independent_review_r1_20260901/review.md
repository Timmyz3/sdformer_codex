# M1639｜M1638 decoder compact L2 session-configuration-bound 独立评审

日期：2026-09-01

状态：`PASS_M1639_M1638_CONFIGURATION_BOUND_SOURCE__AUTHORIZE_ACTUAL_PREFIX_RUNNER_SOURCE_AUTHORING__NO_EXECUTION`

评分：99/100；P0=0，P1=0，P2=2。

## 结论

M1629 发现的 `P1_SESSION_CONFIGURATION_RELABEL_NOT_BOUND_AT_CREATION` 已被 M1638 有效修复。隐藏 issued registry 现在保存 `(exact owner, immutable initial configuration)`，并在 request、state、finish payload、finish authentication 和 bundle inspection 五层重新校验。公开可变 `configuration` 不再能重标一条已执行 session 的身份。

合法三配置 bundle 的覆盖策略严格为 `[(True,True),(False,False),(False,False)]`，对应 `DENSE_TYPED_K8`、`BIT_EQUAL_SERVICE_K1X8`、`BIT_TYPED_K8`。三条实际 dense session 中，后两条在 finish 前重标为两个 bit configuration 时均被拒绝；非 dense receipt 伪造 dense-only coverage 也被拒绝。

## 回归与攻击

本评审不仅运行 M1638 作者的 20 项测试。独立 hammer 重放了：

- M1620/M1628 的 8 类 survivor：早期/跨 destination return、psum-ready、cache continuity、request scope、count/byte ledger、address/commit digest 和手写 finish row。
- M1629 的 6 类 receipt/bundle 攻击：clone、duplicate、reorder、tag mutation、shared-commit mismatch 与 consumed replay。
- 10 类初始配置攻击：request/state 的 owner 与 payload 重标、finish payload/owner 重标、dense×3 重标、bundle payload 配置重标与 coverage 重标。

CPython 3.6 和 3.10 都通过作者 20/20 测试，并产生完全相同的 25 类独立攻击拒绝结果。

## 授权边界

当前只授权下一阶段编写 fail-closed actual-prefix runner source。不授权 runner 执行、release/attempt 创建、payload 打开、L2/L3、pilot/production、GPU 或 EDA。本评审未读取 ep34 payload，未产生 cycle/traffic/energy/speedup 或论文结果。

P2 保留两条：RSS 仍是两路分别有界而非 pairwise exact；当前证据仍是纯 source/synthetic，不能替代实际 prefix 执行。下一个 runner/result 必须继续明确这两个边界。

`docs/359` 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`；未修改 `ucli.key`。
