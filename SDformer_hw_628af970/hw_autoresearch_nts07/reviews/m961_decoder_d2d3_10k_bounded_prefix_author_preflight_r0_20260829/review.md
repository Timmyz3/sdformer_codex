# M961 author source preflight（非独立评审）

M961 source contract、driver、one-shot runner、checker 与 6 项无前缀测试通过。没有创建 attempt/result，也没有运行 D2/D3 10K。

首个 transaction 的冻结 request count 为 D2 231,600、D3 465,600，因此 10K 和后续可能的 100K 都只覆盖 source-fetch。M961 的合法用途只是用两条 10K exact scheduler prefix 决定是否设计独立 100K scheduler-prefix release；不能据此声称 contributor、commit 或 full-row scalability。

当前 runner 缺少 exact M969 release SHA 和 M970 release-hammer 身份时立即拒绝。本目录不是 source hammer，不授权执行。
