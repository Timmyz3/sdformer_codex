# M970 独立 inert release hammer

结论：**100/100，P0/P1/P2 = 0/0/0；`GO_ONE_D2_D3_10K_PAIR_ONLY`。**

M969 只释放一次 D2/D3、sample0、A1_OSG、t0、10K pair。`release=true` 只表示该 pair 在 M970 三层身份由运行方精确提供后可执行；`launch_now=false`，本评审未运行 runner、未执行 prefix、未创建 attempt/result。

## 已核边界

- schema、status、release、launch_now、max_attempts 全部正确。
- exact rows 恰为 D2 与 D3 两行；D1、100K、full-row、production、EDA/GPU/remote 全部禁止。
- exact gate 绑定 M768/M861/M890/M896 的 14 个 schedule/address/commit/terminal/port 字段。
- D2/D3 首个 source-fetch 分别为 231,600/465,600 requests，因此 10K 以及可能的 100K 都只是 `SOURCE_FETCH_ONLY`，不覆盖 contributor、commit 或 full row。
- 100K 只能由 sealed 10K result 的 2× memory/timeout projection 建议，并仍需独立 release 与 release hammer；不能自动执行。
- M961 contract、runner、driver、M946、M950、M968 review/manifest/outer 和作者 release preflight 递归身份全部通过。
- 只读 driver validation 在 M970 缺失时停在 sealed-directory 门，且 attempt/result 仍不存在。

运行安全协议已具备：M970 exact review/manifest/outer 三 SHA 必须在运行时提供；attempt 在 prefix 前消费；结果独立 sealed 发布；目标存在时拒绝；失败 stage 隔离；启动前检查磁盘、MemAvailable 和 commit headroom。

本 review 自身仍不是 paper result，不授权 100K/full-row，也不能声称 decoder complete 或 system speedup。
