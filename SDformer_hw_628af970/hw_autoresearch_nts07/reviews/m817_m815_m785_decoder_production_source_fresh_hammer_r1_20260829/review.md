# M817｜M815 decoder runner recovery receipt-blind source hammer

## Decision

`NO_GO_M815_TRUE_RELEASE__P1_1__ADDITIVE_DELEGATION_ATTEMPT_STATUS_REPAIR_REQUIRED`，92/100，P0/P1/P2 = 0/1/0。不得为 M815 创建 true release，不得运行 formal runner，也不得消费正式 attempt。

M815 已经修好了 M811 指出的 failure-trap 时序：attempt 成功发布后立即设置 `started=1` 和 `ATTEMPT_PUBLISHED_POSTCHECK`，再做 fallible postcheck。独立故障注入得到 attempt consumed、0 schedule row、canonical result absent，以及精确四成员双封 quarantine；collision 和 symlink 攻击均 no-clobber。

## P1 blocker

M815 runner 第 170 行写入的 attempt status 是 `CONSUMED_IMMEDIATELY_BEFORE_M815_PRODUCTION_REPLAY`，M815 自己的 consumed-attempt validator 也接受该 token。可是 M815 的周期边界在第 406 行直接调用冻结 SHA `2b273d...6736d0` 的 `M809.run_production()`；M809 会在任何 schedule row 和 `output.mkdir` 之前重新读取 attempt，并硬校验 status 必须是 `CONSUMED_IMMEDIATELY_BEFORE_M809_PRODUCTION_REPLAY`。

Receipt-blind 临时夹具动态复现了 `attempt receipt identity drift`，schedule row 为 0，output 不存在。因此当前 M815 如果签发 release，会永久消费唯一 attempt 后确定性失败。failure receipt 会被正确保存，但 production 永远无法成功。

最小修复是新建 additive identity，让 runner 与 M815 validator 使用冻结 M809 body 可接受的 parent-compatible attempt token，并增加一个不运行 schedule 的 preproduction traversal 测试，证明已穿过 parent receipt check。不得修改 M809、伪造 M811 PASS 或直接签发 M815 release。

## Passed evidence

- request、candidate、contract、driver、runner、tests、author handoff 及双层 sidecar 全部重算通过；严格 JSON duplicate/non-finite 攻击通过。
- Python 3.10 与 Python 3.6 均通过 compile、self-test、10/10 source-only tests 和 candidate validation；`bash -n` 通过。
- runner 顺序为 publish → started → phase → postcheck → consumed preflight → production。
- 独立 postpublish 故障注入得到精确 `failure.json + driver.log + SHA256SUMS + SHA256SUMS.seal.sha256`，0 row、result absent；重复目录、destination symlink、log symlink 均拒绝且不改证据。
- M811 保持 NO-GO/release=false；M809 文件与 M798 attempt 未变；40+120、T10、96 lane、240 KiB、Acc24、3 ns、192 B/cycle、D1 与 headline 分母均未改。
- M815 release、attempt、result、failure artifact 均不存在；docs/359 SHA 仍为 `dedde7ce...bdfc4`。

本次仅封 source review，不含 production cycles、speedup、decoder completion、Table-A、VCS/EDA/PPA/energy。
