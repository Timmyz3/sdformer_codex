# M1613｜M1609 registered-fault directed source author handoff

日期：2026-09-01

状态：`PASS_AUTHOR_SOURCE_ONLY__READY_FOR_DIFFERENT_AUTHOR_EXACT_SHA_HAMMER__NO_VCS_NO_EDA`

## 已完成

- 新 filelist 仅含 M1609 successor 与 M1613 TB 两行；冻结 M214 predecessor 不在 filelist 中。
- 合法 terminal packet 在采样沿前同时接受 raw 与 descriptor；采样沿后故意保留 `raw_valid`，明确覆盖组合 `illegal_request=1` 但 `protocol_error=0/fault_q=0` 的无假脉冲缝。
- 非法 header 与非法 raw 均在呈现周期 `ready=0/accept=0/protocol_error=0`，并要求采样沿后 `fault_q=protocol_error=1` 且 sticky。
- 所有 posedge 后判断均使用 M1601 模式的 `#1ps` settled sampling。
- runner 只有一次 VCS compile 与一次 simv 预算；没有 DC/PTPX 路径。
- runner 不能仅凭调用者提供自身 SHA 启动：未来必须存在并校验 M1617 different-author hammer 与 M1618 release，且调用者同时 pin runner/release SHA。
- EDA 冲突门只扫描同 UID，并排除 runner 祖先进程；其他 UID 的长期 `simv` 不会误阻。

## 作者验证

- Python 3.6：9/9 PASS。
- Python 3.12：9/9 PASS。
- runner `bash -n`：PASS。
- `git diff --check`：PASS。
- VCS compile=0，simv=0，DC=0，PTPX=0。

## 下一门

不同作者必须审阅 `handoff.json` 中列出的六个 source 文件及所有冻结 authority，生成 exact-SHA M1617 hammer；随后另行生成绑定 runner、contract 与 hammer review SHA 的 M1618 release。两者未齐前 runner 会硬失败，禁止执行 VCS。

即便未来 directed VCS PASS，也只证明 compactor-local registered-fault 时序；M1611 要求的 M216/service outer error OR-chain 仍需独立 integration VCS，不能由本结果外推。
