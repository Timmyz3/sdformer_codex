# M668｜M663 registry r7 fresh independent hammer

## 裁决

**78/100，NO-GO methodology。P0=0、P1=3、P2=2。**

canonical 没有被污染：仍是 `0 authority / 0 bundle / 0 eligible row / headline=false / analytical=false`。NO-GO 针对未来生产 PPA 的准入方法，不代表已有生产数字被错误准入。

## 通过项

- 作者包与请求双封重算通过，冻结源码 SHA 全部匹配。
- Python 3.6.8 与 3.10.18 均为 18/18 PASS；两个解释器 canonical CLI 输出一致。
- 仓库真实 native 报告直接重算：DC area `0.03545917173 mm2`，PT setup/hold WNS `0.841061/0.017869 ns`，PTPX total/leakage `6.25380802/0.0342430696 mW`。
- 三行 numeric wrapper、wrong Design、跨 row 复用、错误 M527 config/scope/run/argv/unit、漏 leakage 和 total 算术漂移均拒绝。
- 六行 PPA 图根数为 `1 extractor + 6 receipt + 6 run manifest + 30 report = 43`。
- exact operator scope 为十项：patch、Conv2d、ConvTranspose2d、fc1、fc2、dynamic BN、ATLIF、attention、prediction head、全部前后处理/完成。

## P1

### M668-P1-01｜macro/capacity 未绑定配置资源

把 manifest 与 `.ds` 一起一致改成 `TOTALLY_WRONG_UNBOUND_MACRO` 后仍通过 `_validate_extraction_receipt`。更关键的是 `.ds` 根本没解析 depth、width、bit capacity、port 与 instance multiplicity，无法证明它对应公共资源中的 240 KiB。

修复门：common-resource/configuration 必须给出 ordered memory inventory；parser 必须从 compiler 证据解析组织/容量/端口，逐宏和总字节闭合。

### M668-P1-02｜library/corner 是自由字符串

PTPX/setup/hold library 填 `UNPARSED_*`、五类 corner 全填 `UNPARSED_FAKE_CORNER_*` 仍通过。当前只解析 DC 与 SRAM library；`.ds` 已解析的 Slow/0.9V/125C 甚至没有和 manifest SRAM corner 比较。

修复门：加入 native `report_environment`/library 报告，解析 DB 与 operating condition；SRAM PVT 必须与 `.ds` 逐字段一致。

### M668-P1-03｜缺工具生成 provenance

独立手写、仅伪装成 Synopsys/TSMC grammar 的五份文本可完整进入 receipt。run manifest 只绑定“提取器 argv”，没有绑定生成这些报告的 dc_shell/pt_shell argv、exit status、netlist/DDC、SDC、library DB、SAIF、parasitic 或 tool log。

修复门：新增 tool-run receipt，把上述根全部并入 run ID；否则“native grammar”不能升级为“native tool run”。

## P2

Standalone extractor 在 `resolve()` 后检查 `is_symlink()`，因此 symlink 身份已丢失；也接受 `reports/../reports/...`。builder 的 inherited secure path 当前会拒绝二者，所以不升级为 P1，但 extractor 自身需在 resolve 前拒绝 symlink、绝对路径、`.`/`..`。

## 允许的下一步

只能做 r8 非生产修复并再次 fresh hammer。不得将本评审解释为 production/Table-A/headline 准入，也不得因为 canonical 为零就跳过三个 P1。
