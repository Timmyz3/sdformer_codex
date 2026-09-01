# M1653｜M1652 C2 resource-gate successor 独立评审

结论：`FAIL_M1653_M1652_C2_RESOURCE_GATE_SOURCE_HAMMER__NO_RELEASE`，91/100，P0=0、P1=1、P2=1。M1652 禁止生成 M1654 release，禁止启动 DC。

预期的结构差异本身成立：相对 M1634，只把 commit headroom 从 67,108,864 KiB 降到 50,331,648 KiB，并更换运行 namespace、补充审计 metadata。12-row filelist、Tcl、SDC 均 byte-exact；K1/K8/K1x8 三轴仍分别 fresh `compile_ultra`；96 GiB MemAvailable、16 GiB SwapFree、same-UID=0、license、输入/tool SHA、setup/DRC/artifact/result 谓词和 diagnostic-only hold 边界均未变。M1635/M1636/M1641 authority 及其 seals 也已复核。

阻塞问题在 runner 的真实内联预检。它要求：

```python
c['authorization'] == {
    'dc_runs_now': 0,
    'future_dc_shell_runs_max': 3,
    'all_other_eda_runs': 0,
}
```

但 sealed M1652 合同中的 `authorization` 还有 `vcs_runs/pt_runs/formality_runs/ptpx_runs/gpu_runs/remote_runs/attempts_created_now/retry` 等字段，所以整字典比较恒为 false。独立 hammer 抽取并执行 runner 中的原始 heredoc，CPython 3.6 与当前 CPython 均复现 `returncode=1 / AssertionError`。失败发生在 lock、attempt 和 DC 之前，不消耗 EDA，但也意味着该 source 永远不可能合法启动。

作者静态测试两解释器均 13/13 PASS，却只逐字段验证合同，没有执行真实内联 preflight，因此漏检。独立 hammer 在两个解释器输出 byte-identical，并拒绝 44 类资源门、三轴、公平性、authority、artifact、result、retry 和 claim mutation。

处置：保留 M1652 为 fail-closed 负样本；不得制作 M1654。新 successor 应逐字段检查或比较完整精确字典，并把真实内联 preflight 加入回归。本评审没有创建 release、attempt、work、result，也没有启动任何 EDA。
