# M168 H67 FC2 K-bank multi-source DSE r1

M168 在冻结 M51 ten-sample payload 上逐 token 重算 12 个 H67 `mlp.fc2`
输入的八个 `input_channel mod 8` bank 占用。每个 bank 每拍最多读一个事件，且
每拍全局最多接纳 K 个不同 bank 事件，则每个 token 的最小服务周期严格为：

```text
max(max_bank_event_count, ceil(active_event_count / K))
```

120 个 payload 的 SHA、大小与 popcount 均由 analyzer 重新检查。合计
3,502,080,000 个输入位、143,894,510 个事件，活动率 4.108830%。按实际
`Cout/96` 加权后的 K1/K2/K4/K8 周期分别为 412,900,394 / 208,127,456 /
106,536,803 / 70,657,362，所以 K1/K4 为 **3.875660x**，K1/K8 为
**5.843700x**。

优先实现 K4。它在真实 bank 冲突下达到理想 4x 的 96.9%，每拍需要四条独立
96-byte INT8 weight row（3072 bit）和每个输出 lane 的四项 signed reduction。
K8 需要 6144 bit/cycle weight payload 与更深加法树，暂时只作 DSE 上界。

这些比值只是 exact-payload、bank-feasible service boundary。它们尚未包含 compactor、
SRAM macro、weight response、Acc24、BN2、residual、反压或物理时序，因此不能称为
RTL/FC2/FFN/系统加速。M168 下一步必须实现 K4 event compactor + weight response +
accumulator，并在 VCS 做 K1/K4 exact-output miter，再做 matched Synopsys DC。

