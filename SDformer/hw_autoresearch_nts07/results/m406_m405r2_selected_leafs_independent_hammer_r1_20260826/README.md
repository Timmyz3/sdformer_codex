# M406：M405r2 selected leaf 独立打铁

结论：`FAIL_NEEDS_M405_R3_INTEGRATION_REPAIR`，72/100，P0/P1/P2=`0/2/5`。
M405r2 的两个 leaf directed VCS 结论可以保留，但不能通过其“独立评审 P1=0”
晋级门。独立 VCS 已把两个 integration P1 复现为确定行为。

## 两个 P1

1. `configuration_live` 直接等于 matcher 的 `phase_active`，最后一行接收即拉低。
   M401 的冻结周期顺序却是 config、matcher、seal、tile DMA、tile0/tile1 replay。
   因此 matcher 结束后的合法 PWP replay 被当作无配置攻击并触发 sticky fault。
   修复必须把 row-ingress lifetime 与 config ownership 解耦，新增显式
   `phase_release/phase_done`，直到 seal、两 tile replay 和 adapter drain 全部完成才
   释放 centers/bitmap/tag 并允许下一个 config。
2. shell 的聚合错误不是全局 fail-closed。`shell_fault`/adapter error 只闸 PWP 路径，
   matcher 的 config/row/result 仍直连。反例在 `protocol_error=1` 后仍看到并接收
   result；同时出现 `pwp_low_valid=1,pwp_low_ready=1,pwp_low_accept=0`。
   R3 应建立 sticky `global_fault/global_safe`，闸所有外部 ready/valid/accept，并加
   fault 后永久静默 SVA。

## Leaf 与既有运行结论

- FIFO full 时 simultaneous push/pop 的 head/tail 顺序、wide 高侧车验证后才可见、
  narrow int8 sign-extension、wide `zero_extend(low8) + signed(high4)<<8` 均未发现
  leaf P1。
- q32 pass0/pass1、`popcount<2` fallback、相等时最低 global ID、单行 scratch 和
  顺序也未发现 leaf P1。Directed 账本为 64 pass0、61 pass1、64 output；固定 `+2`
  仍是 M401 的 fill/drain 周期，不是 leaf TB 测量值。
- all-wide no-gap 使用两 record 预填，只能证明 steady state。它没有直接偷走周期，
  因为 M401 有命名的两周期 tail；仍需 integrated M384→adapter miter 证明 startup 和
  mixed-width 不产生额外 gap。
- r1 保持失败是正确的：VCS 对 TB `$fatal` 返回 0，但 runner 的 Fatal 扫描抓住了。
  r1/r2 的不完整目录不可引用；r1b exact-SHA、日志扫描和双层 manifest 均通过。
- 当前 M384 已是 q32/640 B 新 SHA，但现有 VCS/DC/FM/PT 收据绑定旧 576 B SHA；
  M405r2 对此明确保持 false，必须刷新回归。

## 独立复现

使用 Synopsys VCS V-2023.12-SP1，编译和仿真返回码均为 0。关键输出：

```text
PASS M406 REPRO config_live_after_last=0 legal_replay_fault=1 ready_accept_split=1 result_visible_after_global_fault=1 result_accept_after_global_fault=1 P1_CONFIG_LIFETIME=1 P1_GLOBAL_FAILCLOSED=1
```

测试文件为 `tb_m406_integration_lifetime_failclosed_repro.sv`，日志在 `vcs_repro/`。
本评审不修改 M405 RTL、合同、M384 或 `docs/359`，也不产生新速度；M401/M402 的
1.1563713549830412x 仍只是四个 Conv 的 trace-cycle candidate。
