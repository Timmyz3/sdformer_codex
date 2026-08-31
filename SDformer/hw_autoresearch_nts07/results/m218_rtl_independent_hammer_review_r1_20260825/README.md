# M218 RTL independent hammer review

结论：**92/100，P0=0，GO service-only DC**。不允许将本里程碑外推为完整
FC2/FFN、macro-aware PPA、physical/system speedup 或 DATE headline。M216+M218
联合综合必须等 connected RTL 和 cycle miter 后再开。

我独立校验了 sealed VCS 目录中 61 个 manifest 文件，contract/RTL/SVA/TB/
filelist 均与 exact SHA 一致，`docs/359_DATE终局冻结_20260813.md` 仍为
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
另使用 Synopsys VCS V-2023.12-SP1 从同一份 pinned 源码独立编译和重跑，
compile/sim RC 均为 0，无 warning/error 和 assertion failure，PASS 收据与 sealed
run 逐字相同。

独立守恒重算得到 34 group / 204 request / 204 response / 204 context
write / 864 active bank-slice read / 102 result beat / 5 done。K8 与同一八个
source 串行 K1 的 Acc24 结果 bit-exact。TB 的非退役 slot/context 检查为 0
冲突，命中 FIFO4、O8、73 次同拍 slot 复用、16 次同拍 context 复用、
OOO 退役、request/result stall。A/flush/B 中 stale-A 在 flush 期被丢弃，B 以
1 group / 6 request / 6 response / 6 context / 6 result / 6 read 从零上下文
完成数值 miter；随后 stale-A 以及 wrong epoch/generation/tag/mask、delayed
duplicate、1,024-cycle ack timeout 均 fail closed。

未发现阻止 service-only DC 的问题。主要 P1 是：尚无 M216↔M218 connected
timeline；验证仍是 directed 而非 formal/random exhaustive；定向权重只覆盖
[-15,+15]，尽管独立位宽证明 signed9/10/11 可容纳全 INT8 八路和；
`frontend_done_had_event` 仍信任上游；尚无 DC/Formality/PT/SAIF/PTPX 和
SRAM macro。service-only DC 必须明示 18,432-bit context 是寄存器实现、response
payload 为 1,024 bit，并严格标为 pre-macro logic/register 结果。

完整评分、P1、数字与 DC admission 条件见
`m218_rtl_independent_hammer_review_r1.json`；可复跑独立校验见
`audit_m218_rtl_independent.py`。
