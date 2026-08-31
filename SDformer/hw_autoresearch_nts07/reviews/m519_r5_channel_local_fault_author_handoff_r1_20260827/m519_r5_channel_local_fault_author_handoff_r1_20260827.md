# M519 R5 channel-local fault decoupling 作者交接 r1

日期：2026-08-27  
状态：`AUTHOR_R5_COMPLETE__NO_EDA_RUN__STATIC_HAMMER_REQUIRED`

## 交付结果

R5 已按 sealed TIM-209 diagnosis 的首选方案实现，没有引入 request/response register slice：

- `protocol_error` 仍完整检测 sticky/request/response 三类 fault；
- request fault 当拍拒绝 request 和所有 bank issue，并锁 sticky fault；
- request fault 不再进入 `core_rsp_valid` 或 response retirement enable；
- 独立合法 complete response 在同拍可依 ready 精确接受/retire 一次；
- illegal response 同拍关闭 request/response 两通道；
- normal path 没有新增状态、端口或周期，K8/K1x8 RTL 和旧两份 full workload TB 未改。

新增 unit attack TB/SVA 以正常端口覆盖 12 类 directed attack/recovery；新 VCS runner 必须先跑
unit attack，再无弱化地重跑 r2 的 K1-vs-K1x8 与 K8-vs-K1x8 全 regression，并把五档旧 cycle
逐行作为 exact gate。

DC Tcl 已把 TIM-209/OPT-150 fail branch 改为显式 `exit 36`；`ungroup/compile*` 全部只在
PASS `else`；成功还必须产生 PASS-only terminal。新 DC runner捕获 child/monitor rc、INT/TERM
来源和 runtime latch，失败/中断树在移动 quarantine 前生成 inner manifest 与 outer seal。

## 身份

- recovery contract：
  `contracts/m519_r5_channel_local_fault_recovery_contract_r1_20260827.json`
  (`779180ed7ca889a92c83273476f6d70a970ed5f8a713e235fd18c4600919160a`)
- static request：
  `reviews/m519_r5_channel_local_fault_static_hammer_r1_REQUEST_20260827.md`
  (`82113c9ab1cb56e970ef4ab485f6a63ad7200667f6a68eafb5483fec1b2cc3d6`)
- VCS runner：
  `dc_handoff/scripts/run_vcs_m519_r5_channel_local_fault_exact_sha.sh`
  (`e6d7160b47b4f49827dcf7c65ef7036bb9139911b64de2992a0daec350897dc0`)
- DC Tcl：
  `dc_handoff/scripts/run_dc_m519_r5_channel_local_fault_three_axis.tcl`
  (`317bbc41ee4295455aac2bb0781570d89808e58885058287a9e0f5e2eb1157bf`)
- DC runner：
  `dc_handoff/scripts/run_dc_m519_r5_channel_local_fault_three_axis_exact_sha.sh`
  (`55b02fe51fc30b3ae5e92f68aaf8dae0f780f29855bab9a7667365ec4873b80c`)

合同 `exact_files` 已逐项核对 19/19。两个 runner 均 `bash -n`，contract JSON 可解析，10 个
R5 attack cover 均在 runner 中是非零硬门；没有运行 VCS/DC/PT/PTPX/Formality/开源 EDA。

## 下一步唯一合法顺序

1. 独立 agent 按 static request 做 receipt-blind/static hammer；
2. P0=0 时，只授权一次三阶段 VCS；
3. VCS 双 seal 后再做独立 receipt hammer；
4. 另建 DC launch admission，才可运行一次 K1→K8→K1x8 DC。

当前不能宣称 loop-free、functional PASS、DC/PPA、throughput/mm2、power/energy、完整 FC2、
system speedup 或 DATE headline。

`docs/359` SHA256 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
