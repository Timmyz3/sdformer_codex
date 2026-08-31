# M786 请求：M785 decoder physical-residency source fresh hammer

请先完整阅读 M781 FAIL78，再对 M785 做 receipt-blind、source-only hammer。M777/M768 已冻结；本请求不授权 M686/M699 replay，不允许生成 decoder cycle、speedup、result、Table-A，也不允许 RTL/VCS/DC/GPU/remote。

必须优先复现 M781 两个 P0：容量一的 dirty psum victim 在 external evict return 前不得被 replacement psum read 覆盖；九个 weight tile 必须有真实 key→slot→bank/local-row 映射，不同 output block 不得别名。所有 backing/refill 必须同时收费 local SRAM 端口，且依赖要落到 return cycle，不得以 Python transaction 顺序代替硬件因果。

M722 的两条 line-buffer stripe 与 M785 的 100 条 global-vector stripe 明确不等价。请分别攻击：M722 仍要严格校验自己的 stripe/byte/offchip/完整计划，但只准作 contributor/group oracle；M785 storage oracle 必须独立固定、可复算并拒绝 stripe、partition byte、offchip span 注入。

即使 ≥95/100 且 P0/P1=0，通过也只说明 source candidate 可申请另一份 production release，不直接授权 production。
