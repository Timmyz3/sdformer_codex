# M528 r4 recovery source-only static hammer request

请独立锤审 `reviews/m528_single_port_same_ledger_recompute_author_handoff_r4_20260827`，全程 source-only。禁止执行 analyzer 的任何模式、spawn self-test、preflight/production runner、CPU production、EDA、GPU 或 RTL。

本轮不是重复 r3 的封条检查。必须证明：真实模块名 normal import 可供 spawn pickle；exact `worker_init` 只打开 ledger，`worker_phase` 只 pickle-check 不调用；admission strict JSON + 自身双封；两个 runner 直接解析 static/receipt/hammer 的 PASS/P0/P1/身份/授权；R3 withdrawal 与 wrapper red-team 同时绑定；R2 attempt/quarantine 现场内外封与 canonical absence；以及 frozen compute/coordinate/resource gate 不变。

解释器身份必须是 `/opt/anaconda3/envs/pytorch310/bin/python`，SHA `9f78cd4299cf399449101f35b6d6ae826441657b3853728da50384b406242115`；不得用系统 Python 3.6 替代。

若通过，`review.json` 必须严格采用 request JSON 给出的 schema/status/verdict/identity/authorization key 形状，设置 `withdrawn=false`、P0=0、P1=0。Static review 只允许 root 再签一次 non-production preflight admission，不直接授权执行，更不授权 production。
