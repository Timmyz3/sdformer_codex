# M899 M896 decoder RUN-GTLS source fresh independent hammer

## 裁决

**PASS100，仅准入 bounded source exactness 和 combined-state projection gate。** M896 在封存身份下通过 synthetic/real 1K、10K 和 real D0/A1/t0 100K exact miter，也通过 maximal run、counted AP、priority、liveness、shard 和 hash-domain 攻击。本评审只允许另一作者编写一份 fresh inert full-first-row runtime-gate release，不允许执行 full row。

## Exact 结果

- pytest：11/11。
- synthetic 1K/10K 和 real 1K/10K：M768 = M861 = M890 = RUN-GTLS。
- real 100K：M861 = M890 = RUN-GTLS；100,000 expanded requests，24,852 compressed transactions，live-token peak 50，active-service 仅 1,436 个 maximal run。
- exact 覆盖每个 scheduled endpoint、expanded-address/commit/terminal hash、port calendar、same-cycle response-slot 和六类 cycle priority。terminal readiness SHA256 为 `a55d8cfa67f47863bc561323d01c674f1dd8d35555f3a972ab78d72bf44891ee`。
- 攻击覆盖 touching/nested run 合并、non-touching 保留、21,120 组 counted-AP issue recurrence、same-cycle priority、提前/退役后 liveness、nonterminal dependency、one-shot ledger、deterministic shard 和 compressed/expanded hash-domain 隔离。

## 512 MiB 是 state gate，不是 RSS gate

real 100K 独立子进程从 live in-process objects 测得 combined live-event state = **1,274,626 B**，未用 serialized/compressed file size。按 `ceil(1,274,626 × 38,672,612 / 100,000)` 外推为 **492,931,168 B = 470.096 MiB**，低于 512 MiB 门 **43,939,744 B = 41.904 MiB**。

子进程 wall time 3.51 s，peak RSS 907,140 KiB。RSS 包含 bounded input/reference Python objects，必须与上述 scheduler live-state gate 分开。完整 hammer 保留了多组 detail reference miter，peak RSS 10,980,168 KiB，同样不能装成 512 MiB state gate 的通过证据。

## 边界

full-first-row=false，runtime-100x-gate=false，full-population=false，production=false，decoder-complete=false，cycles/speedup-citable=false，paper-citable=false。本 hammer 没有执行 VCS/DC/PT/Formality/PTPX/GPU/remote。`docs/359` SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
