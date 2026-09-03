# M1973 — C2 K8 hold-open first-principles review

Verdict: **the M1877 hold failure is a design-wide enabled-state self-feedback problem under a prelayout ideal-clock model, not a small set of exceptional control paths. Do not run another buffer-only sweep.**

## Bound evidence and scope

This is an additive, read-only diagnosis. It binds the sealed M1877 PrimeTime result, the sealed failed M1960 exact-50-ps repair, the independent M1969 failure review, the frozen K8 mapped netlist, and the RTL that owns the dominant state arrays. It launched no EDA or license query and changed no RTL, netlist, result, predecessor review, or `docs/359`.

All three predecessor inner/outer seals verified. `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.

## 1. What the 30,442 hold violations are

M1877 reports 30,442 violated register hold checks out of 32,429 tested checks: **93.872768%**. Parsing every path in the full 1.44-million-line `report_constraint -all_violators -verbose` report gives:

| Classification | Paths | Share of 30,442 |
|---|---:|---:|
| Exact same-register Q-to-D feedback | 30,267 | **99.425136%** |
| Same named state family | 30,427 | **99.950726%** |
| Cross-family | 15 | 0.049274% |

The dominant endpoint families are:

| State family | Violations | Share |
|---|---:|---:|
| `service_ctx_q` (Acc24 contexts) | 18,432 | **60.547927%** |
| `slot_weight_q` (8-bit response slots) | 8,192 | **26.910190%** |
| `paired_sink_bitmap_q` | 1,534 | **5.039091%** |
| `compactor_queue_bitmap_q` | 444 | **1.458511%** |
| Four families combined | 28,602 | **93.955719%** |

This is independently consistent with the mapped-netlist topology. Of 32,427 mapped flops, 31,280 (**96.462824%**) have an explicit feedback route through at most two combinational cells. The source RTL confirms word-granular conditional updates: a context update writes 16 Acc24 lanes, while a returned bank writes 16 signed 8-bit weights. Synthesis implements the inactive case as Q-to-D feedback.

The violation magnitudes are shallow and numerous, not a few deep outliers: 61.632% have slack in [−20,−10) ps, 34.321% in [−10,−5) ps, and only four paths are worse than −20 ps. The worst path is −23.259 ps. This distribution explains why scalar buffer tuning pays a broad area tax.

## 2. What the clock model means

The SDC hold uncertainty is **50 ps**, not 60 ps. The approximately 60–61 ps required time visible on the worst paths is 50 ps uncertainty plus roughly 10–11 ps library hold time. Public wording must not call it a 60-ps uncertainty.

M1877 is a legitimate diagnostic fast-min result at `ffg1p05vm40c`, but it is not layout closure:

- `core_clk` is ideal and unpropagated, with zero insertion delay and zero modeled skew;
- both min and max data interconnect use `ZeroWireload`;
- no SPEF or extracted parasitics are present;
- OCV min/max libraries are used, but placement, CTS, routing, useful skew, and post-route hold repair are absent.

Therefore the honest paper boundary is: **C2 is setup-met at 3 ns and formally equivalent, but fast-min hold remains open in a prelayout diagnostic.** The 30,442 count must not be presented as post-route silicon risk, and the ideal-clock result must not be presented as physically closed. Real data routing often adds min delay, while CTS skew can help or hurt; only a propagated-clock extracted run can adjudicate the final sign.

## 3. Can a local structural fix stay below 5% area?

No existing evidence admits such a fix.

- A targeted exception or small control-cone edit cannot address a population in which 99.425% of failures are exact state self-feedback and 93.956% sit in four large arrays.
- Rewriting an RTL enable mux without changing the storage/clock realization is semantically identical; synthesis recreates the Q-to-D hold path.
- M1960's one exact-50-ps `set_fix_hold` attempt nominally drove the optimizer's estimated min-delay cost to zero, but increased mapped leaf area from 130,822.775176 to 141,886.71 µm²: **+8.457193%**, outside the frozen +5% gate. It produced no admitted post-fix timing report or netlist.
- Dedicated delay cells might be more area-efficient than generic synthesis repair, but no bound library cell, placement, or extracted result demonstrates a sub-5% solution. It is speculation and is not a GO basis.

The only plausible structural alternative is word/bank-granular clock-enabled storage or SRAM/register-file mapping for the Acc24 contexts and weight slots. That can preserve protocol and throughput because the RTL already updates 384-bit context words and 128-bit slot/bank words atomically. However, it changes clock/storage implementation, requires generated-clock/gating checks or macro timing, and still must be judged after CTS. It is not a safe local RTL patch and has no proven <5% area number today.

## 4. Unique next GO/NO-GO gate

Authorize **one fresh matched post-CTS/post-route physical implementation**, not another DC buffer/uncertainty sweep:

1. run both K8 and equal-bandwidth K1×8 through the same floorplan, placement, CTS, routing, RC extraction, min/max libraries, uncertainty, IO constraints, and DRC rules;
2. preserve RTL protocol and the frozen directed-cycle result (K8 1,913 cycles, K1×8 1,945 cycles); add no false/multicycle exceptions and no throughput-changing pipeline;
3. allow normal post-CTS targeted hold repair/useful skew; optionally infer word-granular gated storage only if VCS and Formality remain exact;
4. **GO** only if both axes have setup WNS ≥ 0, hold WNS ≥ 0, zero DRC, K8 repair area growth ≤ 5% versus its own pre-repair placed baseline, and K8 retains at least 4.0× throughput/mm² versus physical K1×8;
5. otherwise **NO-GO**: freeze C2 as a setup-met/formally-equivalent logic-only area-efficiency component with an explicit hold-open limitation, and stop hold-closure iteration for this submission.

This gate attacks the missing physical information exactly once. It prevents a third blind buffer sweep and prevents an unbounded RTL redesign from consuming the paper window.
