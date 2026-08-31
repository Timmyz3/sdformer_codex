# M520 H67 paper metric registry static hammer r1

Decision: **STATIC GO for the registry; system table remains blocked.** Score: 98/100, P0=0, P1=2, P2=1.

The implementation satisfies the intended fail-closed contract. It creates the fixed eight baseline rows and all fifteen required columns, producing 120 unique cells. Twenty cells contain scope-limited inventory values and 100 cells remain explicit `null` with a nonempty `blocking_reason`. No cell is system-table eligible and no system speedup is generated.

The strongest property is provenance completeness. Every populated value has an explicit numerator and denominator, workload/checkpoint/sequence identity, operator and resource scope, evidence class, source path plus SHA plus JSON pointer, and a claim boundary. Source SHA drift, pointer drift, row drift, duplicate metrics, non-finite JSON, missing null reasons, and missing numeric provenance are covered by the nine passing tests.

The Prosperity boundary is correctly narrow: the registry records only absolute official product-mode support-tile counters. The external 2.459× ratio is absent, and the row states that it is neither monolithic Conv latency nor same-resource/full-network system evidence. Phi-like is entirely null and blocked. M510's decoder correction prevents both old 620M-class envelopes from being relabeled as complete full-network results.

Remaining P1 gaps are expected paper-closure gaps, not registry defects: there is still no decoder-complete common schedule, matched memory timing/energy, target macro area, multi-sequence population, or full hardware accuracy result. The official artifact paths are host-local and will need path remapping if this evidence package moves to another server. A low-priority test gap remains for a standalone CLI tampered-contract subprocess, although the canonical run did verify builder, config, test, and docs/359 pins.

The existing M520 output was regenerated once after adding generator provenance; only that newly created M520 directory was replaced. Existing evidence and `docs/359` were not modified.
