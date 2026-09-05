# M2234 independent source review of M2233

Date: 2026-09-05. Reviewer: independent `/root/m2224_lm_discovery_review` agent, not the M2233 author.

**PASS 98/100; P0/P1/P2 = 0/0/0.** M2233 closes the complete project-local parser import chain and binds it before the attempt or any tool operation. This review releases exactly one new M2235 campaign. M2236 result review remains required before any paper claim.

The production parser imports M2172 and M2117; M2172 imports M2160. Independent runtime import tracing observes exactly these four files. Every one is in the 29-member exact source inventory. M2160 and M2117 contain only standard-library imports and no further dynamic local import. Three explicit helper SHA checks precede DC-launcher validation, contract processing, and future parser execution. M2234's release identity and M2235's emitted result both bind all three helpers.

The ten author CPU tests pass independently. A separate mutation test makes each helper appear to have drifted and installs a failing sentinel at the later tool gate; all three reject at the helper identity check before that sentinel. The tests also reject wrong review helper bindings and score 94. Contract/author/methodology seals are valid, and the new M2235 result, consumed-attempt, lock, work, stage, and failure namespaces are empty at review time.

The frozen three windows remain slots 1606/526/1071, with saved-read fractions 0/1, 9/25, and 117/163. They were selected before power measurement and are unchanged. The same-workload comparison requires both ordinary and TSBG at each window, yielding six points. The one-third aggregate is correctly renamed a **fixed three-window weighted index**, not the population mean of 2,880 workloads or frame energy. The tie-break toward higher ordinary request count is disclosed.

Both axes use the same 288 KiB, 16-macro external SRAM capacity, modeled area, and leakage power. The verified model uses 22.213 pJ per actual accepted bank activation and 3.826774326764422 mW leakage per axis. Mapping uses SSG 0.9 V/125 C maximum and FFG 1.05 V/-40 C minimum libraries; PTPX uses TT 0.9 V/25 C on the mapped standard-cell netlist. The external SRAM dynamic and leakage models use their separately identified corners. The final result explicitly preserves the mixed-corner component-model label and logic/SRAM-dynamic/SRAM-leakage split.

The fixed serial budget is one license query, two VCS compiles, six simulations, six diagnostic plus six measurement SAIF files, two fresh DC maps, and six PTPX runs. There is no automatic retry or old-SAIF reuse. All six points and exact final operation counts are required. Existing source/results and `docs/359` remain unchanged.

This source release does not promise a successful run or an energy reduction. Native RTL SAIF mapped through DC transformations is the activity source; SRAM is modeled separately. Hold closure, post-layout results, silicon measurements, full-network performance, and energy/frame remain outside this campaign. These are explicit measurement boundaries, not unresolved defects in this narrowly scoped dependency repair.

The root agent may invoke the single M2235 campaign after this exhaustive double-sealed review. This reviewer performed CPU-only checks and no EDA, license, GPU, or Git operation.
