# M1685 independent hammer of M1684 C2 production-energy source

Status: `FAIL_M1685_M1684_C2_M1609_FRESH_MAPPED_PRODUCTION_ENERGY_SOURCE_HAMMER__NO_M1686_RELEASE__SHARED_EDA_QUEUE_REPAIR_REQUIRED`

Score: **94/100**; P0/P1/P2 = **0/1/1**. This was a source-only review. No license query, VCS, simulation, SAIF generation, PrimeTime PX, attempt, result, or release was run or created.

## What closes cleanly

The source binds the fresh M1661/M1677 K8 and equal-bandwidth K1x8 mapped netlists and SDCs by exact SHA and excludes the old M872 paths. It carries the M1609/M1627 registered sticky-fault semantics into the production monitor. Both axes use the same five directed cases, 3 ns clock, direct-DUT accepted-header-to-post-token-done windows, and 261 accepted sources per axis.

The planned geometry is exactly two VCS compiles, ten mapped simulations, ten direct-DUT SAIF files, and ten PTPX runs. All ten binary/X-clean functional and SAIF checks precede the first PTPX launch. Case 4 contributes its 14-cycle zero-event leakage/clock energy before the 261-source denominator is applied. The mW, pJ/cycle, pJ/accepted-source, K1x8/K8 energy ratio, saving fraction, and throughput/W algebra is correct. Twenty-six mutations were rejected under CPython 3.6.8 and 3.10.16.

## Release-blocking P1

The source does not prove the requested unique EDA queue. `/tmp/m1684_c2_mapped_production_energy.lock` is private to M1684, not shared across campaigns. Its two same-UID collision scans occur before license preflight and attempt consumption; there is no new collision scan before either VCS compile or any PTPX launch. Another campaign can therefore start in the gap and overlap M1684.

The successor must acquire the agreed cross-campaign same-UID EDA lock, rescan after holding it, and perform ancestry-aware collision checks immediately before every VCS and PTPX launch. Attempt consumption must remain before the first EDA process and automatic retry must remain disabled.

## P2 hardening

The exact sources are clean, but the checker bans `initreg` across all execution sources while its explicit active-`force` test covers only the additive assertion module. The successor should add a comment-stripped active-force scan over runner, Tcl/UCLI, wrapper, assertion, and inherited case-TB sources.

M1686 is not authorized. Only a successor queue-repair source is authorized.
