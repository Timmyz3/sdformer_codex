# M292: M287 scope-corrected Amdahl overlay

M287's raw weight/payload DSE is reproducible, but it covered ten binary FC1
modules and projected their task ratio onto all twelve FC1 modules.  The frozen
ledger assigns 100,895,624 cycles to the eligible ten modules and 17,474,490
cycles to the excluded stage-3 pair.

After correcting the denominator, group-4 beta-80 falls from the claimed
crossing to 1.145605x and is rejected.  Group-4 beta-96 crosses 1.15 only by
removing 89.7871% of weighted tasks and is also rejected as a primary point.
Group-8 beta-96 barely reaches 1.151312x and is retained for one paired S10
accuracy kill test only.  If absolute AEE increases by more than 0.02, the axis
stops before valid825 or RTL.

These are ideal task-compaction sensitivities, not executable cycles or system
speedup.  The INT8 accumulator bounds are not float-output or AEE bounds.
