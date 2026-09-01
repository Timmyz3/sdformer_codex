# M1702 independent hammer of M1701 C1 tool-entity repair

Status: `PASS_M1702_M1701_M1695_C1_TOOL_ENTITY_REPAIR_SOURCE_HAMMER__AUTHORIZE_ONE_FUTURE_DC_ATTEMPT`

Score: **98/100**; P0/P1/P2 = **0/0/1**. No EDA, attempt, result, or release was created.

The repair admits only the installed official direct link `dc_shell -> snps_shell`. Raw link text, normalized direct target, fully resolved target, non-symlink regular target type, stable target stat, and target SHA are all fixed. Six arbitrary-link, path, type, chained-link, and SHA mutations were rejected.

The M1695 TCL is byte-identical. Frozen M1665 DDC/SDC, 3 ns setup/hold contract, nine macros, one `set_fix_hold`, one hold-only incremental compile, zero timing exceptions, shared EDA queue, resource gates, one attempt, no retry, and positive/negative result gates remain intact.

The only P2 is launch-adjacent hardening: repeat the already-correct tool-entity verification immediately before `dc_shell`. It does not block M1703 because the existing preflight is consistent with the other exact frozen inputs and accepts no arbitrary or chained link.

M1703 release authoring is authorized. M1701 execution is not authorized until that separately sealed release exists and the caller pins both runner and release SHA.
