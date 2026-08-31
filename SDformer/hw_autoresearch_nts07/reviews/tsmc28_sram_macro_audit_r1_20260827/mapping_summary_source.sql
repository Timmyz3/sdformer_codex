-- Frozen report dataset constructed from the audited mapping ledger.
-- NULL means no admitted foundry area/timing model, not zero cost.
SELECT 'C1 center PWP' AS component, '128x768 + 128x512' AS logical,
       '2 independent 1RW SP' AS port, '10x generated 128x128 SP' AS mapping,
       10 AS macro_count, 87583.606 AS area_um2, 0.616 AS cycle_ns,
       'GENERATED_VIEW' AS evidence, 'macro subtotal only' AS status
UNION ALL SELECT 'C1 correction','128x768','phase-separated 1RW SP','6x generated 128x128 SP',6,52550.164,0.616,'GENERATED_VIEW','ping-pong OPEN'
UNION ALL SELECT 'C1 descriptor','32x48','phase-separated 1RW SP','1x SP 32x48m4s',1,2937.195,0.479,'FOUNDRY_QRT','views OPEN'
UNION ALL SELECT 'M498 parent scratch preferred','64x1152','1R1W','16x DP 64x72m4f',16,NULL,NULL,'COMPILER_LEGAL','PPA OPEN'
UNION ALL SELECT 'M498 parent scratch fallback','64x1152','1R1W','32x DP 64x36m8f',32,473034.720,0.853,'FOUNDRY_QRT','model only'
UNION ALL SELECT 'M498 dual response slots','2x1152','elastic same-cycle','stdcell registers',NULL,NULL,NULL,'STDCELL','retain regs'
UNION ALL SELECT 'M498 one psum bank','64x1824','1RW if R/W exclusive','19x SP 64x96m4s',19,113087.468,0.500,'FOUNDRY_QRT','port OPEN'
UNION ALL SELECT 'M467 eight psum banks','8x64x1824','1RW if R/W exclusive','152x SP 64x96m4s',152,904699.744,0.500,'FOUNDRY_QRT','bank sum only'
UNION ALL SELECT 'C2 FC2 weights','8x2304x128','8 independent 1R SP','8x(2048x128 + 256x128)',16,558507.032,0.800,'FOUNDRY_QRT','views OPEN'
UNION ALL SELECT 'C2 K1/K8 context','48x384','same-edge update/result','candidate 6x DP 48x64m4f',6,NULL,NULL,'COMPILER_LEGAL','adapter OPEN'
UNION ALL SELECT 'C3 ATLIF working state','depth 1/2/16 mixed','multi-semantics','stdcell registers',NULL,NULL,NULL,'STDCELL','retain regs'
UNION ALL SELECT 'A1 row + descriptor','4x225x32 + 2x225x20','6 phase-separated 1RW','6x SP 256x32m4s',6,21188.772,0.477,'FOUNDRY_QRT','views OPEN'
UNION ALL SELECT 'A1 slot + score directory','32x16 2enq/2deq + 163x10','multi-access/scan','stdcell registers',NULL,NULL,NULL,'STDCELL','retain regs';
