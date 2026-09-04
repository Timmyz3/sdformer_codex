-- Frozen visualization dataset. These scopes include alternatives and must not be summed.
SELECT 'A1 six SRAM banks' AS component, 21188.772 AS area_um2, 'FOUNDRY_QRT' AS evidence, 'QRT proxy; generated views missing' AS scope_note
UNION ALL SELECT 'C1 center + correction',140133.770,'GENERATED_VIEW','16 generated SP macros; descriptor excluded'
UNION ALL SELECT 'M498 one psum bank',113087.468,'FOUNDRY_QRT','conditional on R/W-exclusive SP protocol'
UNION ALL SELECT 'M498 parent scratch fallback',473034.720,'FOUNDRY_QRT','exact-capacity fallback; preferred DP view still open'
UNION ALL SELECT 'C2 FC2 weight capacity',558507.032,'FOUNDRY_QRT','same 288 KiB deployment for K1/K8/K1x8'
UNION ALL SELECT 'M467 full eight psum banks',904699.744,'FOUNDRY_QRT','bank sum; not floorplanned integrated area';
