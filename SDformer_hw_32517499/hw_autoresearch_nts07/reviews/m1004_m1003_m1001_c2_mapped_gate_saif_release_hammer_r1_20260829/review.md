# M1004 independent M1003 release hammer

Verdict: **STOP (82/100, P0/P1/P2 = 1/0/0).** M1005 and all EDA remained unexecuted.

The M1003 JSON and double sidecar are internally sealed; the M1002 outer seal and M1005 runner pin are correct. Its scope is also properly limited to one 15-simulation VCS+SAIF run, with PT/PTPX/DC disabled.

However, M1003 records source-contract SHA `7afc4c095d7a...`, while the actual frozen M1001 contract SHA is `7afc4c093b80...`. The production runner compares this field to the actual contract before EDA, so the current release cannot legally launch and does not bind the reviewed source.

M1005 is not authorized. A corrected additive release and a new independent release hammer are required.
