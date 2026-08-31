# M1182 / M1181 / M1168R3 C1 VCS source hammer

Verdict: **GO, 99/100, P0=0, P1=0.** This is source-only authority for a separate inert M1183 release; it is not VCS evidence.

The sealed R2 evidence proves compile/elaborate/link success followed by simulation failure. The two observed failures occur in directed negative testing: a request-hold assertion sees an intentional upstream attack, and a service-mutation checker is sampled on the ambiguous R2 boundary. The evidence does not prove a normal-path RTL defect, and R3 does not relax a normal legal path.

R3 uses only two request-attack windows and exposes that mask only to the two request-hold properties. Each service attack has its own one-property mask; the peer service property and the other protocol assertions remain active. The sticky service checker is independent of `protocol_error`, detects on the positive edge, and is sampled on the following negative edge after NBA and UNIT_DELAY activity. Both service tests additionally require that the DUT protocol-error path remain clear, while all seven DUT attacks retain sticky protocol-fault checks.

The package preserves 16 assertions, six covers, the executable II=2 check, 24 deterministic legal random transactions, and one frozen-M935 two-beat/one-row/one-task completion. Four directed legal cases, 24 random cases, and the M935 case explicitly check all three attack masks low, for 29 legal-mask checks.

Independent hammering passed 4,300 source/seal/forensic checks and rejected 37 mutations spanning mask misuse, permanent disable, service-checker coupling, sampling race, scoreboard relaxation, assertion/cover removal, II/M935 weakening, claim promotion, old-namespace reuse, and runner/release cardinality weakening. The consumed R2 namespace remains quarantined; every R3 attempt/result/work/quarantine namespace is fresh.

No runner, VCS, simv, EDA executable, or license client ran. A separately authored and sealed M1183 release is still mandatory before the single allowed functional compile and simv attempt.
