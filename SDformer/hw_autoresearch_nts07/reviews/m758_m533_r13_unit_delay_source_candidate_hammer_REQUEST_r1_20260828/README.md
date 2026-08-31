# M758 / M533 additive r13 source-candidate handoff

M757 proved that frozen r12 failed closed before preflight/result/VCS because its line-314 M743 manifest digest was 63 hex characters.  This additive r13 source fixes only that executable defect by inserting `b` at position 40, producing the live 64-hex digest `626ba66587e86885020031ef5656c3cd971cdacb803bc339b218d1171d796962`.

Top r2, TB r7 (`d194f912...`), SVA r2, the 9x128 1RW adapter/binding, foundry `+define+UNIT_DELAY`, failure gates, watchdogs, resource gates and same-UID collision gates are frozen.  M757 is a byte-exact, double-sealed prerequisite.  M749 and the M753 final-release hammer retain their other checks, but M757 records their P1 omission: neither checked every embedded `require_regular_sha` literal against its referenced live file.

The fresh source hammer must therefore enumerate all 52 hardcoded `require_regular_sha` calls, reject any expected digest that is not exactly 64 lowercase hexadecimal characters, resolve every referenced regular file, recompute all 52 live SHA-256 digests and require exact equality.  Endpoint seal checks or whole-runner identity alone are insufficient.

No runner, VCS, simv, HDL compiler, CPU/GPU experiment or EDA execution is authorized.  The candidate has `launch_now=false`; a later candidate hammer, separate true release and fresh final-release hammer remain mandatory before at most one r13 VCS attempt.
