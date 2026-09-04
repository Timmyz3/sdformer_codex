# M1181 / M1168R3 source author receipt

R2 compiled, elaborated, and linked successfully, then failed in directed negative tests: `ap_psum_request_hold` fired during the attack phase and the service-mutation monitor was sampled on an ambiguous boundary. The sealed logs do not prove a normal-path RTL fault.

R3 is additive and does not change M1162 or M935. It preserves all 16 assertions and six covers. Only the two explicit upstream cancel/tuple-mutation windows mask the two request-hold assertions. Weight and psum service attacks use separate one-property masks, leaving the peer property and the other assertions active. An independent per-service sticky checker is sampled on the following negedge, after NBA updates. Twenty-nine legal directed/random/frozen-M935 cases explicitly require every mask low.

Static checking passed 633 checks and rejected 12 mutations. The R2 attempt remains consumed and non-reusable; the R3 attempt/result/work/quarantine namespaces are fresh. This package is source-only: fresh M1182 hammer and separate M1183 release are mandatory before any VCS execution.
