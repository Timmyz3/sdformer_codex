# M1287 independent source hammer of M1279

**STOP, 74/100; P0/P1/P2 = 0/4/0. No fresh RTL VCS release is authorized.**

The current RTL observation sources themselves clear several important gates. All seven tapped clones are functionally identical to their frozen parents after removing only read-only tap additions. The top exposes exactly thirteen `keep` taps; all current fanouts are exact and none feeds functionality. The valid-qualified endpoint covers all seven request-payload fields and its 24-case four-state projection quarantines unknown valid, payload, or accept while raising `endpoint_protocol_fault_now`. The eleven-member filelist contains the exact twelve expected module definitions. No VCS or EDA tool was run.

The release blocker is the executable meaning of the TB. Its 128-cycle PASS requires only that both DUTs accepted raw input. It never requires a bank request/accept, result, or token completion, and it does not compare functional outputs. Thus the endpoint can be unreachable and the fixed window can still print PASS.

Three fail-closed source-checker boundaries also remain open. The contract schema accepts added `K8`, equal-bandwidth `K1x8`, single-K1 power, fair-energy, performance, system-speedup and paper-headline claims. Both the valid gate and payload-known gate can be replaced by an unconditional branch while retaining the expected lexical token in a comment. Finally, replacing an exact semantic-tap fanout with an X-to-zero coercion passes the target normalizer because every tap assignment line is removed before comparison.

The additive repair is small but mandatory: require request reachability, qualified result/token completion and class-aware dual-DUT comparison; close contract key sets; structurally validate the endpoint block; and bind each tap to its exact RHS while forbidding X coercion. A new different-author source hammer must pass before the root author may sign one fresh RTL-only VCS release.
