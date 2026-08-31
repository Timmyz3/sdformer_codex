# M1307 independent receipt-blind hammer — M1306

Verdict: **100/100, P0=0, P1=0, PASS.**

M1306 closes the sole M1303 finding without changing the frozen execution
mechanism. The independently observed order is:

1. M1306 pins the exact M1301 source/test/contract and M1303 STOP seal.
2. Frozen M1301 verifies M1297 and M1298 authority.
3. Frozen M1301 validates the exact seven-key, exact-`false` claim map.
4. Frozen `M1297.M.verify_frozen_authorities()` verifies M1257.
5. Frozen `M1297.execute_once()` is delegated exactly once.

The blocking replay now behaves correctly: an injected inherited-authority
failure is called exactly once, the delegate is called zero times, and no
attempt exists. Failures at every earlier stage likewise prevent every later
gate and the delegate.

The policy, candidate selection/F1–F4 behavior, E0–E8 rebind map, interpreter
entity, retained FD, `/proc/self/fd` child, three sealed execution sources plus
interpreter `pass_fds`, 11 snapshots, attempt entity digest, `O_EXCL`, and
no-retry behavior remain inherited exactly. Claim, predecessor SHA, seal,
interpreter-entity, path-swap, and attempt-reuse attacks fail closed. A failed
child runs once and cannot be automatically retried.

Authorization: exact reviewed-byte transfer is GO; root-controlled remote live
read-only preflight is GO; exactly one remote production execution is GO only
after the transferred bytes, all sealed dependencies, interpreter entity and
capability fields, candidate/profile population, working directory, and fresh
absence of attempt/result/log match their pins. There is no automatic retry.
The production result still requires a fresh independent result hammer before
checkpoint selection or any hardware rebind claim is admitted.

The seven source claim values remain false because this hammer did not execute
production and did not select a checkpoint. The operational one-shot GO above
is conditional authority for the root-controlled next action, not evidence that
execution or selection has already happened.

The M1306 author receipt was not opened or trusted. No remote, production,
checkpoint, GPU, VCS, DC, PT, or other EDA action was performed. `docs/359`
remains unchanged at
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
