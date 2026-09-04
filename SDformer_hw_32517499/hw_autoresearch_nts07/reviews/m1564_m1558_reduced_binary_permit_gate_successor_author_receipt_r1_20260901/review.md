# M1564 — reduced-binary permit-gate successor author receipt

The narrow successor removes the module-global raw `_mint_permit` capability
rejected by M1560. Permit construction now remains inside a closure whose only
exposed issuer always performs the fresh-namespace, first-principles result-size,
and strict post-result 16-GiB free-space checks.

Both CPython 3.10 and 3.6 author regressions pass 22 attacks. The new attack
calls the exposed checked authority with zero free bytes and confirms rejection.
The frozen population and format are unchanged: 32 layers, 44,640,000 FC tokens,
430,080,000 PATCH token-equivalents consumed histogram-only, and no authoritative
hardware quantization identity.

This is source-only author evidence. It authorizes an independent rehammer only;
it does not authorize remote wrapper authoring, checkpoint load, GPU/SSH use,
capture, release, retry, RTL/EDA, or any AEE/performance/paper claim.
