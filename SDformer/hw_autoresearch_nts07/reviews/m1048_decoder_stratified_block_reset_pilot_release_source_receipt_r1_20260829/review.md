# M1048 decoder stratified block-reset pilot release source

Status: **source-only PASS; M1049 independent hammer required.** No real payload window was executed and no M1050 attempt was consumed.

The release targets the M705-admitted M699 `interlaken_01_a`, sample 0, timestep 0 exact-binary payloads for D0/D2/D3. D1 remains diagnostic-only and has no generator or scheduler route.

The pilot partitions the frozen A1 stream into source census, dependency, compute and commit phase blocks. Every generated compressed transaction must be assigned exactly once. Selection is an online top-k implementation of the frozen M1041 hash order, and the 85-transaction synthetic test matches M1041 exactly. A real run would freeze one source census and eight blocks per noncensus stratum before any scheduler call, then emit all raw cycles, exact-miter identities, coverage rows and CI inputs.

Candidate and baseline intentionally replay the same A1 body. This is a protocol-calibration pilot, not a performance comparison. Its cycles cannot be called local speedup, continuous-row cycles, decoder-complete cycles, a Table-A row, or system speedup.

The one-shot runner is inert without four caller pins. It validates M1048 and independently sealed M1049 authority, obtains a fixed nonblocking lock, rejects same-UID EDA collisions, checks memory/commit headroom and fresh namespaces, and consumes the permanent attempt before work creation or payload access. Runtime execution/assemble/publish modes revalidate M1049. Failures after attempt consumption move work to a unique quarantine; only a sealed work directory is atomically published.
