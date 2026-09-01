# M1823 source-author attestation

M1823 is an additive governance successor to the sealed M1812/M1813 TSBG evidence. I did not modify M1794, M1795, M1812, M1813, docs/359, or any prior artifact. I did not query a license or launch VCS, simv, DC, PrimeTime, Formality, GPU, or remote work.

The M1823 checker parses the concrete runner AST. It requires `verify_authority`, `CHECK.validate_sources`, `namespaces_fresh`, `collision_gate`, and `resource_gate` as direct reachable calls in the ordered main try path; the tool wrapper must independently retain source validation and collision exclusion. It requires `ATTEMPT.mkdir()` to be followed immediately by `state["attempt"] = True`.

The future-release `expected_identity` must bind the exact M1794 source contract, M1795 review, and frozen docs/359 values, together with M1812 and sealed M1813 predecessor evidence. The external self-runner pin must call `exact(RUNNER, authority_pin("M1823_EXPECTED_RUNNER_SHA256"))`.

The original 48 M1812 semantic targets remain present. Ten additive targets cover the nine M1813 escapes plus the self-runner pin. CPython 3.6 and 3.10 each rejected all 58 mutations. The runner remains inert until a different-author M1824 zero-severity source hammer and exact double-sealed M1825 release exist. This receipt authorizes no execution and promotes no result.

