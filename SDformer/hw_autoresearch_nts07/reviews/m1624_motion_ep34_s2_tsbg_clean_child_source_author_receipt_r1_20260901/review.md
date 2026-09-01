# M1624 clean-child reduced-binary successor — author review

Status: **PASS source-only author review; fresh different-author hammer remains mandatory.**

M1598 correctly rejected the use of a Python closure or registry as a security boundary. M1624 removes that boundary entirely: the parent accepts no provider, permit, free-space value, provenance value, registry, callback, or callable. It executes one fixed Python interpreter with isolated mode and one exact source path. The child independently rechecks fixed on-disk identities and fresh namespaces, atomically consumes the sole attempt before model load, and only then constructs and consumes the fixed M1558 producer permit inside the same child.

The compile-free attack suite passed 15/15 under CPython 3.6 and 3.10. It rejects capability arguments, reflection seams, reordered attempt consumption, duplicate child invocation, namespace reuse, and false claim mutations. The source self-check was instrumented to prove that it opens no M1458 payload member; it only checks the small top-level metadata and frozen seals needed for source authoring.

This package does not authorize or execute a checkpoint load, GPU run, capture, retry, AEE measurement, TSBG DSE, RTL, EDA, or paper claim. Only a fresh different-author source hammer (M1625) is authorized. Any capture additionally requires a separate exact release (M1626).

