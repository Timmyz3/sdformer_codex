# M1582 minted-instance registry successor — author review

Status: **PASS author dual-runtime source regression; independent rehammer still required.**

M1576 correctly showed that constructor-only secrets and exact type checks do not establish provenance in Python: `object.__new__` can allocate the exact permit class, and caller code can populate every name-mangled slot. M1582 moves authority out of writable instance state. The permit closure now owns two distinct registries, one for production and one for synthetic permits. Each issuer registers the exact returned instance together with its resolved output, inventory SHA, estimate, and free-byte observation.

`consume()` uses `dict.pop(self)` as the single membership-check-and-removal operation. A forged exact object is absent and is rejected. A second consume is also absent and is rejected. The popped closure record, not caller-writable slots, supplies the receipt. Consequently, rewriting a legitimate permit's private-looking output and free-byte slots did not alter its bound output or the recorded real disk value.

The same 33-attack source suite passed under CPython 3.10.18 and 3.6.8. In both runs, `object.__new__` plus complete slot population was rejected for production and synthetic types; duplicate consumption was rejected for both. The production issuer retained an output-only signature and called the real `shutil.disk_usage`; the observed and receipt values were both 101631811584 bytes. The tiny six-frame synthetic roundtrip remained valid.

This author package executed no remote wrapper, SSH, checkpoint load, GPU work, capture, release, production payload, RTL, or EDA. It creates no accuracy, cycle, traffic, energy, speedup, or paper claim. A fresh different-author dual-runtime rehammer is still mandatory before any remote-wrapper work.
