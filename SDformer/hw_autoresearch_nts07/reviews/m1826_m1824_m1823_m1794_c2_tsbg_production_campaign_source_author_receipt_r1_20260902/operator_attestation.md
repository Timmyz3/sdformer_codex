# M1826 source-author attestation

M1826 is an additive governance successor to immutable M1823/M1824. I did not
modify M1794, M1795, M1812, M1813, M1823, M1824, docs/359, or any prior sealed
artifact. I did not query a license, launch VCS/simv/DC/PTPX/Formality, create an
attempt or result, or create M1828.

The checker now validates exact AST arguments and direct reachability rather
than call names alone. The shared flock must use `queue_handle` and `LOCK_EX`;
the local flock must use `lock_handle` and `LOCK_EX|LOCK_NB`. The seven-entry
freshness tuple, collision command set and direct membership guard, all four
resource thresholds, source authority target/pin pairs, immediate attempt-state
transition, failure-quarantine guard, and private-then-canonical atomic publish
order are exact predicates.

Future M1828 identity must have the exact complete key set and expressions for
M1827, M1823/M1824, M1812 author evidence, M1813 review evidence, M1794/M1795,
and docs359. Extra, renamed, omitted, or misbound keys fail.

The original 58 mutation targets remain. Twelve M1824 equivalent bypasses and
three guard-reachability self-checks bring the suite to 73. Python 3.6 and 3.10
each reject 73/73. A separate eight-attack probe also rejects 8/8 unseen variants.
This source remains inert until a different-author M1827 zero-P0/P1 review and a
future exact double-sealed M1828 release exist.
