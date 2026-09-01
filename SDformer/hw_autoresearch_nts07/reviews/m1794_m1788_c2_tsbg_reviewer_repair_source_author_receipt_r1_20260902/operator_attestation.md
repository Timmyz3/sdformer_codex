# M1794 source-author attestation

M1794 is an additive successor to M1787. M1787 and the sealed M1788 review were not modified. I did not run VCS, `simv`, Design Compiler, PrimeTime PX, a license query, an attempt, or a result.

The production Acc24 proof is now fixed at `48 * 16 * 128 = 98,304`. The elaborated source supports `SOURCE_GROUPS` from 1 through 48. The directed tuple is twelve groups, whose bound is `12 * 16 * 128 = 24,576`; therefore `24,576 <= 98,304 < 2^23`, and both directed DUT tuples satisfy the actual RTL predicate. The checker parses the TB tuple and predicate rather than merely searching for a constant. Three independent mutations recreate a time-zero fatal and are detected.

The original fabricated stale attack remains. Separately, the TB saves the exact epoch, slot, generation, tag and payload of an accepted bank-3 response. After the complete bundle retires, the same bank replays that exact legal identity. The replay must have zero accepts and cause sticky protocol/stale state.

After a separate bogus stale attack, the TB holds reset for three clocks and then runs the minimum complete legal B8 service: one live group for each token identity, real bank requests and reordered responses, typed signed bridge updates, 48 Acc24 commits and eight terminals. The SVA recovery cover accepts one through eight reset clocks and observes a later legal terminal.

All 1,152/144 aggregate bundle-beat and 9,216/1,152 scalar-bank counts remain directed expectations, not VCS measurements or hardware speedups. A different author must complete M1795 before any launch release.
