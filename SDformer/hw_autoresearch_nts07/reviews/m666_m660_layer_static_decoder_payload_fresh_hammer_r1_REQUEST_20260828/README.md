# M666 fresh independent static hammer request

This request freezes the M660 author candidate for an independent, adversarial
static review.  It does not authorize the GPU command, consume the M660
one-shot, run a simulator/RTL/EDA flow, or modify any author/predecessor file.

## Frozen target

| target | SHA256 |
|---|---|
| producer | `2e1ea26b5293ba1063e7be0056cebd2b25e09903bb528c31427c032df8b73acc` |
| runner | `ae9902b42331f3e88e94b11d9c5a5f6f3bdfc3e2b473939a7569af38f2396281` |
| contract | `38200ef4db5795d8be70e6e776aabf09dad10818344b972add535900a95f2cb4` |
| author tests | `0dc63c88349dec0ecc77d2fb4aa51f0df82316d1c435a73f1d760ae50fb54cc0` |
| author handoff outer-seal file | `341db83d1c084b3ea6e41b155d4a24039b858fafa9a23ca45e7a3319f105f414` |

## Required attacks

- independently rehash the target and all contract roots; verify M658/M659/M662
  double seals and typed claim boundaries;
- reject symlink and parent-traversal aliases before one-shot consumption;
- prove `take_exact(...,10)` never asks for item 11;
- force non-`{0,theta}` values in arbitrary D1 chunks/samples and check that no
  D1 bitpack/folded weight/output-scale candidate can survive the negative
  route;
- reject scalar theta that is NaN, infinity, zero or negative and detect any
  runtime threshold drift, not only endpoint equality;
- independently check little-bit-first order, tail alignment, byte population
  and popcount;
- ensure a folded miter with unequal output bytes cannot admit deployment,
  including `+0.0` versus `-0.0`;
- keep output-scale sidecar unadmitted; never serialize raw D1 values;
- verify the 40-hook/30-or-40-payload lattice, fresh staging, atomic publication,
  post-publication quarantine, sanitized environment and nested double seals.

The review may issue `GO` only with P0=0 and P1=0.  Any P0/P1 finding requires
`NO_GO` and no candidate command.

