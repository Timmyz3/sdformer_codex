# M1527 independent actual-result hammer

Status: `PASS_M1527_M1521_EP34_DECODER_POSITIVE_PLANE_ACTUAL_RESULT__ADDRESS_REPLAY_SUCCESSOR_ALLOWED`.

The frozen public `M1521.verify_materialized_seal` was rerun against the canonical result. It regenerated the expected semantics from M1458 through M1510/M1516 and passed a type-strict full-tree comparison: 122 sealed members, 120 canonical output paths, and `full_tree_equal=true`.

The result has exactly 120 positive-plane payloads in `c000_s10_d0` through `c119_s39_d3` order, zero negative-plane files, an exact unique consumed-attempt marker, and the frozen no-performance claim boundary. The result is therefore safe input to a separately reviewed address-timed replay successor. No cycle, traffic, speedup, energy, RTL, PPA, or Table-A claim is admitted by M1527.
