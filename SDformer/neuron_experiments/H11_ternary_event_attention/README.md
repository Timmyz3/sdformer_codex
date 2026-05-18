# H11 Ternary Event Attention

H11 keeps the H9a training scaffold and replacement scope, but changes the
extra attention gate from continuous `sum(Q * K)` to sign-only ternary event
compatibility:

```text
carrier = K * sn2_q(sum_channel(Q))
score   = pp(sign(Q), sign(K)) + alpha * nn(sign(Q), sign(K)) - beta * mismatch(sign(Q), sign(K))
gate    = Shiftmax(score)
attn    = carrier * gate
```

The default config uses `alpha = 0.25` and `beta = 1.0`. This makes
negative-negative agreement weaker than positive-positive agreement, and keeps
learned ATLIF threshold magnitudes out of the attention score calculation.

The baseline SDFormerFlow folder is not modified. H11 code lives under this
experiment folder.
