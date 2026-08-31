# M209 admission revocation

M209's 92,878,814-cycle replay remains a useful ideal, non-truncating control
model on the frozen low-density H67 payloads, but it is not an admitted M207
RTL-semantic result.  An independent legal dense-bank VCS case proved that
M207's five-bit per-packet bank sum truncates 48 to 16 and can deadlock after
bank-count underflow.  M210 r2 fixes the width and adds the missing adversarial
regression.  Consequently the M209 legacy-baseline factor must not be cited as
implemented RTL speedup.
