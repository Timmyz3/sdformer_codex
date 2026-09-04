# M746 / M533 r12 source-candidate handoff

This package requests a fresh, read-only static hammer of the unique r12 source candidate. The only functional-source delta from consumed M737/r11 is the M744-admitted TB r7 causal RAW monitor. Top r2, SVA r2, the 9×128 1RW adapter/binding and the checksum-identical foundry model remain frozen; compilation remains exactly `+define+UNIT_DELAY` and functional-only.

The evidence-correct TB r7 SHA is `d194f91293cf7e533e099d8b36956fb00db16402340c8e6e678059cb9adb0fd2`. SHA `10fb3f30...` belongs to TB r6 and must not be relabeled.

No runner/VCS/simv/EDA execution is authorized. The candidate has `launch_now=false`; after a 100/100 source/candidate hammer it still needs a separate true release and a fresh final-release hammer before one attempt.
