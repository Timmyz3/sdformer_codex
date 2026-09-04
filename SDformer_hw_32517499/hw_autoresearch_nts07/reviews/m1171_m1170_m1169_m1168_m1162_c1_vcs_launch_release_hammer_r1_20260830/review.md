# M1171 independent hammer of the M1170 C1 VCS launch release

Verdict: **GO** for exactly one no-argument M1168 foundry-`UNIT_DELAY`
functional VCS compile and exactly one `simv` run, after root repeats the live
same-UID collision, memory, and namespace checks immediately before launch.

The exact M1170 release SHA is
`a66e4b1f9beb9fcdfb2c1fe8d0b474dc1bf7e9101b1658249a59f35db4d89487`.
It is byte-bound to M1169 review
`8599a332cc0c4e2289969c5eede2fc20850a32ce2541112d2727fbba41eb6fdc`
and outer seal
`cc37cf92b3b30a9c6b13b7625591c262539b2461961f9bdc840660fc1a338121`.

The independent hammer checked 36 exact files and five recursive sealed
directories, performed 1,332 fail-closed checks, and rejected 48 controlled
mutations: 27 release mutations, 19 runner mutations, one duplicate-key JSON,
and one non-finite JSON.  It separately checked the one-compile/one-simulation
cardinality, attempt-before-EDA ordering, all three exact environment values,
UNIT_DELAY-only model selection, same-UID process scan, 64-GiB memory gate,
fresh result/attempt/work/quarantine namespaces, and recursive success/failure
sealing.

No runner, VCS, simv, license query, or EDA tool was invoked.  Therefore this
review authorizes the one functional attempt but does not itself verify RTL,
timing, cycles, performance, PPA, power, energy, or any paper claim.

`docs/359_DATE终局冻结_20260813.md` remains
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
