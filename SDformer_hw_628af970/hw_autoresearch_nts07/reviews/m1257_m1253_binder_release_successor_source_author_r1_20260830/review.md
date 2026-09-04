# M1257 binder release successor source author receipt

Status: **SOURCE ONLY; fresh different-author hammer required; production not authorized.**

M1257 closes the four M1255 findings without changing the exact M1241/M1234/M1228 execution bytes or the one-shot mechanics:

- full `st_mode` now travels from every pre-attempt snapshot through the sealed child receipt and exact manifest/config/checkpoint/profile comparisons, including candidate rows, selected projection and sidecar;
- the selection root, all artifact/profile/activity maps and relevant scalar types are closed schemas, so unknown claims are rejected even when their value is `false`;
- one source-generated ordered nine-row E0-E8 policy is independently required from both the result and sidecar;
- three fully sealed memfds, exact `pass_fds`, candidate/epoch order, minimum finite nonnegative AEE, lowest-epoch tie break, O_EXCL consume-before-child and no retry are retained.

The temporary-fixture suite passes 14/14, including post-prepare chmod, missing-mode, nested extra false claim, root false/positive claims, joint E0-E8 splice, profile/activity key and bool-type attacks, wrong candidate pair, nonminimum selection, wrong tie break, selected sidecar splice, sealed launcher mode publication and O_EXCL/no retry.

No production binder was executed. No real checkpoint/profile was opened by the test suite; no remote host, GPU, valid825, VCS or Synopsys tool was used. M1257 may only advance after a fresh different-author source hammer. Even a hammer PASS would authorize one future execution only after all four real strict-valid825 artifacts exist and output/attempt/log namespaces are fresh.

`docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
