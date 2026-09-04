# M567 / M560 / M533 r8 true-release author handoff

The exact `launch_now=true` release has been authored and double sealed. It binds the immutable r8 runner, source contract, M561 source-static 100/100 PASS, M564 candidate-hammer 100/100 PASS, all member manifests and outer seals, VCS/foundry identity, frozen TB/core/SVA/macro sources, prior failure provenance, and the exact eight-file SHA inventory of the consumed old partial.

The release grants only a closed `run_vcs=true`, `max_attempts=1` intent. It did not run the runner, VCS, `simv`, any EDA tool, CPU/GPU experiment, or remote job, and it did not create the new result or an attempt marker. A fresh independent final-release hammer at 100/100 with P0/P1/P2 = 0/0/0 remains mandatory.

At release authoring, a different-UID `simv` process (UID 1909, PID 580855) was visible on the shared host. The release does not admit that live host state. Root must repeat a full shared-host collision check plus every immutable-runner resource/collision preflight immediately before any invocation and make a fresh go/no-go decision. Release is not execution.

`docs/359` remains frozen at `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
