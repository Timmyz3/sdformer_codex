# M965 | M964 final launch-release hammer

Verdict: `GO`, score 98/100, P0=0/P1=0/P2=2.

The exact double-sealed M964 release legally authorizes one future M962
3.000 ns macro-aware setup/area DC attempt and no other EDA work. M965 did not
launch DC or consume the attempt.

M964 pins the exact M962 runner, source contract, Tcl, SDC, two-entry filelist,
M935 RTL, nine-macro wrapper and `docs/359`. All five M962 source payloads and
their nested SHA sidecars validate. The recursively sealed M963 gate also
validates and preserves the narrow rule: only the unique M923 wrong-parent
negative-test assertion is excepted; every unexpected assertion or fatal/error
still stops launch.

The runner requires both exact caller pins, consumes the single attempt before
DC, disallows retry and replacement, blocks same-UID DC, and repeats memory,
swap, commit, license, source, tool and foundry checks. It allows one
`compile_ultra`; incremental compilation and timing exceptions are absent.

At hammer time all canonical result/attempt/lock/work/failure namespaces were
fresh, same-UID DC count was zero, and live resources exceeded the frozen
96 GiB MemAvailable, 16 GiB SwapFree and 64 GiB commit-headroom gates. An inert
clean-UTF-8 invocation without either caller pin stopped with rc=3 before
resource/license/attempt/DC work and left the namespace fresh.

A complete negative setup result is not discarded: it must be published and
double-sealed with WNS, TNS, violation count and top-100 paths. Tool/link/macro
or incomplete-report failures are sealed in quarantine. Positive timing requires
setup to meet 3 ns.

The authorized caller pins are:

- `M962_EXPECTED_DC_RUNNER_SHA256=7ec1138696c40b923d6841dc21749aed35e93da266e00910b6715278c51da7fd`
- `M962_EXPECTED_DC_RELEASE_SHA256=9d47a2c204bf89204ec124214ed64935a8fcc401d2ed34f5a881006f8c3bb1d2`

P2: the memory gate must be called 96 GiB, not 100 GiB; and all live gates are
snapshots that the runner must repeat at launch. Timing, setup, hold, PPA, power,
cycles, speedup, system and paper claims remain false.
