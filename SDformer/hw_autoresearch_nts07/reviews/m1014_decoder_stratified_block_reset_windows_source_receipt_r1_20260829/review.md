# M1014 source-only receipt

Status: `PASS_M1014_SOURCE_ONLY__INDEPENDENT_HAMMER_REQUIRED_BEFORE_EXECUTION`.

M1014 implements the M1009 block-reset measurement protocol without opening a real decoder payload. It supplies the paired boundary/fill/body/drain adapter, D0/D2/D3 route identities, strict D1 rejection, the four frozen strata, cycle-blind deterministic sampling, exact replay and the frozen finite-population paired estimator.

The only executable workload was a small `M890_SYNTHETIC` self-test. Its 448-request body became 451 requests after three explicit reset requests. Candidate and baseline each measured 649 cycles and 64 commits; the M768/M861/M890/M896 exact miter, cycle-class conservation, terminal drain and outstanding-return drain all passed. This equality tests paired accounting; it is not a speedup result.

Seven unit tests passed. Static validation confirms that the CLI exposes only `--validate-source` and `--self-test`, contains no frozen real-prefix/payload generator call, keeps pilot=8, max=32, window cap=10,000, and refuses D1 scheduling. Python compilation also passed.

No real window, full row, CPU production run, GPU, remote host, VCS or DC was launched. These sources are not paper-citable and cannot populate Table A. An independent agent hammer and separate execution release are mandatory before running real D0/D2/D3 pilot windows.
