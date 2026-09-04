# M1725 independent M1721 source hammer

Verdict: **FAIL-CLOSED, 78/100, P0/P1/P2 = 0/2/2.** M1721 must not be released or run.

Two parts are already sound. The vector LRU matched a scalar persistent LRU in 1,000 independent random cases, and both TSBG paths use the same B-row buffer capacity. M1707 tree sealing/completeness, separate fetch/compute/roofline axes, FC2 fixed NO-GO, PATCH histogram-only blocking, fresh result namespaces, future result sealing and the paper-result=false boundary are present.

Two P1 findings block release. First, `--run-analysis` has no M1721 review/release authority gate and does not validate the source contract in production mode. A sentinel proves capture verification is reached while the contract still says analysis_run=false and release=false. Second, `sum_abs_output_code_debt` omits the number of output channels represented by each 16-output-block beta. A one-source, 32-output, all-unit-weight case has true sum debt 32 but reports 2.

The same-B row-buffer comparison is not yet full same-resource accounting because B-token accumulator/context/FIFO and broadcast-control cost are absent. The 4-byte weight traffic also has no hardware-quantization authority. Both are acceptable only as model-screening limitations, not paper cycle claims.

No capture, analyzer, GPU, EDA, release, result write, commit or push was performed.
