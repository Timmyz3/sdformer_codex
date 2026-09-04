# M2210 independent source hammer

Verdict: **PASS (99/100, P0/P1/P2 = 0/0/0).** M2209 makes only the narrow nonfunctional repair established by M2200. The exact mode-0664 parser is launched as `/opt/anaconda3/bin/python3.12 -B <parser>` through a pinned regular mode-0755 interpreter. No direct parser execution, `chmod`, copy, install, or parser edit exists.

The RTL, M803 adapter, M2018 frontend, SVA, testbench, filelist, and parser match their frozen SHA-256 identities. Independent extraction confirms the VCS compile and simv command blocks are byte-identical to M2199. Ten in-memory mutations were independently injected and rejected: direct parser execution; wrong Python path, SHA, or mode; parser SHA or mode drift; missing `simv.vdb` cleanup; old result identity; retry enablement; and old-artifact reuse.

On the successful path, parser completion is followed by removal and absence checks for `simv`, `vc_hdrs.h`, `csrc`, `simv.daidir`, and `simv.vdb`; any remaining symlink is rejected. Logs, sim return code, parser log, and receipt remain. `RUN_COMPLETE.txt` is then written, the directory is exhaustively double sealed, and only then is it published.

M2200 and the M2209 author receipt were independently revalidated as exhaustive double-sealed inputs. The consumed M2199 attempt remains sealed, and all 92 regular files plus two symlinks in its unsealed failure quarantine still match the M2200 read-only snapshot. M2199 remains failed, non-citable, non-retryable, and forbidden as an artifact source. The M2211 result, attempt, and lock identities are virgin.

Exactly one M2211 execution is authorized: one license query, one VCS compile, one simv run, and one parser run; no other EDA, automatic retry, or old-artifact reuse. This source review is not an RTL, performance, PPA, power, energy, or paper result. No VCS, license query, simv, EDA, GPU, or Git mutation was performed by M2210. `docs/359` remains `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
