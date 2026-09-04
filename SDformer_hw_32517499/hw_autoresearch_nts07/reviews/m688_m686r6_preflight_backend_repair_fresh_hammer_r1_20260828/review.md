# M688｜M686-r6 preflight/backend repair fresh static hammer

## Verdict

**Scoped GO: M686-r6 may create the CPU exact-load preflight only. GPU one-shot remains unauthorized until the new preflight receives an independent two-root review.**

Score: **97/100**. Severity: **P0=0, P1=0, P2=1**.

Both M685 P1 findings are closed.

## M685 P1-1 closure

After parsing the semantic receipt, the runner now executes a second complete `SHA256SUMS` member check and outer-link check. It then compares the independently reviewed `preflight.json` and outer-seal-file SHA roots, followed immediately by attempt creation.

The old attack was replayed independently. After a valid initial double seal, `RUN_COMPLETE.txt` was changed while `preflight.json` and the outer-seal file were left untouched. The external pair would still match, but the new second member check returned 1 and no attempt directory was created. The repaired order is therefore operational, not merely documentary.

## M685 P1-2 closure

The producer no longer equates `resolve_snn_backend(config) == cupy` with executed CuPy. Its runtime gate now performs a complete `model.named_modules()` backend-attribute inventory and requires:

- zero surviving `PSN` backend targets and zero effective CuPy assignments;
- 105 `ATLIFTernaryPSN` modules, none with a backend attribute;
- four `IFNode:torch` modules; and
- 49 `Dropout:torch` modules.

Independent static inspection confirms that `ATLIFTernaryPSN` has no backend assignment and its frozen implementation uses `torch.addmm`. The ATLIF implementation, original spiking-submodule source and backend resolver source are all exact-SHA contract inputs. CuPy is now described only as configured, resolved and installed—not as the effective execution backend.

## Remaining P2

The assigned post-preflight test is intentionally not executable yet: the M686-r6 preflight is absent and the file still contains earlier hard-coded receipt identities. After the CPU-only preflight is produced, those roots must be updated, the test must run, and a separate independent preflight review must explicitly authorize the one-shot. This does not block CPU preflight creation.

## Other passed gates

- Five assigned target SHA identities match.
- Compilation, strict JSON parsing and runner shell syntax pass.
- Main author suite: **23/23 PASS**.
- Required input population: **45/45**, all exact SHA matching.
- M685 review/manifest/outer roots and complete double seal verify.
- Native cuDNN TF32 and the remaining deterministic controls retain live checks after config/model, around every sample and before finalization.
- S00/D0 remains fail-closed at 839586 ones, 3768414 zeros and packed SHA `ad2251f...`.
- D1 scrub, exact packing, nested seals and folded-miter boundaries remain covered.
- M686-r6 preflight, canonical output and attempt were absent throughout M688.

M688 executed no CPU preflight, GPU workload, payload capture, performance simulation, RTL or EDA action.
