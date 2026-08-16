# DATE algorithm integrity audit

Date: 2026-08-13. Status: `PASS_WITH_NARROWING`.

H81 is still training (epoch 14/40). This audit does not steal the GPU. Main-table
numbers were re-read from `spike_profile.json`, not from secondary markdown.

## What is solid

| claim | source | check |
|---|---|---|
| NB0 DSEC AEE 1.4454 / spikes 126.12G | plus10 `epoch29` | match |
| H67 DSEC AEE 1.3297 / spikes 82.11G | plus10 `epoch35` + identity contract | match, SHA still binds |
| Local5 DSEC AEE 1.3153 / spikes 84.42G | plus10 `epoch39` | match |
| Table C 30/35/40 points | same three run dirs | match |
| valid825 list | SHA `7f3dc280…`, 825 frames, 18 sequences | both path aliases share one inode |
| H67 identity | checkpoint / config / profile / HW evidence | all six SHA checks pass |
| MVSEC H67 vs NB0 | four sequences all improve, spikes −44% | gate PASS |
| MVSEC Local5 | indoor_flying1 worse than NB0 | correctly disqualified |
| July 28 AEE≈2.07 runs | first broken fullres | not used in the paper table |

## Holes that change how the paper may talk

**1. H81 is not a step-paired Motion ablation.**
Crop H67 vs crop H81 is clean: same parent, only `binary_motion_xor_alpha`.
The *paper* H67 number is later plus10 `ep35` after a five-stage fullres rescue.
Live H81 is one uninterrupted 40-epoch fullres job. Do not write “the only
difference is Motion-XOR at every optimizer step.”

**2. DSEC valid825 is local validation.**
NB0 AE-3D 6.18 is not the official hidden-test 4.871. Internal H67/NB0/Local5
deltas are valid. Absolute comparison to the SDformerFlow paper row is not.

**3. MVSEC models are not the DSEC checkpoint.**
They share an NB0 initialization and a frozen day2 manifest. They do not carry
DSEC H67 `ep35` weights or RTL provenance.

**4. Seed0 only.**
seed1/2 configs exist and are not launched.

## Smaller issues, already contained

- The 2026-08-05 closure file still lists Local5 as ep29 / 1.3286. Table B is
  already on ep39 / 1.3153. Treat the August 5 Local5 row as historical.
- NB0 uses `pretrained_window_size [2,9,9]` with window `[2,15,15]`. H67/H81
  use `[2,15,15]` for both. Same geometry, different relative-position recipe.
- H81 in-training val is `n_valid=1` and is noise. Selection stays valid825
  ep29/34/39.
- Energy is a spike proxy.
- QF5–QF8 only needs H67 ep35. This was remediated after the audit: QF now runs
  immediately after H81 ranking/audit, before Local5 40–50.
- MVSEC smoke `overlay=0 missing=210` is the intended NB0-init signature.

## What I did not change

No training job, no paper mainline, no hardware file. The H81 result waiter
now writes the plus10 / not-step-paired boundary when H81 finishes.
