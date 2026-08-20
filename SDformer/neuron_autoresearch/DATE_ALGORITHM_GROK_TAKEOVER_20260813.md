# DATE algorithm queue takeover

Date: 2026-08-13. Source session: Codex `019ec76b-ea14-7862-be41-45ea956713db`.

## Policy

- H67 Motion-TTX ep35 stays the only DATE mainline.
- Local5 stays an accuracy/topology extension.
- Do not start a new GPU job while H81 owns the A800.
- Do not write hardware code or hardware docs.
- Do not delete old modules.

## Live queue

| step | owner | state |
|---|---|---|
| H81 no-motion fullres40 | `run_dsec_fullres_w15_h81_after_mvsec.py` / `train.py` | training, epoch 14/40 |
| H81 valid825 ep29/34/39 | same H81 runner | waits for `checkpoint_epoch39.pth` |
| Local5 40-50 | `run_local5_fullres_plus10_after_h81_20260812.py` | waits for H67 QF summary |
| H67 vs H81 audit | `audit_h67_h81_nomotion_result_20260812.py --wait` | waits for H81 ranking |
| Final mainline audit | `audit_date_final_mainline_20260812.py --wait` | waits for H81 result + Local5 summary |
| H67 QF5-QF8 | `run_h67_score_precision_sweep_after_mainline.py` | waits for H67/H81 result |

H81 saves only ep29/34/39. First checkpoint is therefore still hours away.

## Offline work done on takeover

- Stopped H67/H81 and Local5 waiters from appending hardware-board markdown.
- Registered NB0/H67 seed1/2 configs; they are not launched.
- Frozen valid825 voxel-L1 quartiles: Q25=556401.6, Q50=719957.9, Q75=891402.7.
- Updated the 2026-08-12 algorithm blueprint.

## Live 2026-08-17

- Table G attached: `DSEC_DENSITY_QUARTILE_TABLE_G_20260817.json` (`PASS_AEE_ATTACHED`).
- Four-line comparison: `DATE_FOUR_LINE_PAPER_FIT_20260817.md`
- Load-audit appendix: `DATE_LOAD_AUDIT_APPENDIX_20260817.md`
- Figures: `figures/date_four_line_20260817/`

## Not started

- Seed1/2 training
- Any hardware job
