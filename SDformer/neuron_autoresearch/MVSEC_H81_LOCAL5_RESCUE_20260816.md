# MVSEC H81 completion and Local5 DSEC-to-day2 rescue

Status: both jobs finished. GPU idle.

H81 is the same day2-scratch protocol as NB0/H67/Local5-scratch.
Local5-FT is a separately labeled DSEC-ep44 pretrain + 15-epoch day2 fine-tune.

## Full-sequence AEE

| method | protocol | OD1 | IF1 | IF2 | IF3 | macro | all 4 < NB0 |
|---|---|---:|---:|---:|---:|---:|---|
| NB0 | day2 scratch | 0.8450 | 1.5998 | 2.7536 | 2.1106 | 1.8273 | reference |
| H67 Motion | day2 scratch | 0.8201 | 1.5868 | 2.6258 | 2.0357 | 1.7671 | yes |
| H81 no-motion | day2 scratch | 0.8205 | 1.6248 | 2.6670 | 2.0581 | 1.7926 | **no** (IF1) |
| Local5 scratch | day2 scratch | 0.8414 | 1.6282 | 2.6679 | 2.0669 | 1.8011 | **no** (IF1) |
| Local5 DSEC-FT | DSEC ep44 + day2 FT | **0.8070** | **1.4811** | **2.4704** | **1.9159** | **1.6686** | yes |

Full-sequence spikes: H67 140.7G; Local5-FT 200.4G (+42%).

## Fixed800 AEE

| method | OD1 | IF1 | IF2 | IF3 | macro |
|---|---:|---:|---:|---:|---:|
| NB0 | 0.8379 | 1.5977 | 2.7469 | 2.1102 | 1.8231 |
| H67 | 0.8181 | 1.5850 | 2.6212 | 2.0352 | 1.7649 |
| H81 | 0.8154 | 1.6224 | 2.6628 | 2.0586 | 1.7898 |
| Local5 scratch | 0.8380 | 1.6259 | 2.6646 | 2.0650 | 1.7984 |
| Local5 DSEC-FT | **0.8050** | **1.4767** | **2.4643** | **1.9098** | **1.6639** |

## How to write this

- Same-protocol DATE MVSEC table: H67 remains the only scratch model that clears the four-sequence gate.
- Local5-FT may appear only as a transfer row. It is not the paper identity and does not inherit H67 or Local5-ep29 RTL.
- Do not mix Local5-FT numbers into the H67/H81/Local5-scratch table.
