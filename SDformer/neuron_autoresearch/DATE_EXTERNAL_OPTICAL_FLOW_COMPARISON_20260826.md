# DATE External Optical-Flow Comparison Protocol

## Decision

The paper needs external optical-flow context, but it must not mix local DSEC
`valid825` with official hidden-test results. Downloading and retraining every network
is unnecessary. Use two explicitly separated tables:

1. **Official DSEC-Flow test:** published/server results for representative methods;
   add our row only after an official submission.
2. **Local controlled ablation:** C00/C10/C12 on the exact same parent, split,
   resolution, evaluation epochs, and local valid825 evaluator. Do not place external
   official-test values in this table.

## Representative Official DSEC Rows

The following rows are literature/server values, not local reproductions. They are a
paper-table scaffold and must be rechecked against the cited primary source when the
final manuscript is frozen.

| method | model class | EPE/AEE | AE (Barron/Middlebury) | 3PE/Fl (%) | role in comparison |
|---|---|---:|---:|---:|---|
| E-RAFT | ANN correlation/recurrent | 0.788 | 2.851 | 2.684 | foundational dense event-flow baseline |
| TMA | ANN temporally dense aggregation | 0.743 | 2.684 | 2.301 | temporal-motion baseline |
| E-FlowFormer | ANN transformer/cost volume | 0.759 | 2.676 | 2.446 | transformer baseline |
| IDNet | lightweight ANN iterative deblurring | 0.719 | 2.723 | 2.036 | efficiency-oriented ANN baseline |
| EDCFlow | ANN difference + correlation | 0.720 | 2.65 | 2.10 | recent efficient event-flow context |
| OF_EV_SNN | convolutional SNN | 1.707 | 6.338 | 10.308 | prior SNN baseline |
| SDformerFlow-v2 | spikeformer parent | 1.602 | 4.871 | 10.051 | nearest parent model |
| Proposed ATLIF + Complete TTX | unified SNN | pending official submission | pending | pending | proposed row |

Primary sources:

- DSEC benchmark and metric definitions:
  https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/
- SDformerFlow paper/submission table:
  https://dsec.ifi.uzh.ch/wp-content/uploads/sourcenova/uni-comp/optical-flow-benchmark-v1-0/submissions/269/details.pdf
- EDCFlow CVPR 2025 table and efficiency columns:
  https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_EDCFlow_Exploring_Temporally_Dense_Difference_Maps_for_Event-based_Optical_Flow_CVPR_2025_paper.pdf
- TMA ICCV 2023 protocol:
  https://openaccess.thecvf.com/content/ICCV2023/papers/Liu_TMA_Temporal_Motion_Aggregation_for_Event-based_Optical_Flow_ICCV_2023_paper.pdf
- IDNet official DSEC entry:
  https://dsec.ifi.uzh.ch/uzh/dsec-flow-optical-flow-benchmark/idnet/

## AE-3D Naming

`AAE_Benchmark` in this repository is not a new metric. It implements the classic
angular error between `(u, v, 1)` and `(u_gt, v_gt, 1)` in degrees. DSEC names this
metric `AE` and points to the Middlebury/Barron evaluation methodology. In our
internal tables, `AE-3D` is only a disambiguating label that separates the official
formula from the legacy two-dimensional direction-only `AAE` implementation.

Paper terminology:

- Use **AE** in the official DSEC table.
- Use **AE-3D (official AE formula)** in local diagnostic tables if AAE-2D is also
  retained.
- Never claim that AE-3D is proposed by this work.

Middlebury formula reference:
https://vision.middlebury.edu/flow/floweval-ijcv2011.pdf

## Reproduction Budget

Do not clone and retrain all listed models. If a reviewer requests local same-split
comparisons, reproduce at most three open-source representatives:

1. SDformerFlow-v2, because it is the exact parent family.
2. OF_EV_SNN, because it is the nearest prior SNN architecture.
3. One efficiency-oriented ANN, preferably IDNet, as a hardware-relevant context.

E-RAFT/TMA/EDCFlow remain official-result context unless the paper makes a direct
same-hardware runtime claim against them. Any local reproduction must be labeled
`local reproduction`, include checkpoint/config identity, and never replace the
official benchmark row.
