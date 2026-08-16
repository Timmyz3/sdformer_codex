# H66 External Attention Survey

Date: 2026-07-11. Scope: top-conference primary papers and official repositories checked independently of the local redesign notes.

| work | venue | official evidence | actual paradigm | fit to full105/all12 DSEC |
|---|---|---|---|---|
| a-XNOR SSA | CVPR 2025 | [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Xiao_Rethinking_Spiking_Self-Attention_Mechanism_Implementing_a-XNOR_Similarity_Calculation_in_Spiking_CVPR_2025_paper.html) | gives silent-silent matches weight alpha because dot product ignores informative non-spike agreement | strongest accuracy-first score replacement; H66a is the full matrix oracle, TTX is its factorized selector form |
| STAtten | CVPR 2025 | [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Lee_Spiking_Transformer_with_Spatial-Temporal_Attention_CVPR_2025_paper.html), [code](https://github.com/Intelligent-Computing-Lab-Panda/STAtten) | merge time and space tokens within a fixed chunk and compute `Q(K^T V)` | DSEC Swin windows already use `T=2`; attractive for accuracy but needs V and `D x D` state |
| A2OS2A | CVPR 2025 | [paper](https://openaccess.thecvf.com/content/CVPR2025/html/Guo_Spiking_Transformer_Introducing_Accurate_Addition-Only_Spiking_Self-Attention_for_Transformer_CVPR_2025_paper.html) | binary Q, nonnegative/ReLU K, ternary V; addition-only and no softmax/scaling | accuracy-oriented but three activation alphabets complicate the neuron and attention hardware |
| SpiLiFormer | ICCV 2025 | [paper](https://openaccess.thecvf.com/content/ICCV2025/html/Zheng_SpiLiFormer_Enhancing_Spiking_Transformers_with_Lateral_Inhibition_ICCV_2025_paper.html), [code](https://github.com/KirinZheng/SpiLiFormer) | shallow feed-forward LiDiff plus deeper feedback LiDiff suppress irrelevant context | published topology is stage-dependent and feedback state changes scheduling; not an all12 first choice |
| SpikeVideoFormer | ICML 2025 | [paper and code](https://proceedings.mlr.press/v267/zou25b.html) | binary-to-bipolar Hamming attention with `Q(K^T V)` and linear temporal complexity | unified and multiplication-light, but needs `D x D` accumulators; H66b is the all-binary all12 retest |
| LRF-Dyn | ICLR 2026 | [paper](https://openreview.net/forum?id=jJedqisfOt&noteId=Mkz4JcsrMq) | local receptive-field SSA plus charge-fire-reset dynamics to avoid explicit attention matrix storage | directly motivates local a-XNOR around each flow token; better hardware path than full `N x N` H66a |
| spiking RPE | NeurIPS 2025 | [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/6fdbdaf19f4f3f8e2e08aa87987e459c-Abstract-Conference.html), [code](https://github.com/microsoft/SeqSNN) | Gray-PE and Log-PE approximate relative position while preserving spike-friendly representations | useful only after a pairwise/local attention exists; not a standalone TTX replacement |
| MaxFormer | NeurIPS 2025 | [paper](https://proceedings.neurips.cc/paper_files/paper/2025/hash/956834836f36dd07df7064ff42ca69f2-Abstract-Conference.html), [code](https://github.com/bic-L/MaxFormer) | restores high-frequency information with max pooling and depth-wise convolution replacing attention | changes the attention paradigm to convolution/token mixing; hardware dataflow diverges too far for this phase |

## Candidate order

1. H66a full binary a-XNOR matrix: accuracy oracle. It determines whether pairwise Q/K correlation adds value over factorized TTX.
2. TP-TTX: compare each Q token with same-position K at both temporal slices, normalize two candidates, and aggregate K. This adds one temporal K buffer and keeps linear token complexity.
3. LR-TTX: self plus four spatial neighbors, each using binary a-XNOR; local Shiftmax aggregates five K values. This is a fixed local stencil, not TX/SC mixing.
4. H66b binary Hamming: unified addition-only fallback with larger `D x D` state.
5. STAtten-B2: only after the above, because it requires an independent V path and a larger attention-engine redesign.

The repository's historical `h59_local` implementation is not LR-TTX: it rolls and averages already computed same-token scores. It does not evaluate `alpha-XNOR(q_i,k_j)` for neighboring token `j`, and edge rolls introduce wraparound. New LR-TTX evidence must use a separate optional branch and fresh results.
