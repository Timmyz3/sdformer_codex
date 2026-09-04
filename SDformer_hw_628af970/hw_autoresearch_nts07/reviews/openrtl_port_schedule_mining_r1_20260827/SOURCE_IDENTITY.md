# Source identity

Access date: 2026-08-27. Repositories were inspected read-only. No repository build, simulator, synthesis, or test command was executed.

## Official/open repositories

| Work | Repository | Frozen commit | Inspected source path | Git blob |
|---|---|---|---|---|
| FEATHER | https://github.com/maeri-project/FEATHER | `225951bfce36ce865e8a5d2165382884134536ed` | `FEATHER_RTL/RTL/sram_bank_sp.v` | `5b6e4ee2307f3380ee50df1532cbee2da1675cec` |
| FEATHER | same | same | `FEATHER_RTL/RTL/sram_sp_2d_array.v` | `4fbc97c4bb315038fc75f528dafd6cf0e8d08050` |
| FEATHER | same | same | `FEATHER_RTL/RTL/feather_controller.v` | `e863cff9d1130ca2d6a7b9739d4a74ea68beb20a` |
| FEATHER | same | same | `FEATHER_RTL/RTL/birrd_simple_cmd_flow_seq.v` | `5b5701a620f7cd99948bac6f702374414d28e18e` |
| ActiveN | https://github.com/CRAFT-THU/ActiveN | `ba03b4775711695925a364e899a939994bed6331` | `ActiveN/src/main/scala/koneko/exec/LSU.scala` | `bfc9f1e8360a1089ca8f561e0da0cf51ab25f758` |
| ActiveN | same | same | `ActiveN/src/main/scala/koneko/exec/SPM.scala` | `dbee006837827a42efaaf628700813638947bcbb` |
| ActiveN | same | same | `ActiveN/src/main/scala/koneko/bus/Distributor.scala` | `5efcf558774ab6c81fbd58a10ed0501c5b0775f7` |
| ActiveN | same | same | `ActiveN/src/main/scala/koneko/bus/Crossbar.scala` | `d76548986289e3d55b0bf5321ec25f132bd573ee` |
| ExSpike | https://github.com/xiaoyuehai/ExSpike | `51accc76936588705255487d101fcc80092b98ce` | `rtl/sparse_processing.v` | `c69b98fddd242f6acc397ea9310a00695089170d` |
| ExSpike | same | same | `rtl/weight_top.v` | `db46436a3cee7af69d4a0a739456cea4f4cd91e5` |
| ExSpike | same | same | `rtl/elastic_fifo.v` | `c6143a39d9d78c4925cd462e6fa3ac26da7a9bf4` |
| SNE | https://github.com/pulp-platform/sne | `92449df7a49f485f331dc785522b82acd33759ae` | `rtl/evt_misc_components/memory_wrapped_fifo.sv` | `31437aa9a8906fbf85e5de78bf28ec7240dd3981` |
| SNE | same | same | `rtl/evt_memory_sequencer/evt_mapper_src.sv` | `98bf706cd3bfb72ed705cc6fd08cc5accb3cced2` |
| SNE | same | same | `rtl/evt_memory_sequencer/evt_memory_sequencer.sv` | `2ed119a74447154106d2f2e6c060ee6b262fae2a` |
| Prosperity | https://github.com/dubcyfor3/Prosperity | `6ee1c6f1cb419fcf942f2eda63db84ca28248f4b` | `simulator/simulator.py` | `83a7d6d6b088257730b8ff8184b12ce93627459f` |
| Prosperity | same | same | `simulator/accelerator.py` | `51a575b14352973f59e6c2e396672cf6fdcb9fb7` |
| Prosperity | same | same | `simulator/sram_config.json` | `d73a4fff008e866ff07199c312fb039b52eef500` |
| LoAS author repository | https://github.com/RuokaiYin/LoAS | `1edd2e98aa016892338b49ca6629c3aec87e93c3` | repository tree / README / artifact index | no RTL present in frozen tree |
| LoAS Yale mirror | https://github.com/Intelligent-Computing-Lab-Yale/LoAS | `a03716a6ecd82582891fc3867c1985daeb4a4439` | repository tree / README | model and profiling code; no RTL located |

## Paper-only sources used for collision boundaries

- FEATHER, ISCA 2024: https://arxiv.org/abs/2405.13170
- ActiveN, MICRO 2024: https://doi.org/10.1109/MICRO61859.2024.00085
- SNE, DATE 2022: https://doi.org/10.23919/DATE54114.2022.9774552
- Prosperity, HPCA 2025: https://arxiv.org/abs/2503.03379
- Phi, ISCA 2025: https://arxiv.org/abs/2505.10909
- FireFly-T, accessed as a 2025 arXiv preprint: https://arxiv.org/abs/2505.12771
- LoAS, MICRO 2024: https://arxiv.org/abs/2407.14073
- ExSpike, repository identifies the work as FPL 2026: https://arxiv.org/abs/2606.20414

No official RTL repository for Phi or FireFly-T was located in this audit. This is a bounded search result, not a claim that no artifact exists.

## Local frozen evidence

| Evidence | SHA256 |
|---|---|
| `contracts/m504_h67_single_port_parent_scratch_execution_contract_r1_20260827.json` | `162e3bfdc1ae45f03d9d8da0aad64d819bbb1d6842fe925836547bf7eb7c35d6` |
| `reviews/m504_single_port_preflight_hammer_r1_20260827/m504_single_port_preflight_hammer_r1_20260827.md` | `bba610351d249f30ced3db0b4c14e3a48b107d28b0d66aeab63d6c79d3ddbc0f` |
| `reviews/tsmc28_sram_macro_audit_r1_20260827/tsmc28_sram_mapping_r1.json` | `68017fb51773713dd7dbee9463ec60d1dcdac9dea6e56588463e7f4ded96be4d` |
| `results/m473_h67_online_subset_live_pwp_r3_20260826/m473_h67_online_subset_live_pwp_result_r1.json` | `a415f8474f3a351d123670c2d3691a6414f620e3d60848a9c51242802a6956e5` |
| `contracts/m498_segmented_enable_parent_queue_logic_only_dc_contract_r1_20260827.json` | `87d77361232c637ac2b92d9ce75dfc9d1c632fbd0ba0a37f9e5719473cdc6600` |
| `docs/359_DATE终局冻结_20260813.md` | `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4` |

