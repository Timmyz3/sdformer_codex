# M587 request｜M579 r2 fresh source-static hammer

请对 M586 的 M579 r2 source tuple 做一次**全新、只读、独立** static hammer。不得运行正式 80-record CPU replay，不得创建 execution contract/result/attempt，不得运行 GPU/EDA/远程，不得修改 source tuple、原 M584 评审、原 r1 双封或 docs/359。

## 冻结评审对象

- `system_simulator/scripts/analyze_m579_paft_control_single_port_product_capture_r2.py`
  - SHA256 `70eb07465bb008569967f69ae0ea0d51057d64dd0d51669b604a8f1cd4d4b471`
- `system_simulator/scripts/run_m586_m579_paft_control_single_port_product_capture_r2_exact_sha.sh`
  - SHA256 `8e0efbb6c9f1e188f45fe37f4ae15b4f60f9b8cff9c533a0e822f3549aecd45e`
- `contracts/m586_m579_paft_control_single_port_product_capture_source_contract_r2_20260828.json`
  - SHA256 `319d1c895fd2327f0320c4277cc6f853d2fe8536d20406110784dc04a5fa44ec`
- `reviews/m586_m579_paft_control_single_port_product_capture_r2_source_author_handoff_20260828/`

## 必攻项

1. 实际执行精确 runner 的 `--preflight-only`，确认绝对 Python/SHA/version/NumPy、spawn child、M43/M504/M505 import 和八 row recurrence；确认零正式 record、零 result/attempt。
2. 攻击 M528 顺序：r1 worker 产出是 partition-major，r2 transpose 后必须精确为 `[sample,operator,chunk,partition]` C-order。验证 `[0,47,94,141]` anchor、20,304 tasks/operator、末 chunk 56；证明 `pipeline_cycles` 接收的确是 chunk-major。
3. 确认 M504 是 direct SHA input 且 worker import 前验证；确认 r1 base、M43、M505 也全部锁定。
4. 独立 AST/静态证明每个 64-row task 仍只调用 frozen M505 dead-write-only recurrence，bit/product/parent conservation 不变；DMA=160、tail=2/task、commit=96,000/sample、8 output blocks 两臂公平。
5. 攻击 record/cohort/plane 断言：10x4、sample/operator、shape/output shape、2,304,000 elements、三个 288,000-byte planes、offset/size/packing、negative=0、timestep support sum、basename uniqueness、80 payload SHA。
6. 攻击 M255：必须 strict parse，同时输出 global valid825、同十帧与完整 64 帧；64 帧 PAFT AEE 退化 1.0189020311889285%；single seed、无共同 evaluator runtime SHA、`accuracy_performance_pareto=false`。禁止选择性只报 valid825 +0.573%。
7. 攻击容量：M528 hammer/JSON/CSV 都要 strict/field check；候选 9 行和必须 213,376 B，240 KiB 余量 32,384 B；macro integration/PPA/energy 仍 open。
8. 攻击 runner 的原子性：默认不得执行；正式模式必须另有 launch contract；pre-attempt validation、atomic attempt、staging、terminal all-input+80-payload rehash、member/outer seal、race check、rename、success consume 与任意失败 quarantine 均无窗口。检查 signal/EXIT、partial output、二次 attempt 和 overwrite attack。
9. 攻击解释器/runner/analyzer 在长 run 前后身份，workers<=3，以及 terminal rehash 确实包含 execution contract、所有声明输入、80 payload 和 docs359。
10. 确认 arithmetic-work、local-cycle、PAFT/control activity increment 三列从未相乘；所有 system/RTL/PPA/energy/headline 标志保持 false。

## 裁决门

- 给出 100 分制、P0/P1/P2 计数和 PASS/FAIL。
- PASS 必须 score>=95、P0=0、P1=0。
- 即使 PASS，也只能允许 root **另建** launch_now=false execution candidate，不能由 source hammer 直接授权 release 或 CPU production。
- 输出 review.md、structured receipt、mechanical checks 与 member/outer 双封。
