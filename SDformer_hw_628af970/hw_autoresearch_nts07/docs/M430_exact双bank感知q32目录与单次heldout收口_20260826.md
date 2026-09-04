# M430 exact 双-bank 感知 q32 目录与单次 held-out 收口（2026-08-26）

## 结论

M430 将 q32 目录的训练目标从 M423 的串行 PWP 代价 `2+d` 改为合法 dual co-read 代价 `1+d`，并保留 bit-sparse fallback `popcount(x)`。在目录双封以后，只完成了一次冻结 M40 S10 / 17,280-phase held-out replay：

- M430 dual：`517,041,352` cycles；
- 相对强 zero-elided baseline `742,148,386`：`1.435375301x`；
- 相对原 M338 dual `530,606,660`：`1.026236408x`；
- 相对 M423 目录的 dual 投影 `527,837,132`：再省 `10,795,780` cycles（`2.045286%`），即 `1.020879916x`；
- 相对 M401 serial `641,790,704`：`1.241275386x`。

因此目录选择为 `GO_M430_DUALAWARE_CATALOG`。这只是四个冻结 H67 ep35 bottleneck Conv3x3 的 exact architectural cycle replay，不是全网/系统倍速，也不是 RTL、Synopsys、功耗或 PPA 结果。

## 训练与 held-out 隔离

- M77 q16 的前 16 项逐 partition 位同一且顺序不变；
- q32 的 16 项 tail 只允许来自对应 M338 q128 IDs 16..127；
- 目录只读取 M73 DSEC train-only S32、18 sequences；
- train 与 M40 held-out sample key 集合为 `32 / 10 / overlap=0`；
- 三个确定性 train 候选为 M338 q32、独立 marginal-gain 和 sequential greedy；1,728 个 partition 最终都由 dual greedy 胜出；
- train 固定 M338 q32 为 `1,681,063,835` cycles，M430 hybrid 为 `1,632,333,987`，训练侧下降 `2.898751%`；
- catalog / train audit / per-partition ledger 在第一次 M40 payload read 前写入 `SHA256SUMS` 并再封 `SHA256SUMS.seal.sha256`；
- M40 one-shot marker 在第一次 payload read 前生成，最终只有一个 payload-read invocation、一个完整 17,280-phase evaluation，之后没有目录调参。

## exact 与 population 账本

held-out 共 `51,840,000` 个 16-bit source row：

- zero：`24,534,432`；active：`27,305,568`；
- PWP：`15,909,646`；其中 exact-pattern PWP `5,048,754`，positive-residual PWP `10,860,892`；
- fallback：`11,395,922`；
- signed correction：`38,055,489 ops/output-block`；
- PWP + correction dual issue：`53,965,135 ops/output-block`；
- used PWP pattern occurrences：`548,711`；center runs：`20,941`；
- q16 prefix matcher：`67,912,100` cycles，与冻结 M401/M423 完全一致；
- static codec：`442,368` blocks / `42,467,328` lanes，signed12、wide/narrow reconstruction、padding 全部 0 mismatch，PWP 范围 `[-1089,1059]`。

运行恒等式仍是：

`old_psum += PWP[p] + signed W*(x-p)`；fallback 为 `old_psum += W*x`。

M427r3 已撤销的 seed-first-correction fusion 未使用，persistent `old_psum` 没有被 PWP 替换。

## 周期口径

每个 output tile 的 active work 为：

`4 * correction_ops_per_block + 4 * pwp_rows`

即每个 PWP output block 都使用一次同拍 low8/high4 co-read；没有 M423 serial 的第二拍，也没有 narrow subtraction。其余与 M401/M423 保持同口径：q32/O4/SHARED96 控制骨架、cmd32、descriptor L8、FIFO D8、640-byte center stride、32 B/cycle DRAM、tile DMA、双 tile max-overlap、tail 和 per-sample commit。

两个独立周期复算均闭合：component sum 与 17,280 条 timestamp phase-delta 加 commit 均等于 `517,041,352`。

## 端口与 traffic 红线

dual co-read 不是 SHARED96 的免费升级：

- peak PWP source：`144 logical B/cycle`（low8 96 B + high4 48 B）；现有 padded 接口为 `160 B/cycle`；
- strong-zero/correction source reference：`96 B/cycle`；
- PWP output-block issues：`127,277,168`；
- PWP logical on-chip reads：`18,327,912,192 B`；padded signal：`20,364,346,880 B`；
- correction reads：`29,226,615,552 B`；
- PWP DRAM physical payload：`702,350,080 B`；weight DRAM：`212,336,640 B`。

所以 `1.435375301x` 只能作为新 dual-port standalone module 的 cycle point。下一步必须用独立 RTL/VCS/形式等价/DC/PT 和 common-area/common-bandwidth reference 报 throughput-area/port Pareto；宏和 PTPX 未闭合前不得升级为功耗/能效/论文 headline。

## 证据

- train contract：`contracts/m430a_trainonly_dualaware_q32_catalog_contract_r1_20260826.json`
- train result：`results/m430a_trainonly_dualaware_q32_catalog_r1_20260826/`
- held-out contract：`contracts/m430b_h67_dualaware_q32_heldout_once_contract_r1_20260826.json`
- held-out result：`results/m430b_h67_dualaware_q32_heldout_once_r1_20260826/`
- local postrun consistency receipt（非独立评审）：`results/m430c_local_postrun_verification_r1_20260826/`
- exact-SHA runner：`system_simulator/scripts/run_m430_exact_dualaware_q32_chain_exact_sha.sh`
- train outer seal SHA256：`622bd5fea18ee20cb49eaf4c74dd2a311a56250d7d21a0e453cf7bb90f8a547c`
- held-out outer seal SHA256：`462501b849f42f1a0690d2fe8dbe3dc226e83ae05dea86f7cb0396d60e9faf7e`
- docs/359 SHA256（未修改）：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`
