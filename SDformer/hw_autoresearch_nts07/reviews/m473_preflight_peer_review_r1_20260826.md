# M473 online-subset/live-PWP 预飞行严苛评审

评审对象：`contracts/m473_h67_online_subset_live_pwp_preflight_contract_r1_20260826.json`，评审时 SHA256 为 `8d57080c1b486a3653d9ad1c8ee2387d809f26d81a042d5dc3ab1a12b153f8b7`。

## 结论

**REVISE BEFORE PRODUCER，62/100。** 机制本身值得继续：官方 Prosperity 的 subset-parent 语义可以转换成 H67 的精确 parent-result reuse，按 `(popcount,row_index)` 发射也确实构成 DAG 拓扑序；但当前合同尚不能生成可准入的周期和 240-KiB 数字。主要缺口不是算法语义，而是 bucket 建造、CAM 资源公式、parent scratch 地址、同步读延迟、row-result 到 signed19 psum 的提交周期，以及小型调度存储被排除在容量门之外。

当前允许做的只有：修订合同、冻结 analyzer SHA、做 CPU DSE。当前不允许启动 producer，更不允许把未来 CPU 点称作 RTL、PPA、系统或 DATE headline。

## 已确认正确的部分

1. **官方 subset 语义描述正确。** 官方 `find_product_sparsity` 对当前原始 16-bit mask 搜索其子集；相同 mask 只允许更低原始行号，严格子集可位于 tile 内任意原始行号；最大 subset popcount 后按最低原始行号打破平局。`residual=current XOR parent` 也正确。候选比较必须始终使用 original masks，不能使用已变换 residual。
2. **popcount 拓扑证明成立。** 严格子集的 popcount 必然更小；相同 mask 的合法 parent 行号必然更小。因此稳定升序 `(original_popcount, original_row_index)` 保证所有 parent 在 child 前完成，不改变官方 parent 选择。
3. **单 output-block scratch 在功能上可与 8 blocks 串行共存。** 对同一 partition，descriptor/parent DAG 与 output channel 无关；每个 96-lane block 可重放同一 descriptor 顺序，block 间清空 signed12 scratch，而各自 signed19 psum bank 跨 432 partitions 保留。前提是每个 block 都完整计入 issue、scratch read/write、row completion 与 psum commit。
4. **最新 4-bank 修订正确且必要。** 4 个 psum banks 不能同时让两个 output half 跨 432 partitions 常驻，因此必须执行两个独立的 432-partition half sequence。第二个 half 必须重新 source scan、popcount、online match、reference count 和 descriptor，不能借用第一个 half 的 descriptor。当前合同已经明确这一点。
5. **最新 same-resource 修订正确。** product 与 bit 的 same-resource 比较现在固定为同一 `row_tile / banks / bandwidth / CAM lanes / scratch allocation / schedule` 坐标，不再跨 row tile 冒充 same-resource。这是合法的 iso-coordinate mechanism ablation。
6. 所有冻结输入 SHA 当前均匹配；Prosperity 仓库为 `6ee1c6f1cb419fcf942f2eda63db84ca28248f4b` 且 clean；`docs/359` SHA 仍为 `dedde7ce...`。

## P0：producer 前必须修订

### P0.1｜“online”必须定义为完整 tile capture 后的 runtime matching

官方严格子集 parent 可以来自更高原始行号。因此真正的单遍到达即发射不可能复现官方语义；必须先捕获整个 bounded row tile，再完成 parent discovery，最后按 bucket 拓扑序执行。

合同应明确三阶段：

1. capture `valid_rows` 个 original16 与原始 row ID；
2. 仅对 `popcount>1` 的行，面向整个 tile 的 original masks 做 subset search 并冻结 parent/residual/refcount；
3. 按稳定 `(popcount,row_id)` 顺序重放 descriptor。

如果仍称 “online”，必须限定为 **online at inference/runtime, tile-buffered**，不得暗示 streaming one-pass issue。

### P0.2｜bucket 构造吞吐目前没有可实现合同

`ceil(valid_rows/8)` 假设每拍接收 8 个 popcount 并同时写入 17 个稳定 bucket。最坏情况下 8 行落入同一个 bucket，普通单写口 FIFO 做不到一拍 8 enqueue。当前合同既没有 8-way scatter/multiwrite 架构，也没有为冲突、stable ordering、merge/drain 计周期或存储。

必须二选一并冻结：

- 8 个 lane-local × 17 bucket FIFO，随后稳定 merge；明确 enqueue/merge 周期、row-ID 位宽和容量；或
- 先把 `(popcount,row_id)` 写入 metadata，再做 counting-sort/多遍 scan；把额外 pass 计入 frontend。

仅写“17 popcount bucket queues，PPA 再看”不足以支撑当前 frontend 公式。

### P0.3｜CAM 资源公式与 cycle 公式自相矛盾

周期式 `search_rows * ceil(valid_rows / cam_compare_lanes)` 表示物理上只有 `cam_compare_lanes` 个并行 16-bit subset comparators，每拍检查一组 candidate。容量段却写成 `cam_compare_lanes × row_tile × 16-bit subset comparators`，多乘了一个 row_tile；两者不是同一个硬件。

修订为：

- candidate store：`row_tile × 16 original-mask bits`，由 source ping-pong 或明确的寄存器/CAM bank 提供；
- compare fabric：`min(cam_compare_lanes, valid_rows)` 个 16-bit subset comparators；
- 每拍 chunk 内最大-popcount/最低-row-ID reduction，加跨 chunk running winner；
- 明确 compare pipeline fill、final reduction、parent/refcount finalize 周期。

不同 CAM lane 点不是同资源，结果必须附 comparator count、candidate-store read-port/banking 和 reduction proxy，禁止跨 CAM lane 直接宣称同资源提升。

### P0.4｜row contribution 到 psum 的完成周期被遗漏

当前 `issue_cycles_per_output_block = residual_popcount`（exact parent 为 1）只计算 source/parent fold，没有定义完成的 signed12 row contribution 如何写 parent scratch、并加到跨 432 partitions 的 signed19 psum。

必须冻结以下时序之一：

- 独立 signed12 row-fold datapath + 独立 signed19 psum RMW datapath，最后一个 residual cycle 可同时完成 scratch write 和 psum update；需要明确双级 forwarding/关键路径；或
- 每个 active row 另付 1 个 completion/psum-commit cycle。

这一项可能多出 `active_rows × 8 blocks` 量级的周期，不能等 RTL 阶段才补。bit peer 必须使用完全相同的 row-fold/psum-completion规则。

### P0.5｜同步 scratch 延迟与 RAW forwarding 未进入 cycle model

144-byte 1R1W scratch 是 96×signed12。物理 SRAM 通常是同步读：parent request、read data、第一 residual fold 至少涉及一个明确 pipeline。连续的 parent→child 或 exact-equal child 不能默认零延迟。

合同须给出：scratch read latency、write-to-next-child RAW 行为、same-cycle 1R1W、write-first/read-first 语义以及 forwarding 命中规则。CPU DSE至少同时报：

- optimistic fully-forwarded 点；
- conservative 1-cycle parent-read/RAW-stall 点。

只有后者或被 RTL/VCS 证明可隐藏的点能进入 nomination。

### P0.6｜scratch traffic 必须按 parent edge，而不是 unique parent row

一个 parent 可有多个 direct children，每个 child 都需要一次 144-byte parent read。当前字段 `pwp_scratch_read_bytes_per_parent_row_per_output_block` 容易只按 unique parent 计一次，会系统性低估。

正确账本为：

- reads = `direct_parent_edges × output_blocks × 144 B`；
- writes = `rows_with_refcount_gt_zero × output_blocks × 144 B`；
- peak capacity = issue transcript 上、在同拍读释放前的最大 live parent entries。

sidecar 必须新增 `parent_edges`、`unique_parent_rows`、`scratch_reads`、`scratch_writes`、`peak_live_before_reclaim`，并检查 reads 等于 parent-valid child 数。

### P0.7｜compact peak-live scratch 缺少 row→slot 映射合同

若 scratch 用 original row ID 直接寻址，物理深度必须按 `row_tile`，不能按 peak-live。若只按 peak-live 分配，则需要 row-ID→slot map、free list、slot width、更新端口和 RAW 行为；当前 32-bit descriptor 没有容纳这些字段，相关存储也未计容量。

必须二选一：

- **保守首选**：row-indexed scratch，深度=`round64(row_tile)`，无需 compact mapping；或
- compact scratch：显式计入 row→slot map/free-list，并在 transcript 中证明分配/释放无冲突。

不得同时使用 peak-live 容量和免费的 row-index寻址。

### P0.8｜bucket/refcount/map 是存储，不能排除在 240-KiB 门外

CAM comparator 和 reduction logic 可以只进入 PPA proxy；但 bucket row IDs、reference counts、scratch slot map/free list、descriptor metadata 都是 live storage。把它们列入 `not_in_capacity_but_required_for_ppa` 与“所有 live memory 过 240-KiB”原则冲突。

建议把 metadata 扩成每行 64 bits 并 ping-pong，包含 residual、parent ID/valid、active、popcount、refcount、bucket-next/current-row metadata；小型 bucket head/tail 可计入 itemized fixed reserve。即使 64-bit descriptor 的 144-bit-slice macro宽度可能与当前 32-bit 相同，logical bytes 和端口仍必须如实列出。

## 容量公式必须写死

令 `R=row_tile`、`B∈{4,8}`、`D=64*ceil(R/64)`、`L=peak_live`。至少应逐项输出：

- psum logical = `ceil(R*B*96*19/8)`；macro = `B*13*18*D` bytes；
- source ping-pong logical = `2*R*2`；macro = `2*18*D`；
- descriptor/metadata ping-pong按实际位宽计算；macro宽度片不得小于 logical；
- weight logical = `(B/4)*6144`；macro = `(B/4)*22*18*64`；
- parent scratch logical = `144*L`；macro = `8*18*round64(L)`；若 row-indexed 则令 `L=R`；
- bucket/refcount/map/free-list 与 fixed reserve 分列，不能只写一个不可审计的 16 KiB 总数；
- 每个 macro item 必须满足 rounded bytes ≥ logical bytes，再求总和。

用 row-indexed scratch 做保守 sanity check，暂不计新增 metadata，得到：

| banks | row tile | M468-like base macro | + one-block scratch | 240 KiB |
|---:|---:|---:|---:|---:|
| 4 | 64 | 108,544 B | 117,760 B | PASS |
| 4 | 128 | 175,360 B | 193,792 B | PASS |
| 4 | 192 | 242,176 B | 269,824 B | **FAIL** |
| 8 | 64 | 193,792 B | 203,008 B | PASS |
| 8 | 96 | 320,512 B | 338,944 B | **FAIL** |

因此 4-bank/row192 只要存在任何 parent scratch macro 就不可能过门；8-bank 最大实际可行 row tile 很可能仍是 64。producer 必须从公式得出，不能用平均 peak-live 或零深度小宏逃门。

## matcher 与 peer baseline 公平性

### 合同还需明确的周期式

`product_frontend` 不应只有 `scan + search_rows*chunks + 2`。应拆成：

`capture/popcount + stable_bucket_build_or_merge + subset_compare_chunks + winner_finalize/refcount_update + descriptor_ready/drain`。

每项须说明能否重叠。8-bank 的 “two half-weight-DMAs” 必须写成精确字节和命令式，例如它究竟是一个 12,288-byte command，还是两个 `6144-byte + 32-cycle setup` 的串行 command；4-bank 每个 half 独立支付一次。第一 task、相邻 task 和最后 drain 的公式也必须直接写出，不能只引用 M468 的“frozen equation”，因为 M468 独立评审已指出旧恢复文本存在 stream-boundary 歧义。

### baseline 裁定

- exact same-coordinate bit peer：**公平，保留**；bit peer 是否仍通过 bucket scheduler必须明确。若 bit 直接原序发射而 product 用 bucket，属于同硬件最佳 bypass，反而是对 product 更保守，但不要再写“schedule unchanged”。
- M468 best strong-zero：可作同 block-bank/BW、同 240-KiB 门的架构 baseline，但不是 iso-area，因为 M473 新增 CAM、reduction、bucket、1R1W scratch。合同还应冻结 `m468r6_independent_hammer_review`，而不仅是 producer result。128 B/cycle 的独立复算锚为：4-bank `752,580,192` cycles（row192），8-bank `760,350,133` cycles（row64）。
- M430 `517,041,352`：只保留 frontier diagnostic，现有边界正确。
- M472 `2.459487x`：只作机制动机，现有边界正确。

### materiality 门是否夸大

`≥1.75x` exact-coordinate bit 且 `≥1.50x` best M468 zero 是偏保守而不是宽松门，本身不会夸大。但 nomination 还必须增加：

1. 报 candidate 相对 **best feasible M473 bit point** 的鲁棒性比值，明确它不是 same-coordinate ratio；
2. `same_resource_across_cam_lanes=false`，禁止从 16 到 256 comparators 的点当同一硬件挑最快；
3. CPU nomination 始终 `performance_admitted=false`，并标明 `area_unclosed=true`；
4. 若只有 optimistic zero-latency scratch 模型过门，而 conservative 模型不过门，则 NO-GO RTL。

## exact arithmetic 身份补强

M473 使用任意 runtime parent mask，不再只使用 M468 的 32 个 catalog centers。signed12 的数学上界 `[-2048,2032]` 对任意 16 个 INT8 权重子集成立，方向正确，但合同必须冻结相应证明回执或四个 INT8 weight payload 并在 producer 中复算。signed19 accumulated psum 的既有证明身份也应冻结。否则 `exact_arithmetic=true` 只有代数等价，没有位宽不溢出证据。

## official mapping validation 的最低要求

128 个检查不能随意抽样，应分层覆盖全部 6 个 row tile、4 operators、10 samples，并强制包含：zero、pop1、no-parent、strict parent 位于更高原始行号、equal-mask lower-index、max-pop tie、exact-parent residual0、partial last tile。每个检查比较完整 `parent_id / parent_valid / residual16 / processed nnz`，并验证：

- parent 属于同一 bounded tile；
- `parent & residual == 0` 且 `parent | residual == current`；
- parent 与 official reference 完全相同；
- issued parent position < child position；
- DAG 无环且 refcount 总和 = parent edges。

另外应加入小型 synthetic exhaustive suite，避免真实 trace 没覆盖 tie/RAW/peak-live 极端。

## 修订后的 GO 条件

只有以下全部写入新 SHA 的 r2 preflight 后才 GO producer：

1. 明确 tile-buffered 三阶段语义和 stratified official mapping；
2. 解决 8-way bucket enqueue/merge，修正 CAM comparator 公式；
3. 冻结 row-fold、scratch latency/forwarding、psum completion 和 4/8-bank 精确流水式；
4. scratch reads 按 parent edges，compact mapping 不免费；
5. 所有调度存储进入 logical/macro 240-KiB 门；
6. 冻结 M468 independent hammer 与 signed12/signed19 位宽证据；
7. 同时输出 optimistic 与 conservative scratch timing，只有 conservative 点可 nomination；
8. analyzer/execution contract 在运行前冻结 exact SHA，当前 dangling 的 “Fill every TO_BE_FILLED” 改成具体字段检查。

若 producer 在未修订上述 P0 的情况下运行，评审结论是 **NO-GO admission**，即使表面倍速超过 1.75x/1.50x 也不能使用。
