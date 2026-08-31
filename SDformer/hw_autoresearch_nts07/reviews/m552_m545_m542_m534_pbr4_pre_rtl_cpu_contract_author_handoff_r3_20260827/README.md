# M552｜M545/M542/M534 PBR4 pre-RTL CPU contract r3 语义修复交接

日期：2026-08-27  
状态：`R3_SOURCE_ONLY_TRANSITION_REPAIR_COMPLETE__RUN_AUTHORIZED_FALSE__FRESH_STATIC_HAMMER_REQUIRED`

本交接只新增 r3 contract、future-runner schema、handoff、fresh static hammer request 与双封文件。没有运行
候选 CPU/analyzer，没有建立 runner、result 或 attempt marker，没有修改 RTL，没有运行 VCS/iverilog/
Verilator/DC/PT/PTPX/Formality/训练/GPU/远端任务，也没有修改 M534 r2/r3/r4、r2 contract、`docs/524`
或 `docs/359`。

## 1. 被审对象

- contract：`contracts/m552_m545_m542_m534_pbr4_pre_rtl_cpu_execution_contract_r3_20260827.json`
- contract SHA256：`16119c935cd4357da477fee7b0416dcbb38a3c467a7d95c9e8b3b7487f5ebb57`
- member sidecar file SHA256：`9242bab37ac9bc1589fcb33689df27b477b4813b9e9cb9dd6c14a3a88559f89f`
- future schema SHA256：由本目录 `SHA256SUMS` 冻结；schema 明确把最终 launch authorization 定义为
  final-release hammer 之后另行生成、再由未来 runner source 冻结 SHA 的独立文件，避免 review 自引用。

## 2. M549 P1 closure

r3 把每个 local cycle 定义成 prior-state guard evaluation、唯一 winning action、atomic state delta 与唯一
primary cycle class。公共优先级先处理 fault、FINAL_OUTPUT hold、external/directory、L4 response/pending
write、active service，再进入 architecture transition、frontier、bundle 与 block/time transition；任何合法
普通状态若没有唯一 next state 都 fail closed。

- `A1-SC8`：当前 atomic bundle 余下 lane 按 stable order、同 Cin tile、唯一 weight bank 与唯一 phase
  贪心形成 round；1--7 lane partial 在本 bundle 扫描尾立即 flush，禁止等待下一 event/bundle。
- `A1-ISO8`：只看 head 与紧邻 lane；full destination/epoch/tile 相等且 bank 不冲突才成 pair，否则 head
  singleton 当场 flush，禁止 deeper search。
- `A1-OSG`：`serviceable` 唯一定义为 `FULL || selected PRESSURE victim || CLOSE || BLOCK_DRAIN`；movable
  ingress 先 move，blocked ingress 选择 matching-full 或当前 phase 最低 context，随后 full、close、drain
  按 phase/index 服务。1--7 slot 只能因 pressure/close/drain flush。
- `PBR4`：movable ingress lane 优先于 context drain；blocked lane 选择 matching-full 或该 phase 最低 context；
  ingress 空后所有 partial context 按 phase/index tail drain；显式 bundle-epoch retire 后下一 bundle 才可 accept。
- OSG/PBR4 的 slice5 只在 context 无剩余 slot 时产生 `release_pending`；`CONTEXT_RELEASE` 是下一独立收费
  cycle，禁止 same-edge retire-and-replace。SC8/ISO8 group done 与下一 lock、bundle retire 与下一 accept 也
  禁止同 edge。

四点都复用唯一 bank-round/service 定义：round 首 slot 冻结 Cin tile，每 bank 取最低 slot；`GROUP_LOCK`
收费一 cycle；六个 slice 以 issue1/L4/O8/1RW 明确 issue/retire；每 destination 每 round 恰好六 read + 六
write。四份 minimal cycle golden schedule 已内嵌且有 no-newline SHA，分别为：

- SC8 `69f86a715ea5c2644aaa30136e3105ac6f91d27b325dd7a7eae42ee736aec152`；
- ISO8 `89d7a3ee74d6a9b599bd1ecac47481796674a09194245fd6bbae1bdb7abb73ee`；
- OSG `88b397ce590ba252fa21b2ee6fe5f3a47aa3a3a40f86be460a7a5671713119dd`；
- PBR4 `f8bbfb3c638bae1e3163ad541217601759bfe44046278c9ac6cdac85aa8cebdc`。

## 3. M549 P2 closure

future schema 的 required exact-key set 同时包含 `result_path_absent` 与 `attempt_marker_absent`，并为三道未来
独立审查固定 canonical path 与实际身份字段：contract hammer、runner static hammer、final launch release
各自的 `review.md/review.json` SHA、`SHA256SUMS` SHA 与 outer-seal-file SHA。未来 hash 现在未知，因此当前
launch 保持 false；最终 release 必须写入实际值，runner preflight 必须重新 hash、验 member/outer seal 并逐值
比较。仅手写 `score=100` 或路径/JSON 字段不得授权。

## 4. 保持冻结的边界

- exact T10：`92,688,000 bit/sample`、`926,880,000 bit/S10`；raw M511 为 `696,240,000 bit/S10`；
- typed sign：numeric `1` 是合法 `+1`，独立 `source_sign_bit=0`，bit 1 malformed，product sign 只来自
  signed INT8 weight；
- 四点 exact set 固定为 `A1-SC8/A1-ISO8/A1-OSG/PBR4`，runner 无 architecture/transition option；三个
  A1 全 S10x4xT10 完成且双封后，以固定 tie order 选一个 A1-STRONG，禁止 per-sample/layer/time oracle；
- modeled logical total 仍为 `239,636 B <= 240 KiB`，foundry/CACTI/mapped-PPA 均为 false；
- S10 仍只是 single-sequence pre-RTL fast-kill，不是 full-network/system/headline。

## 5. 下一门

当前 `run_authorized=false`。fresh independent static hammer 只有 P0/P1=`0/0` 才可 source-only PASS；即使
PASS，也只能另行审 author runner source，不能直接执行 CPU 或开始 RTL。
