# M559｜PBR4 pre-RTL CPU contract r4 终端 FSM 与单向授权链修复交接

日期：2026-08-28  
状态：`R4_SOURCE_ONLY_TERMINAL_AND_IDENTITY_REPAIR_COMPLETE__RUN_AUTHORIZED_FALSE__FRESH_STATIC_HAMMER_REQUIRED`

本交接只新增 r4 contract、公共 terminal golden、future runner/launcher schema、handoff 与 fresh static hammer
request。没有运行候选 CPU/analyzer，没有 author 或执行 runner/launcher，没有建立 authorization、result 或
attempt marker，没有修改 RTL，没有运行任何 EDA/训练/GPU/远端任务，也没有修改 r3、M556 或 `docs/359`。

## 1. M556-P1-01：公共 terminal FSM 已唯一化

r3 的公共 priority row 8 已被完整替换，不能与 r4 合并。四个架构现在逐字复用同一 terminal FSM：

- 非末 output block：`NONLAST_BLOCK_RETIRE` 与 `NEXT_BLOCK_OWNER_LOAD` 各收费 1 cycle；
- 末 output block：`LAST_BLOCK_RETIRE` 收费后，才进入 directory clear；
- directory clear：`START` 收费 1 cycle，word 0--1023 各收费 1 个 accepted 1RW zero-write cycle，`END`
  再收费 1 cycle；禁止 start+word0、双 word、word1023+end 融合；
- clear 完成后才收费 `TIME_RETIRE`，随后单独收费 next-time owner load；literal T10 的 time9 再唯一进入
  layer/sample/cohort retire 分支。

每个 row 都固定 prior-state guard、action、完整 state delta、charged_cycles 与 primary_class；output block
`<last` 与 `==last`、clear index `<1023` 与 `==1023`、time `<9` 与 `==9` 均互斥。architecture table 不能
覆盖、绕过或添加 terminal edge。

新增两份四架构共用的 no-newline canonical golden：

- nonlast block：2 cycles，SHA256
  `dc68fdfc65716ec084377bb1bda5ed454504fe35f9d0acdbd8f094cc86bab628`；
- last-block/time：1029 cycles，其中 1024 个清目录写，SHA256
  `46526954f88c08a91f082713d0f1248bdec23137fdb372f697601953257fa819`。

精确串与 run-length 展开规则见 `terminal_goldens.json`；四份原 resident-hit golden SHA 保持不变。

## 2. M556-P1-02：后生授权身份环已拆成 DAG

未来身份只允许按以下方向生成：

`r4 contract/static review -> immutable runner/static review -> candidate review -> final-release review ->`
`canonical 双封 authorization -> post-auth launcher wrapper -> 独立 wrapper static/release review -> one shot`

关键约束：

- immutable runner 只冻结 canonical authorization **路径**，不嵌任何后生 authorization SHA；
- authorization 等 final-release review 已存在后才生成，单向绑定 runner 与之前所有 review/input identity，
  自己独立 member-hash + outer-seal，不含 self hash 或 wrapper identity；
- 后生 launcher wrapper 冻结 authorization JSON/member/outer 三个文件 SHA 与 runner SHA，再单独审查；
- wrapper 只冻结 canonical wrapper-review 路径，不嵌后生 review SHA。执行时它重算自身 SHA 并与双封 review
  内的 frozen value 比对；runner 与 wrapper 都重算 authorization 及其绑定的全部 earlier bytes；
- wrapper review 是终局 release，不再生成更晚的 author permit，因此没有反向 hash edge。

直接调用 runner、修改 wrapper、手写 score/launch JSON 或环境覆盖任一 workload/architecture/transition/gate
均在 result/attempt 创建前拒绝。

## 3. 保持不变的边界

- exact T10：block replay `92,688,000 bit/sample`、`926,880,000 bit/S10`；raw M511
  `696,240,000 bit/S10`；
- numeric `1` 是 `+1`，独立 `source_sign_bit=0`，bit1 malformed，product sign 仅来自 signed INT8 weight；
- 四点仍是 `A1-SC8/A1-ISO8/A1-OSG/PBR4`；三个 A1 完整 S10x4xT10 后只选一个固定 A1-STRONG，禁止
  per-sample/layer/time oracle；
- service/resource/GO gates 不变，logical-only 仍为 `239,636 B <= 240 KiB`，foundry/CACTI/mapped-PPA=false；
- 当前仍是 single-sequence pre-RTL fast-kill，不是性能、system speedup 或 paper headline。

## 4. 下一门

当前 `run_authorized=false`。fresh independent static hammer 必须独立复算 terminal goldens、验证 terminal FSM
全分支唯一性，并证明 future identity DAG 可构造；只有 P0/P1=`0/0` 才能 source-only PASS。PASS 也只允许
另开 immutable runner source admission，不授权 CPU 或 RTL。
