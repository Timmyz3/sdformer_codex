# M528 r4 preflight receipt hammer（r1）

## 裁决

- 得分：**99/100**
- P0/P1/P2：**0/0/1**
- 状态：`PASS_R4_PREFLIGHT_RECEIPT_HAMMER__AUTHORIZE_ROOT_TO_SIGN_ONE_PRODUCTION_ADMISSION`
- 裁决：`GO_ONE_R4_PRODUCTION_AFTER_NEW_DOUBLE_SEALED_ADMISSION`

本次只读审阅已封存 preflight 结果、attempt、admission、admission-only review、source-only static review 及两条 runner；未执行 analyzer、preflight、spawn、production、EDA、GPU 或 RTL，也未修改被审证据。

## 已闭合的证据

1. 结果目录及其 attempt 均通过成员 manifest 与外层 seal 校验。receipt SHA 为 `bcf4e20a7114c5a190a326eb9bc0c7fea5f197104b3444dee65b377276391eca`，结果外层 seal 文件 SHA 为 `be18a1cd7bc5a66b2b0c93f2f9a323fc778a962840aa023df845086ea14849a3`。
2. 正例 `exit=0`，唯一 PASS token 恰好出现一次，stderr 为空。wrong-pointer 与 wrong-corner 均 `exit=1`、PASS token 为零，stderr 分别含有精确的 pointer/corner mismatch 文本。
3. spawn receipt 与冻结源码一致：稳定模块名、单 worker、实际调用 `worker_init`；`worker_phase` 只做 pickle identity 检查而未调用；未进行 row replay。
4. 所有禁止活动均为 false/0：没有 production process pool、生产结果、生产 attempt、CPU production、EDA、GPU 或 RTL。
5. preflight attempt 已按一次性合同消费，且其中明确记录 `production_attempt_consumed=false`。生产 canonical 与生产 attempt 路径当前均不存在。
6. receipt、admission、static review、live analyzer、preflight runner 和 execution contract 的 SHA 交叉一致；docs/359 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 唯一 P2

负例退出码由精确封存 runner 写入 receipt，没有另存独立的 `.rc` 文件。这不阻塞：runner 在构造 receipt 前直接捕获两个返回码，receipt 均为 1，且各自封存的 stdout/stderr 同时证明零 PASS token 与精确错误文本。后续通用 harness 可加 `.rc` 文件，方便外部审计。

## 授权边界

本评审只允许 root 创建**一份新的、双封存的 production admission**；它不直接授权任何 CPU production run。production admission 还需独立 admission-only review，启动前必须通过至少 48 GiB commit headroom、内存/swap/OOM 与本用户 Synopsys/simv 冲突门。生产结果仍需独立 result hammer 后才可引用或进入 RTL 决策。
