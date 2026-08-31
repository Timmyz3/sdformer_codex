# M1176｜M1174 checkpoint-parametric unified capture 独立源码打铁

## 裁决

**52/100，FAIL CLOSED。M1174 r1 不得生成远端 release contract，不得启动 GPU。**

M1175 对 ep29 binder 的独立结果打铁本身有效，M1174 也已具备 source-only、单一 `build_model` 调用点、40-sample 统一入口和 SIGSTOP watcher 检测的正确骨架；作者控制测试 7/7 通过。但生产准入链、固定 cohort、逐层覆盖和结果封存仍有可复现的 fail-open，必须收成 r2 后重新由不同作者打铁。

## 阻塞项

1. **作者封存已坏。** `SHA256SUMS` 记录的 `author_receipt.json` 为 `4379cdc6...`，当前文件为 `a01e18b8...`；收据绑定 contract `ba5d549e...`，当前 contract 为 `ed4177e3...`。
2. **M1175 未被生产验证器语义绑定。** 源码只验证合同指定目录的 seal 和某个 member SHA，不解析 M1175 的固定 schema/status，也不钉住已准入 outer `2a448149...` 与 M1171 result 的关系。
3. **M1174 自身 hammer 未被消费。** `validate_launch_contract` 没有读取或验证 fresh M1174 source-hammer seal；任意 JSON 只要写 magic schema/status 就可绕过这一门。
4. **公共 GPU lease 可被合同重定向。** `main` 使用合同给出的 lease path，而不是源码冻结的公共 `gpu_profile_lease.lock`，因此并发启动可选择另一把锁。
5. **固定 cohort 没有真正冻结。** 变异测试用同一个文件/SHA 重复 40 次、任意 C1 sequence/sample key、任意 decoder cohort label，仍被 `selected_samples` 接受。
6. **逐层覆盖被 category-set 弱化。** 仅保留四层 C1 中一层和 decoder 中一层，再各放一个其余 category，`attach` 仍通过。缺层/漏层不会 fail closed。
7. **attention 完整度未关门。** 发布前没有 `40 x 12` Q/K/gate 记录数、sample-block 笛卡尔积及 payload 存在性检查；0 条或部分 attention bit trace 仍可写 terminal。
8. **嵌套 seal 自相矛盾。** 生产 writer 会封 `payloads/...` 与 `attention_qk/...`，但本源码 verifier 禁止成员名含目录；独立变异证明自己的 writer 输出无法被自己的 verifier 验证。

## 最小 r2 修复单

- 重做 r2 source/contract/test/author receipt，并对最终实物重新双封存。
- 固定解析 M1175 review schema/status、outer/review SHA 与 M1171 result binding；固定消费 fresh M1174 r2 hammer seal。
- 公共 lease 路径写死，合同只能重复声明且必须相等。
- 将 40 个 source 的有序 path/size/SHA/key/cohort/sequence 与冻结 authority 逐项相等，拒绝重复 SHA。
- 静态和运行时都要求四个 C1 target、四个 ConvTranspose、ATLIF=105、attention=12，并冻结 FC1/FC2/patch/BN/QKV inventory。
- terminal 前要求 attention `40 x 12` 完整矩阵和每条 Q/K/gate payload。
- 统一支持嵌套相对路径的 seal writer/verifier，并补 partial/tamper mutation。

## 边界

本打铁只读 M1174/M1175 证据并运行本地静态/单元变异。未访问远端，未启动 GPU、EDA、capture 或 production namespace。`docs/359` SHA 仍为 `dedde7ce...`。
