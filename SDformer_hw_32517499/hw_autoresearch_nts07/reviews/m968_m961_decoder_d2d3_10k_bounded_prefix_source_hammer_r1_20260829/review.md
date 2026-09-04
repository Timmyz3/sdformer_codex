# M968 独立 source hammer

结论：**100/100，P0/P1/P2 = 0/0/0；仅允许 `GO_TO_INERT_M969_RELEASE_DESIGN`。** 本评审不授权执行 prefix、创建 attempt/result、100K、full-row、production、EDA/GPU/remote 或任何论文指标。

## 核验结果

- M961 driver、runner、checker、tests 与 source contract 的 SHA 均与冻结身份一致；M946 source 及 M950 review/manifest/outer 三层身份递归通过。
- 精确解释器固定为 Python 3.10.18、SHA `9f78cd...2115`；默认错误解释器在加载 M946 前 fail closed。
- Python compile、`bash -n`、static checker、6/6 无 prefix tests 全部通过。
- clean-env runner 未提供 M969/M970 身份时退出码为 1，在创建 attempt/result 之前拒绝；测试前后 M969、M970、attempt、result 均不存在。
- runner 只可能在未来独立 M969 + M970 双身份通过后消费一次 attempt，再运行固定的 D2/D3、sample0、A1_OSG、t0、10K pair；发布使用独立 sealed result namespace，失败 stage 进入 quarantine。

## exact 与口径边界

10K exact 路径逐字段比较 M890/M896 的 schedule、地址、commit、cycle classes、terminal readiness 与 port calendars，并在 10K 内经 M890 继续绑定 M768/M861。

D2 首个 source-fetch transaction 是 231,600 requests，D3 是 465,600 requests。因此 10K 与可能的 100K 都仍是 `SOURCE_FETCH_ONLY`：没有触达 contributor mapper、psum commit 或 full row。100K projection 只能建议另一个独立 release，`automatic_100k_authorized=false`。

禁止据此声称 contributor/full-row scalability、decoder complete、Table-A、system speedup 或 paper-citable 性能。

`docs/359` 保持 SHA `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
