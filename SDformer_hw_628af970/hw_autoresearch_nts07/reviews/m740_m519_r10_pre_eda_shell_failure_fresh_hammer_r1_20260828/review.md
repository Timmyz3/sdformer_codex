# M740｜M519 R10 pre-EDA shell failure fresh hammer

裁决：`PASS_FAILURE_AUDIT__R10_BLOCKED__PRE_EDA_JQ_ESCAPE__ADDITIVE_R11_REQUIRED`。

## 1. 直接根因

R10 runner 的第 246 行把反斜杠写进了单引号包围的 `jq` 程序：

```text
jq -e '.verdict == "PASS" and .score_out_of_100 == 100 \
       and .severity_counts == {"p0":0,"p1":0,"p2":0}' ...
```

在 shell 单引号内，行末 `\` 不是 shell 续行符，而是传给 `jq` 的字面字符
`0x5c`。对冻结的 M576 `review.json` 重放该程序稳定返回 `jq_rc=3`，错误为
`unexpected INVALID_CHARACTER`；只删除该字面反斜杠、保持布尔条件不变时返回 0。

静态搜索在 runner 中只发现这一处“`jq` 单引号程序内以反斜杠结尾”的模式。因此它是本次
已观察退出的唯一、充分且最早可达的直接根因。这个结论不等价于证明第 249 行之后未执行的
路径不存在其他潜伏问题；successor 仍必须重新走完整静态评审和一次性发布链。

## 2. 零 EDA 与身份状态

- 双封存失败回执记录 `exit_code=3`、`attempt_consumed=false`、
  `PRE_ATTEMPT_SHELL_FAILURE__NO_EDA_RESULT_ADMITTED`。
- 失败点是第 246--248 行；K1 preflight 从第 707 行开始，attempt sentinel 到第 823 行才
  原子发布，第一次 `dc_shell` 调用在第 1171 行。
- 运行目录根下与 R10 相关的唯一对象是
  `m519_r10_pre_attempt_shell_failure.693765.receipt/`。R10 canonical、attempt sentinel、work、
  preflight staging 和 quarantine 均不存在。

所以本次没有启动 DC/VCS/Formality/PT/PTPX，没有产生面积、时序、功耗或性能证据；R10
EDA attempt identity 未消费。R10 runner/contract/admission 已冻结且本次失败不可引用。

## 3. 身份与封存

- runner SHA256：`7dc7d79c27b85820c621ac142e104cc155afd949ffa8b2ec46dd7279a314d27f`
- recovery contract SHA256：`2ba563ed4c3ddb2c89d0a13855bb4b11be7522aef505cfe1ef374a33b5501a4e`
- launch admission SHA256：`f4bccc501dea216396d2755ef6b1f627209efe18346701cd5d448367cf4a3424`
- failure receipt `SHA256SUMS`/outer-seal 文件 SHA256：
  `34e3049ebb6dc29dba5daf7ca8102ef27e82f7b4debfa821014e326ace52d97e` /
  `da19a42e10b299c22d700ac41c31842a883d33be141c883aad0e05799b9139d0`
- R10 contract、admission、M704、M708、M694、M701、M576 与失败回执的双封存均校验通过。
- `docs/359` SHA256 仍为
  `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

## 4. 唯一合法 successor

创建 additive M519 R11；不得原地修改 R10，也不得放宽任何资源、碰撞、工具身份、输入身份、
三轴公平性或 claim-boundary 门。R11 的最小要求是：

1. 新 runner、contract、admission、canonical、attempt sentinel 和 pre-attempt receipt identity；
2. 唯一语义修复是删除该 `jq` 单引号程序中的字面反斜杠，保留 M576 的
   `PASS/100/P0=P1=P2=0` 精确断言；
3. 把本 R10 失败回执的 payload、manifest、outer seal 和 M740 结论作为精确 provenance；
4. 增加覆盖完整 admission/contract `jq` 路径、并在第一次 preflight/attempt/tool 调用前退出的
   no-EDA admission-path 自测；仅 `bash -n` 与现有早退 self-test 不足以发现本缺陷；
5. 经作者静态交接、fresh independent hammer、fresh launch admission 三段封存后，才可发布
   一条精确 pin runner+admission SHA 的 `env -i` 命令；最多一次 DC-only 三轴 attempt；
6. 继续禁止 VCS/Formality/PT/PTPX/remote，结果仍须标成 logic-only、pre-macro、
   `paper_ppa_ready=false`、`system_speedup=false`、`headline=false`，直到后续独立结果审计。

当前授权：`NO_LAUNCH_FROM_M740`。M740 只封失败审计，不授权 R10 重跑或 R11 EDA。
