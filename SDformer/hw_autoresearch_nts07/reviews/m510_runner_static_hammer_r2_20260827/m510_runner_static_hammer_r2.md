# M510 exact runner r2 静态终审

日期：2026-08-27  
结论：`STATIC_GO__EXACT_ENV_PINNED_ONE_SHOT_ONLY`  
评分：**97/100**  
P0：**0**  
P1：**1**  
runner/audit 实际执行：**否**

## 结论

r1 的 cwd-sensitive `${BASH_SOURCE[0]}` P0 已修复。r2 runner SHA 为：

```text
0d05e6e668e6446575a6477ef7e151f9a6c459c275073ab0a789048915da8020
```

可授权一次 one-shot audit，但 caller 必须以上述字面值设置
`M510_EXPECTED_RUNNER_SHA256`；禁止临时用 `sha256sum` 命令替换该字面值。

## 关键复核

1. **self path 已去除 cwd 歧义。** runner 在 `cd` 前执行
   `readlink -f ${BASH_SOURCE[0]}`，并校验它等于硬件根下唯一预期路径。
   从 SDformer 根以 `hw_autoresearch_nts07/...` 相对路径、从硬件根以
   `system_simulator/...` 相对路径，或以绝对路径启动，都得到同一
   canonical runner/root。
2. **reviewed runner SHA 是 attempt 前门。** 环境变量缺失、为空或与当前
   runner SHA 不同，都在 attempt 之前退出。本评审只授权字面值
   `0d05e6...`。
3. **全部外部身份在 attempt 前预计算。** analyzer、contract、docs510、
   docs359 和 r2 static review seal 先生成在 `mktemp`，立即 `sha256sum -c`；
   此时还没有创建永久 attempt。
4. **single-owner 安全。** 预检后以固定目录的 atomic `mkdir` 竞选唯一
   owner。失败者不能进入 audit；获胜者把已验证的 identity 移入 attempt。
5. **cwd 和 identity.sha256 一致。** self 条目是绝对路径；其他条目是相对
   hardware root 路径。预检和最终复检的 cwd 都固定在 hardware root。
6. **失败不可冒充 PASS。** analyzer 失败、output seal 失败、冻结输入漂移、
   review seal 失败或 identity 复检失败都会因 `set -euo pipefail` 在
   `POSTAUDIT_PASS` 前退出。失败后只留“已消耗但未 PASS”的 attempt。
7. **最终输出与输入双复核。** output 成员 seal/outer seal 通过后，
   runner 再次校验 analyzer/contract/docs/review，再校验 `identity.sha256`；
   `POSTAUDIT_PASS` 记录 output outer-seal-file SHA 并进入最终 attempt seal。

## 唯一 P1

最终 top-level receipt 绑定了 `SHA256SUMS.initial.seal.sha256` 文件，但最终
阶段没有再跑一次它所声明的 nested check。初始阶段已检查过，且本合同
是单 owner，因此不阻塞 one-shot。通用封存 helper 后续可在写 PASS 前再检一次。

## 唯一授权启动方式

```bash
M510_EXPECTED_RUNNER_SHA256=0d05e6e668e6446575a6477ef7e151f9a6c459c275073ab0a789048915da8020 \
  /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/system_simulator/scripts/run_m510_h67_convtranspose_coverage_gap_audit_r2_exact_sha.sh
```

启动前必须再确认 canonical output 与 attempt 均不存在。本评审不授权直接
运行 analyzer，也不授权 RTL。
