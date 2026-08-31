# M1149R6 C2 preflight-redaction repair source author receipt

结论：**PASS；只授权不同作者做最终 hammer。** 本回执不授权真实 `lmstat`、
launch、attempt、VCS、case0 或 DC。

M1149R6 仅对 M1146R6 做 additive preflight repair，并精确绑定 M1146、M1147、
M1148 的双封身份。它保持 `SNPSLMD_LICENSE_FILE` 优先、
`LM_LICENSE_FILE` fallback，不向 child environment 写入 `HOME`，复用仍 fresh 的
M1146 namespace，未来仍只有一次 compile、一次 128-cycle case0、零次 DC、零重试。

修复后的 `lmstat` helper 只依据进程 return code 作判定。stdout/stderr 仅在内存中
瞬时接收，既不返回也不写入日志、JSON、异常、manifest 或 seal；timeout 安全返回
false，Popen 异常统一改写为不带链的固定错误。公开 preflight 只保留 route 的变量名、
存在性、字节长度和 SHA-256，不保存 route 原文。

controlled mock 覆盖了 stdout/stderr 同时回显完整 secret、非零 rc、timeout、包含
secret 的 Popen 异常，以及完整 success/compile-failure 流。成功流严格截获一次
`lmstat`、一次 compile、一次 case0；失败流生成一个双封 quarantine、消耗 attempt
并拒绝重试。两条流的所有 sealed members 均未出现 secret。

共 227 checks、4 attacks；真实 `lmstat`/VCS/DC/launch 调用均为 0，真实 M1146
attempt/result namespace 保持 fresh，`docs/359` 身份不变。
