# M510 exact runner r1 静态终审

日期：2026-08-27  
结论：`NO_GO__RELATIVE_INVOCATION_CAN_CONSUME_ATTEMPT_BEFORE_IDENTITY_FAILURE`  
评分：**76/100**  
P0：**1**  
P1：**2**  
runner/audit 实际执行：**否**

## P0｜`${BASH_SOURCE[0]}` 在 `cd` 后变成 cwd-sensitive

runner SHA `e929848e...` 先计算：

```text
m510_root = dirname(BASH_SOURCE[0])/../..
cd m510_root
```

但在 atomic attempt 已经创建之后，`identity.sha256` 仍直接使用未规范化的
`${BASH_SOURCE[0]}`。从 SDformer 根目录以常见相对路径启动：

```text
bash hw_autoresearch_nts07/system_simulator/scripts/run_m510_...sh
```

子进程 `cd` 到 `hw_autoresearch_nts07` 后，原字符串会被解析为：

```text
hw_autoresearch_nts07/hw_autoresearch_nts07/system_simulator/scripts/...
```

该路径不存在。失败顺序是：

1. 所有冻结输入预检通过；
2. `mkdir` 原子创建永久 attempt lock；
3. 写入 `ATTEMPT_CONSUMED.txt`；
4. `sha256sum ${BASH_SOURCE[0]}` 失败，`set -e` 退出；
5. 生产 audit 未开始，却不得重跑。

这是 one-shot 可用性与身份链 P0，不能靠操作人“记得在特定 cwd 运行”
规避。

必修：在 `cd` 之前将 self path 规范成绝对路径，验证其预期位置与被评审
SHA，之后 `identity.sha256` 只使用该绝对路径。修改后必须另起 runner 版本和
静态重审。

## 已通过项

- analyzer `117384e...`、contract `4bda9f...`、docs510 `940621...`、
  docs359 `dedde7...`、r2 review 内外 seal 都有外部硬编码绑定。
- `mkdir attempt` 本身是原子 single-owner 锁，并且失败时保留消耗记录。
- analyzer 的相对 contract/output 路径在切换到硬件根后解析正确。
- 任一 analyzer、output seal 或输入复核失败都会在
  `POSTAUDIT_PASS` 之前退出，不会冒充 PASS。
- 成功路径会复核 output 成员/outer seal，再复核全部冻结输入和
  `identity.sha256`，最终 PASS 回执绑定 output seal-file SHA。

## P1

1. runner 在创建 attempt 前没有将 self 校验为被评审的 `e929848e...`；
   只在消耗 attempt 后记录并做 start/end 不变检查。
2. 最终封存会绑定 `SHA256SUMS.initial.seal.sha256` 文件，但最终阶段
   没有再执行一次该 nested seal 的 `sha256sum -c`。在严格 single-owner 下是低风险，
   修订时可一并补上。

## 裁决

P0 非零，所以 **不准 one-shot**。必须 supersede runner r1，在新 SHA 上重审；
禁止通过手工选择“恰好可用”的 cwd 绕过。
