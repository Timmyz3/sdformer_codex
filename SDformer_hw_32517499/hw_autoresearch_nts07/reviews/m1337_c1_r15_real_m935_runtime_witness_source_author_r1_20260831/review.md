# M1337｜C1 R15 real-M935 runtime-witness source authoring

## Verdict

`PASS_SOURCE_AUTHORING__FRESH_DIFFERENT_AUTHOR_BLIND_HAMMER_REQUIRED`

R15 是 additive verification-only successor。M528、M935、M1162、R3 SVA、R13 TB、R14 source/contract 与 214,912 B ledger 均未修改；未运行 VCS、simv 或 EDA。

## 六个 false-negative 的关闭方式

1. 第二次 weight request 只能在周期入口已注册的 `W_FIRST_ACCEPT` 阶段发生，不能与第一次 core accept 同拍。
2. 第二 accept、psum commit、row completion、task completion 分别要求前一注册阶段；它们不能通过 `*_after` 组合值在一拍内串级。唯一允许同拍的两组是 first weight+psum request 与 response+core accept。
3. event control 与 source/address/row/epoch identity 均有显式 `$isunknown` sticky fault；identity 比较使用 `===`。
4. checker 先移除注释，再解析 active bind 的全部 named connections；真实 child output、attack mask、design/service fault 任何一项绑常量均失败，注释残留不能救活。
5. 唯一 PASS 在 `if (pass === 1'b1)` 成功分支内；operand dump 在分支前，唯一 fatal 在 else。把 PASS 提前或移出成功分支会失败。
6. checker 解析完整 `frozen_design`，强制整数等式 `214912 = 18432 + 196480`；三个 ledger 字段任一改为 1 均失败。

## Source verification

- 20/20 directed Python tests PASS。
- exact seven-member filelist：foundry unit-delay model、M528、M935、M1162、R3 SVA、R13 TB、R15 witness。
- M1335 R14 FAIL 双封被精确绑定：review `31abaa97...`、manifest `b918e1c8...`、outer `05c76b26...`。
- `docs/359` 保持 `dedde7ce...`。

## Claim boundary

这只是 source author PASS。它不证明 wrapper functional VCS、3 ns timing、cycles、speedup、PPA、power、energy 或 headline，也不创建 release。下一步只能由不同作者做 fresh blind source hammer。
