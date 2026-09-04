# M930｜M925/M896 decoder GTLS R2 source-only 独立打铁

## 裁决

**PASS 100/100，P0/P1/P2 = 0/0/0。** M925 的 R2 修复确实把 M900 的后台 shell-function PID 错认修成了 `setsid --wait` 私有进程组中的实际 Python worker，并把正常、超时、资源门和 shell signal 路径都约束为“整组 TERM，有限 grace 后整组 KILL，回收 root，证明组空，最后才允许 rename/seal”。

本审阅不直接授权 full-row。唯一下一步是由不同 author 写一份绑定精确 M925 与本 M930 身份的 inert M927 release；之后还必须有 fresh independent M928 final-launch hammer。只有该链闭合，主线程才可最多运行一次 no-argument full-first-row exact/scalability diagnostic。

## 独立重放

- 冻结 M896 测试 **11/11 PASS**，包括 synthetic 1K/10K、真实 D0/A1/t0 1K/10K/100K；没有运行 9,582,057 条 compressed transaction 的 full row。
- M925 driver/M896/tests 用内存 `compile()` 通过，runner `bash -n` 通过；未向被审源码写入 bytecode。
- exact-pin `--dry-run-no-work` 通过。缺 pin、malformed/wrong runner pin、wrong contract pin、非法参数，以及 future M927 缺失下的 no-argument 路径均在 attempt 前被拒绝；前后 M925 result/attempt/stage/log/failure namespace 均为空。
- target `strict_json` 拒绝 duplicate key、NaN、`+Infinity`、`-Infinity`；`renameat2(RENAME_NOREPLACE)` 碰撞不覆盖 source 或 destination。
- 私有进程组攻击全部通过：正常退出；整组 TERM；root 与 descendant 都忽略 TERM 后整组 KILL；两成员 RSS 聚合。四条路径都回收 root 并证明非 zombie process group 为空。
- M902、M900 consumed attempt、M900 failure receipt 和 M926 request 的 manifest/outer seal 全部重算；M900 封存未被恢复、复用、删除、修改或别名化。

## 口径红线

9.320783571 秒仍是 **M900 已经失败的 100× 科学假设**，R2 禁止重试或改名。2715 秒只是由 bounded real-100K 推导出的 host operational safety timeout，不是 accelerator cycle、speedup 或论文指标。

本收据只证明 source/process-control closure；它不证明 full-row、production、decoder-complete、可引用周期、系统加速、能量或 PPA，也没有调用 VCS、EDA/license、GPU、remote 或 network。
