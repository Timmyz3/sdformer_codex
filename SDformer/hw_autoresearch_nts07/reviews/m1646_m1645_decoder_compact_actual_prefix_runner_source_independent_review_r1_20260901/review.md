# M1646｜M1645 decoder actual-prefix source 独立评审

日期：2026-09-01

状态：`NO_GO_M1645_ACTUAL_PREFIX_EXECUTION_RELEASE__ONE_P1_PRESENCE_ONLY_AUTHORIZATION_GATE__SUCCESSOR_SOURCE_REPAIR_ONLY`

评分：88/100；P0=0，P1=1，P2=0。M1645 的 synthetic 功能链通过，但不授权 actual-prefix execution/release。

## 已通过的部分

固定人口精确绑定为 D0/call0/module0/timestep0/destination 0..41/output-block 0..3，三配置顺序为 `DENSE_TYPED_K8` / `BIT_EQUAL_SERVICE_K1X8` / `BIT_TYPED_K8`，product-capture 仍禁用。M1539、M1610、M1638 源和 M1639 review/manifest/outer seal 均按 exact SHA 通过。

作者测试在创建 M1646 评审目录前于 CPython 3.6/3.10 各 10/10 PASS，两版 `py_compile` PASS。独立 synthetic hammer 观测到三个不同的 configuration-bound session，每配置 42 次 per-destination miter 与 168 个 commit；per-request miter 调用分别为 2184/6888/2184。request cycle、request coordinate、destination cycle 和 destination configuration 四类差异全部被拒绝。RSS 的 current/HWM、absolute/increment 门均通过边界注入。

全程未打开 payload，未执行 actual prefix，未产生周期/流量结果，未运行 GPU/EDA，未创建 release/attempt。

## P1：private runner 只用路径存在性授权

`_run_bound_actual_prefix()` 只检查 `FUTURE_REVIEW.exists() and FUTURE_RELEASE.exists()`。它没有在 payload 选择/打开前验证两个路径是 regular non-symlink，也没有绑定 exact SHA、内外层 seal、status 和 execution authorization。

CLI 和公开 `actual_prefix_release()` 确实不会到达该函数，但 Python 的下划线 private 命名不是授权边界；两个伪造空路径就能满足当前谓词。评审未实际触发该攻击，因为那会违反本次 no-payload 边界；源码证据已足够确认缺口。

## 裁决

禁止执行旧 M1645，禁止为它直接建 release。只允许新命名 source-only successor：在任何 payload 操作前绑定 M1646 tree 双封、release file 双封、exact identity/status/authorization、fresh result/attempt/work/failure namespace、one-shot 和 RSS 门；然后重过一次不同作者 P0=0/P1=0 评审。

本 review 不准入 payload、actual-prefix、周期、流量、加速、能量、Table-A 或论文结果。
