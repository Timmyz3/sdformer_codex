# M1341｜M1340 Table-A common-charge compiler independent blind hammer

## Verdict

`FAIL_DO_NOT_CITE__ADDITIVE_SUCCESSOR_REQUIRED`

作者双封与 `docs/359` 身份均通过，作者 10/10 测试复跑通过。独立 hammer 共做 30 项变异，24 项预期边界正常关闭，但发现 6 个可接受攻击，其中 5 个是 P0。

最关键的 P0 是：`PRODUCTION_CANDIDATE` 只是 config 里的自声明字符串。盲审器在临时目录自行创建假的 checkpoint/config/profile/capture/result-hammer、direct/common charge、95% coverage、17 SRAM 计数和 energy rate，把文件 chmod 0444 并填写它们自己的 SHA，M1340 随即输出：

`PASS_PRODUCTION_CANDIDATE_UNHAMMERED`

这证明当前代码只验证了“自造 JSON 此后没变”，没有验证“JSON 来自已准入的真实权威”。因此 M1340 不能生产或喂给 Table-A。

## 正常关闭的边界

- 漏 common 类别、漏 row、漏 direct branch、common row 化均拒绝。
- row numerator 直接不等、漏 DRAM 字段、少一个 SRAM、energy 少一个宏、coverage <95% 均拒绝。
- 非 3×10、权重不归一、缺 density strata、resource lane/port/queue 漂移均拒绝。
- identity/SHA 漂移、可写 leaf、硬链、直接 symlink、duplicate key、NaN 均拒绝。
- CLI 确实使用 O_EXCL + 0444；第二次写同一路径失败。O_EXCL 源码替换也被独立静态 invariant 捕获。

## 六个 false negative

1. **P0｜自造权威可冒充 production。** checkpoint/capture/hammer 只当 opaque bytes；charge/energy 只核自述 schema/identity/coverage，没有 allowlisted review/manifest/outer、producer/tool 身份或 admitted status。
2. **P0｜fixed numerator 可按人口点错位抵消。** 六行各在不同 sample 增加 1，weighted total 相等，compiler 通过；同一个 sample 的 numerator 实际不同。
3. **P0｜common energy 被 row-specific rate 不公平定价。** B0 rate 设 1e6、Ours 设 1e-9 后，同一 common work 被不同费率乘，伪造出 `0.9999999999999992` energy reduction。
4. **P0｜人口/strata/weights 未绑定冻结 manifest。** 三个虚构 sequence、重标 strata、首样本权重 0.5 均可通过，只要总和为 1 并同步自造 charge keys。
5. **P1｜父目录 symlink 可穿过。** leaf 本身是 regular file 时，workspace 内 symlink parent 不会被拒绝。
6. **P0｜零内存流量可自报为完整 accounting。** 所有 17 宏字段存在但 SRAM/DRAM 全为 0，仍得到 production-candidate PASS；没有绑定 address-timed replay conservation receipt。

## 最小 additive successor

1. Production 路径必须 allowlist 并双封绑定 final checkpoint/config/profile、canonical capture、fresh result hammer、每个 charge producer 与 energy producer；检查 review/manifest/outer、source/interpreter/tool identity 和 admitted status。仅 read-only+SHA 不足。
2. 在加权前逐 `population_key` 比较六行 fixed numerator，再复核 aggregate。
3. common 与 direct energy 拆开：common 使用唯一 row-invariant admitted rate；row-specific rate 只作用于明确的 row-dependent block，并绑定 native PTPX/macro authority。
4. 人口、sequence、sample、strata、weight 全部来自冻结 sealed manifest，config 不能自行替换。
5. 从 workspace root 到 leaf 的每一级祖先都用 lstat 拒 symlink；config 与 output parent 同样检查。
6. SRAM/DRAM 数量绑定 address-timed simulator 收据并做 conservation；不能用“字段齐全”代替真实 transaction authority。
7. 输出写入完整 input authority identities 与 config digest；fresh different-author bundle hammer 前仍保持 non-admitted。

没有修改 M1340 source，没有创建 Table-A，没有运行 GPU/VCS/EDA；`docs/359` 保持 `dedde7ce...`。
