# M461 prereview independent hammer

## 结论

独立评分 **95/100，P0=0、P1=1、P2=2**。只给未来 M461 exact-SHA contract 与逐 phase event model **CONDITIONAL GO**；不授权 RTL，也不授权 cycle/system speedup、energy、PPA 或 DATE headline。

唯一首选仍是 `compact used-center bank + original-order replay`，并且必须等最终 M453b 的 sealed `Nmax`。A 全 q128 cache 因容量故事 NO-GO；B q32 parent + 单 child scratch 只保留为 backup interface/event screen；C 在 ordered selected-ID ledger 前保持 unknown；true group replay 不是主线。fold 只能在逐 phase 的 exact `prep_done` 后发生，必须把 replay 变短导致的 prep overrun 重新联立，不能转移 M451 的 1.202 opportunity。

## 独立重算

- M461 subject outer seal SHA `f7f8bbaac92440e8f90c920854a51f08dc85cb3b2e0632c8a0b634a2de9ddf02`，双封验证通过。
- M453a catalog outer seal SHA `e154a4a1667458732fac1bbb2416d8966e9569d00401ba8bc66c7701cc695de8`，双封验证通过。
- 独立遍历 1728 partitions、165,888 parent-child edges；Hamming hist 完全一致：`{1:138391,2:16746,3:9451,4:460,5:679,6:83,7:74,8:1,9:3}`。
- 每 partition 96 child flips 为 `98 / 119.515046 / 119 / 136 / 160`；q32 zero-rooted Prim MST 为 `38 / 58.848958 / 58 / 70 / 84`，全部对上。
- all128×8 weight-update-only reference 的 min/mean/median/max 为 `1160 / 1426.912037 / 1424 / 1736`。唯一统计差异是 p95：标准 nearest-rank 是 `1584`，subject 的 `1576` 实际是 `sorted[floor(0.95*(n-1))]`，不改变 admission，但下份 contract 必须固定口径。
- bytes 独立对上：144 B logical/block、160 B padded/block、1,280 B/center/eight blocks、163,840 B q128 PWP/phase、176,704 B two expanded tile slots、327,680 B two-phase PWP、36,000/48,000 B 双 48/64-bit assignment bank、primary fixed subtotal 61,564 B、B lower bound 156,896 B。
- compact bank 公式正确：physical `2560*Nmax`，logical signed12 `2304*Nmax`；`Nmax` 必须是所有 sealed runtime phase 中 `pwp_rows>0` distinct-center count 的最大值，平均数、population、set/count_runs 都不能替代。

## 攻击结果

全部通过：used-center set/count_runs 没有冒充 ordered stream；group replay 没有冒充免费；M451 1.202、PAFT 数字没有迁移；q128 容量没有消失；compact/B/C 优先级自洽；fold 只在 prep_done 后逐 phase 联立。

## 剩余问题

- **P1**：primary compact path 计了 3000×48-bit assignment bank，但尚未冻结 48-bit 字段布局、valid/sentinel、fallback 与 remap lookup timing。未来 event-model contract 打开 ordered ledger 前必须冻结，且保留 64-bit macro padding sensitivity。
- **P2**：all128 p95 字段命名和算法不一致，见上。
- **P2**：B 的 per-tile footprint 把 288 B config 复制进两个 tile，integrated lower bound 又按每 phase 一份共享 config 计。可以成立，但下一表必须明确“兼容地址 footprint”和“物理共享 storage”的差异。

R1 reviewer 因本机 Python 不支持 `int.bit_count` fail-closed；R2 暴露 p95 定义差异和三个字面匹配过严的 attack predicate，再次 fail-closed；R3 只做 Python-3.6 popcount 兼容与语义 attack 检查恢复，没有改 subject、阈值或结论。

本审阅没有接收 M40 路径，没有读取/运行 M453b，没有修改 `docs/359_DATE终局冻结_20260813.md`。其 SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
