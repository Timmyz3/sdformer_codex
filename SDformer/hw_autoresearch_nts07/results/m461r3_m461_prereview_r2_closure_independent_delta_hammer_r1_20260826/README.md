# M461 prereview R2 closure independent delta hammer

## 结论

**100/100，P0=0、P1=0、P2=0。** R2 精确关闭上一轮独立评审的 `P1=1/P2=2`，且没有改变优先级或 claim boundary。

这个 GO 只允许在既有最终 M453b gate 之后编写未来 M461 exact-SHA contract 与 executable per-phase event model；仍然不授权 RTL，不授权 cycle/system speedup、energy、PPA 或 DATE headline。

## Delta 重算与攻击

- 48-bit layout：`12+16+7+5+1+1+6=48`，无 gap/overlap；双 3000-row bank 为 36,000 B logical，64-bit row sensitivity 为 48,000 B，padding 12,000 B（+33.33%）。
- descriptor：`valid=0`、reserved 非零、row=3000、original=0、distance mismatch、use_pwp mismatch 全部被拒；fallback 不访问 remap/PWP。
- sentinel：full-3000 使用地址 0..2999，pointer=3000 控制合成 end，不损失一行；empty count=0 不发 SRAM 请求；sentinel 不进 backend。
- remap：独立检查 1,030 个 bitmap，包括 empty、边界 center、all128 与 1,024 个确定随机向量。injective、contiguous、inverse、8-block valid 均成立；duplicate slot、inverse mismatch、missing block、unknown/invalid lookup 均 sticky fail-closed。
- lifecycle：逐一攻击 9 个 role-switch 前置条件；assignment seal、epoch/tag、generator/pending-write、old replay/downstream drain 任一未满足都不能切换。
- percentile：标准 nearest-rank index/value 是 `1641/1584`；legacy floor index/value 是 `1640/1576`，两者不再混称。
- B config：per-tile address footprint 27,552 B、two-tile 55,104 B；shared physical lower bound `156,896 B`，replicated physical lower bound `157,472 B`，口径已分离。

## 优先级与权限

- compact used-center + original-order：唯一首选，等 sealed `Nmax`。
- A：NO-GO。
- B：backup interface/event screen only。
- C：unknown。
- true group replay：NO-GO primary。
- fold：只作 exact `prep_done` 后的 measured DSE axis。

审阅只打开了 R2 closure 与上一轮独立 review 的双封目录。没有接受或读取 M40、M453b、docs/359、RTL、catalog 或 runtime payload。
