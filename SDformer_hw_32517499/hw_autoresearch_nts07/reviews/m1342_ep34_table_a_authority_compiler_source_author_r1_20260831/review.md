# M1342｜ep34 Table-A authority compiler source authoring

## Verdict

`PASS_SOURCE_AUTHOR__FRESH_DIFFERENT_AUTHOR_BLIND_HAMMER_REQUIRED__NO_PRODUCTION_ALLOWLIST`

M1342 是 additive successor；M1340 源码保持 `9cbf2262...` 未修改。作者复跑 M1340 10/10，并完成 M1342 16/16，总计 26/26 source tests PASS。

## M1341 六个漏洞的关闭

1. Production 只接受代码内固定的五类 authority allowlist，caller 不能传入。当前真实 capture/energy/transaction authorities 尚不存在，因此 production allowlist 精确保持空，任何 `PRODUCTION_CANDIDATE` 在读自造 payload 前就失败。未来必须用 additive release 写入五套 exact root/review/manifest/outer/producer/tool SHA。
2. fixed numerator 在加权前按每个 population key 比较 B0/B1/B2/B3/C2/Ours 六行，错位抵消被拒绝。
3. common energy 只使用一个 row-invariant logic rate并只算一次；C1/decoder/attention direct energy 按 branch/row 分列。冻结 M1340 compatibility energy 若含 row-specific common rate 会失败。
4. sequence、sample、density stratum、weight 必须逐项等于 sealed population manifest。
5. workspace root 到 leaf 的每一级 path component 均用 `lstat` 拒 symlink；config/output ancestry 同门。
6. 每行、每 sample 的 cycles/numerator/17 SRAM/DRAM 必须等于 sealed transaction receipt；系统内存流量全零被拒绝。

## Authority 与输出身份

每个 authority 验证 outer 文件 SHA、outer 内容、manifest SHA、递归人口、review member SHA、role/identity/status/claim semantics，以及 code-pinned producer/tool SHA。输出绑定 config/base-config、M1340/M1342 source、五套 authority、population manifest 与 address trace digest。

## 诚实边界

当前 source milestone 没有真实 production allowlist，因此不能产生 production candidate，更没有 Table-A、cycles、energy 或 speedup。没有读取 capture，没有运行 GPU/VCS/EDA。不同作者 blind hammer 通过后，也只证明 source gate；真实 authorities 到齐后仍需 additive release 与 fresh bundle hammer。
