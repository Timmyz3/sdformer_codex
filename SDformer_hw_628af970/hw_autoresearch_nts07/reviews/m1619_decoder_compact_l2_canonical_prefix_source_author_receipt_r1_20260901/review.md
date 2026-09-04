# M1619｜decoder compact L2 canonical-prefix source 作者收据

状态：**SOURCE-ONLY PASS；必须由不同作者复核，当前不授权真实 L2。**

M1619 严格沿 M1615 的唯一授权新增三件东西：L2 canonical-prefix source、fail-closed contract、双 Python static tests。没有改 M1610 或 docs/359，没有打开 ep34 payload，也没有运行 L2/L3、pilot、production、EDA 或 GPU。

prefix 冻结为 D0/call0/timestep0 的连续 row-major destination `0..41`。42 是覆盖 `(y mod 2,x mod 2)` 四种组合的最小 row-major 长度（destination 0、1、40、41），同时包含 corner、edge 和 interior；每配置 4 output block，共 168 commit。cache、24-entry port calendar、129-slot outstanding projection、numeric dependency、计数与 digest 必须在 42 个 destination 间持续，禁止逐 destination reset 或跳过中间历史。

未来 miter 接口逐请求核对 issue/return/dependency/port-ready/beats/count/width/address/bank/packed event，逐 destination 核对 cumulative cycle/count/bytes/commit/address、cache、calendar、outstanding、numeric dependencies 与 RSS。payload-free dense geometry proof 在前两个连续 destination 中得到 1,152 hit、1,152 miss、1,143 eviction，cache tick 从 768 延续到 2,304；这只是静态覆盖证明，不是实际 ep34 周期或流量。

CPython 3.10.16 与 3.6.8 均编译和测试通过：`destinations=42 commits=168 cache_history=1 attacks=4 actual_payload=0 L2exec=0 L3=0`。下一步仅允许不同作者复核 exact source/test/contract；M1619 本身不授权任何真实 payload 访问或执行。
