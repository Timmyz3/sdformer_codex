# M769 请求：M768 decoder address-timed source fresh hammer

请对 M768 做一次独立、只读、fail-closed 的 source hammer。先完整阅读 M766 的 `review.md`/`review.json`，再检查 M768 contract、analyzer 和 15 项测试。

本请求只允许验证 source package；不允许跑 M686/M699 production replay，不允许生成 decoder cycles/speedup，不允许 EDA/GPU/remote。通过也只能说明最小地址时序执行语义可进入下一份增量 launch release，不能形成 Table-A 或 full-network 行。

必须独立攻击：输入双封、243,200 B 分配加 2,560 B 保留的容量 cliff、bank conflict、1RW/1R1W、响应同拍释放、dependency/issue/return/commit 时间戳、commit/resource/fallback 三重公平、D1 统一收费 fallback、人口混合和 K8 headline comparator。输出必须双封。
