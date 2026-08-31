# M817 request｜M815 decoder runner recovery fresh source hammer

Receipt-blind 审阅 M815 additive repair。只允许源码、临时夹具和双封审阅；不得创建 true release、不得调用 formal runner、不得消费正式 attempt、不得生成生产周期，也不得调用 VCS/EDA/license/GPU/remote。

决定性攻击是：在 flat attempt 已成功 no-clobber 发布后、fallible postcheck 尚未完成时注入失败。必须看到 attempt 已消费、0 schedule row、canonical result absent，并得到精确四成员双封 failure quarantine。Runner 源码顺序必须是 publish → started → explicit phase → postcheck → full consumed preflight → production。

M815 只能作为 additive identity，M811 的 NO-GO 和 release=false 必须保持原义。薄 driver 只能委托精确 SHA 的冻结 M809 schedule body，不得改变 40+120、T10、资源、D1 或 headline 分母。

唯一 PASS 是 `PASS100_M815_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY`，且仍不能直接生产。
