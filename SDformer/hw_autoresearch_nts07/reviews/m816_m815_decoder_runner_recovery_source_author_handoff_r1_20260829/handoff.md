# M816｜M815 decoder runner recovery source author handoff

M815 是对 M811 P1 的 additive 修复，不修改或伪造 M809/M811。新的薄 boundary driver 只负责 M815 candidate/release/attempt/failure 身份；周期生成仍委托 SHA 为 `2b273d...6736d0` 的冻结 M809 schedule body。

关键顺序已经变成：attempt no-clobber publish 成功 → `started=1` → `ATTEMPT_PUBLISHED_POSTCHECK` → fallible postcheck → full consumed-attempt preflight → production。动态注入 post-publish postcheck failure 时，scheduled rows 为 0、canonical result 不存在、attempt 已消费，并生成精确四成员双封 failure quarantine；重复 destination 攻击不改写已有证据。

Python 3.10 和 Python 3.6 均为 10/10 source-only tests 通过。Python 3.6 运行使用系统内纯 Python dataclasses backport；没有调用 EDA、license、GPU 或 remote。没有 true release、formal attempt/result/failure 或 production cycle。

下一步只允许 M817 receipt-blind fresh source hammer；PASS100 后仍只能另行创建 true release，不能直接生产。
