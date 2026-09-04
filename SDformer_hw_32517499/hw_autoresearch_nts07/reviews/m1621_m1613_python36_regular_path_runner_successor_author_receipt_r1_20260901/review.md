# M1621｜M1613 Python 3.6 regular-path runner 窄修作者收据

裁决：**SOURCE-ONLY PASS；必须先做 M1622 独立 hammer，当前没有 VCS 或 attempt 权限。**

M1613 的实际 launch 在创建 attempt 前以 exit code 3 停止：`/usr/bin/python3.6` 是指向 `/usr/libexec/platform-python3.6` 的 symlink，而 runner 的 `expect_sha` 明确拒绝 symlink。两个路径解析后的文件 SHA 相同，均为 `9c9502e...`。因此 M1621 只把 runner 的 Python 3.6 authority 改成后者这个普通可执行文件。

除该路径以及必须更新的 M1621/M1622/M1623 控制平面身份外，新旧 runner 归一化后逐字节一致。M1609 RTL、M1613 TB/filelist、seed 1613、VCS 命令、PASS token、result/attempt namespace、一次 compile + 一次 simv、无重试与原子发布规则均未改变。

CPython 3.6 和 3.12 均通过同一静态测试：旧 runner 精确复现 `/usr/bin/python3.6` 的 pre-attempt failure；新 runner 通过 regular Python gate 后，精确停在尚不存在的 M1622 sealed-review gate。两次检查前后 M1613 result 与 attempt 都不存在，VCS/simv 次数均为 0。

下一步只能由不同作者创建并运行 M1622 source hammer。仅当 M1622 PASS 后，M1623 才能另行封一份单次 release。当前 M1621 不授权 VCS、simv 或任何 EDA。
