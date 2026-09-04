# M1141CA production schedule-release source author receipt

结论：**PASS；只允许不同作者继续做 source hammer。** 本回执不授权打开真实
M410、不授权 production schedule execution、真实 schedule/result、digest compiler、
driver、full replay 或 EDA。

新增 source 是零参数、硬编码 frozen M410 path/SHA/466,560,000-byte identity 的
release builder。它不导入 M1016 runtime，而是从每个冻结 raw tile 独立复算三轴
preprocess/work，并独立执行 M1016 task recurrence。目标流为 812,160 tasks × 3
固定轴序 = 2,436,480 records；状态仅为 O(axes) 加一个有界 64-row tile，不保留
record/key history。

输入由单个 `O_NOFOLLOW` file descriptor 读取，打开前后核对 regular-file、size、
inode 和整文件 SHA。输出逐条写入 0700 private staging，封存 record count、records
SHA、schedule provenance SHA、每轴计数、轴序及 authority identity 后，才以
`renameat2(RENAME_NOREPLACE)` 原子发布。任何中断只允许形成双封
`failed_or_incomplete.*.quarantine`，且 automatic retry 为 false。

受控 fake canonical 的 2 tasks × 3 axes 共 6 条记录得到独立 recurrence
`[0,0,0,34,42,42]`，records/provenance SHA 与逐条独立复算一致。238 checks 和
13 类攻击覆盖缺失、重复、乱序、provenance 漂移、短文件、坏行、错误 SHA、
symlink、打开后路径替换、流中断及结果碰撞；所有失败均未发布 result。

本轮真实 M410 open 次数为 0，production records 为 0，生产 namespace 前后为空，
`docs/359` 未修改。
