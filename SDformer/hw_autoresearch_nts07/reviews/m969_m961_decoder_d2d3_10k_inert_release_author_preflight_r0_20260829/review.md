# M969 author inert-release preflight（非独立 hammer）

M969 release 只允许未来消费一次 attempt，并执行 D2/D3 各一条 10K exact scheduler prefix。Release 本身 `launch_now=false`；runner 还要求 M970 review/manifest/outer 三重身份，因此 M970 缺失时仍不可运行。

100K 仅可由未来 10K 结果推荐，不能自动执行；full-row 永久不由本 release 授权。本目录不是 M970，不能替代独立 release hammer。
