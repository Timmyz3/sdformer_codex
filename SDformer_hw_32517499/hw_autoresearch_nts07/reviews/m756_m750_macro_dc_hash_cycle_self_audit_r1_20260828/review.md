# M756 对 M750 macro-DC 身份链的自审

结论：`NO_GO`。M750 runner 要求未来 release 内嵌 final-release review 的
payload SHA，而 final review 又必须绑定 release SHA。release 字节依赖 review SHA、review
字节依赖 release SHA，形成不可按顺序构造的哈希环。

旧 M750 runner、source contract、candidate 和 request 均保持原字节，不修改、不执行、
不补写 release，也不得在未来解释为已放行身份。结果路径和 attempt sentinel 在本次静态
审计时均不存在，未运行 DC 或其他 EDA。

唯一合法修复是 additive M756/r2：runner 固定 final-review 路径，但不内嵌未来 review
SHA；最终 one-shot 命令用独立环境变量 pin review payload SHA，runner 启动时核对路径、
payload、双封和 review 对 release SHA 的反向绑定。这样依赖关系是
`source -> source hammer -> release -> final hammer -> caller SHA pin`，不再成环。

本审计不承认功能、时序、面积、PPA、能量或系统性能结论。
