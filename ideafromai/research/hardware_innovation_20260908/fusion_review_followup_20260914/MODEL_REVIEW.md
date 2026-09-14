本机检查结果没有发现Luna。`/home/zhumd/.codex/config.toml`为`gpt-6-astra`、`model_reasoning_effort=xhigh`，没有agents模型覆盖；当前主线程最近上下文以及上一批三个代理最近可读上下文同样为`gpt-6-astra/xhigh`。检查只读当前线程有界尾部与本地线程关系，没有恢复/注入禁止使用的旧554MB线程。

这只能说明本地配置及记录请求了什么，**不能证明服务器实际后端一定没有降级或界面误标**。当前没有服务端路由记录，因此不能给用户编造“为什么显示Luna”的原因，也没有凭空声称已切换后端。没有必要把已正确的配置反复改写。

本轮两名独立审阅代理显式指定`gpt-6-astra`与`xhigh`，不依赖默认模型；父代理继续执行代码、质量复算与新RTL。另一个代理启动因线程数量限制失败，由父代理接手，并未降为其他模型。官方文档说明子代理可以继承父设置，也允许覆盖；这与本地检查结果相符，但不能代替后端诊断。[OpenAI子代理配置说明](https://learn.chatgpt.com/docs/agent-configuration/subagents)
