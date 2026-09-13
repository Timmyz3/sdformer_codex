# 网页 Pro 协作：本机检查与可行连接

2026-09-13。用户提供 [共享对话](https://chatgpt.com/share/6aa6557c-f654-83ea-b433-eda456d5c9c4)，标题为“硬件创意调研”。web fetch 超时后，普通公开 HTTP 请求返回200；只解析返回HTML中的可见 user/assistant 文本，取得四份主要调研。没有登录、提取 Cookie、读取凭据或向网页版提交消息；共享链接只用于读取现有材料。

四份主要调研的新增内容、旧身份纠正与下一阶段纳入项见 [共享对话逐项审阅](SHARED_DISCUSSION_REVIEW.md)。

## 已实际确认

- 本会话工具清单没有可操作用户浏览器的 browser/computer/Playwright 工具。
- ismd 有 chromium-browser、Firefox 可执行程序；有 Firefox 进程，但本 shell 无 DISPLAY，常用本机调试端口没有可连接浏览器。
- 没有检查或读取 Firefox 的 Cookie、密码库或会话内容；Firefox 存在不代表已经连接到用户的 Pro。
- 当前环境没有 `OPENAI_API_KEY`。只检查变量是否存在，没有查找/打印凭据。
- 因此，本轮**尚未自动向网页版 Pro 发问，也没有调用付费 API**。

## 可行路径

| 路径 | 能做什么 | 当前缺什么 |
|---|---|---|
| 已登录浏览器＋浏览器控制 | 在网页选 Pro、提交明确研究问题、读取答案，再由本地实现/验证 | 用户实际浏览器所在机器及可连接的浏览器工具；服务器的shell不自动继承另一台电脑浏览器 |
| 官方 Responses API 的 Pro 推理模式 | 脚本化提交紧凑证据包、长推理审阅、回收结构化意见 | API凭据和计费配置；不是调用用户正在看的那条网页会话 |
| 共享对话/文件往返 | 立即读用户已有Pro研究，生成下一次询问的完整问题包 | 可立即做；新的Pro回复由用户提供分享或文件后回读 |

官方文档说明桌面应用的浏览器扩展可使用已登录网站；设置入口为 Settings→Computer Use，安装对应插件/扩展后在聊天里选择浏览器。这个能力需要当前应用/会话实际连接，不能因产品文档说支持就宣称本远程CLI已经拥有它。[官方浏览器扩展文档](https://learn.chatgpt.com/docs/chrome-extension)

官方 Responses API 支持适用模型的 `reasoning.mode="pro"`，按模型实际用量计费；用户的网页Pro会员和本地API凭据不是同一个连接条件。确切可用模型与账号权限应在接入时核实，不把网页Pro能力和任意模型ID强行等同。[官方Pro推理模式](https://developers.openai.com/api/docs/guides/reasoning#reasoning-mode)、[认证/计费边界](https://learn.chatgpt.com/docs/auth)

本轮已在对话中询问已登录浏览器位于Windows/Mac/ismd哪台机器，尚未假定答案。接入前可以先使用 [下一次Pro询问包](NEXT_PRO_REVIEW.md)；这份包已带真实RTL负结果、强先验与明确问题，不会让Pro继续基于旧96×32或“非折权连续发放”推演。
