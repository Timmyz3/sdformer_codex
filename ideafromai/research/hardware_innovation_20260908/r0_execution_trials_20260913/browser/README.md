# 在 ismd 上登录 Pro

2026-09-13，已实际启动独立 Chromium 133、TigerVNC 和 noVNC；本机 noVNC 页面HTTP200，Chrome调试连接可读取该浏览器页面。初次“Just a moment...”等待页随后正常加载为ChatGPT首页，已检查截图并点击可见Log in按钮。**尚未登录，尚未向Pro提交研究问题。** 账号登录由用户在可见浏览器完成。

## 用户连接

在用户电脑运行（主机名使用平时连接这台服务器的地址）：

```bash
ssh -N -L 6088:127.0.0.1:6088 zhumd@ic.ismd-nemo
```

在用户电脑浏览器打开 `http://127.0.0.1:6088/vnc.html` 并连接。查看器密码在服务器：

```bash
cat ~/.local/state/codex-pro-browser/viewer-password.txt
```

此密码仅用于VNC查看器。ChatGPT密码和验证码直接在远程浏览器输入，不用发到聊天。登录后确认网页模型可以选Pro，再告知本地执行agent。遇到网站验证时由用户处理，不更换指纹或绕过访问限制。

## 实际本机连接

- 专用桌面`:88`；VNC `127.0.0.1:5988`；noVNC `127.0.0.1:6088`；Chrome调试 `127.0.0.1:9222`。
- 持久浏览器目录 `~/.local/state/codex-pro-browser/chromium-profile/`。进程号、日志、查看器口令和X授权均在同一私有父目录，不进入Git。
- 浏览器使用本机已有HTTP代理`127.0.0.1:7897`。未禁用浏览器沙盒或TLS检查。
- noVNC作者源在`~/.local/share/codex-pro-browser/noVNC`；websockify依赖装入该目录的独立`python/`，只在websockify子进程设置PYTHONPATH，未修改训练Python环境。
- `start_browser.py`只启动这套专用进程。只读页面检查：`/opt/anaconda3/bin/python3.12 page_status.py`。

这次建立的是实际本机可见浏览器和本地CDP控制通道，不是声称远程CLI已经装有ChatGPT桌面扩展。后续交互仅通过网页界面；不提取或复用Cookie调用私有后端API。登录与模型确认前，不自动发送准备好的[Pro评审问题包](../../pro_fusion_trials_20260913/pro_assist/NEXT_PRO_REVIEW.md)。
