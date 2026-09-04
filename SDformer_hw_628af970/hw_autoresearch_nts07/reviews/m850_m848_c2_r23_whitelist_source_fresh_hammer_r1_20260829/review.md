# M850 — M848/C2 R23 whitelist source fresh hammer

## 裁决

**88/100，FAIL：P0=1，P1=0，P2=0。当前 M848 身份不得进入 true-release author，更不得运行 VCS。**

R23 的核心修复方向正确：它不再封存整个 VCS work tree，而是用 `O_NOFOLLOW` 复制 15 个精确常规文件；独立测试确认工具 symlink 和额外文件不会进入 stage，白名单 symlink 会被拒绝，M803 RTL/SVA/TB/filelist 与 R22 的命令和周期门也全部冻结。

但 runner 在 stage 双封以后，仍把带 `attack/...`、`equalbw/...` 的递归成员交给继承自 M826 的 `verify-sealed-directory --exact-root-members`。该旧接口不是递归 exact-population API：它在集合相等之后还强制所有成员名不得含 `/`。因此正式 runner 会在 `RESULT_STAGE_SEAL` 确定性失败，并永久消费这次 attempt。

## P0 独立复现

我构造了完整 15 文件 work，依次调用：

1. `stage_result_whitelist(work, stage)`；
2. `base.seal_directory(stage)`；
3. `base.verify_sealed_directory(stage, whitelist + seals)`。

前两步通过，第三步稳定报错：

```text
flat root contains nested member
```

根因对应 runner 第 311–321 行与 M826 guard 第 164–168 行。作者现有 5 个测试全过，但只覆盖到 whitelist staging，没有覆盖 `stage -> seal -> recursive exact verify -> no-replace publish -> post-publish verify`，所以未发现该错误。

## 已通过的边界

- `stage_result_whitelist` 对每级目录和源文件使用 `O_NOFOLLOW`，要求常规文件；目标使用 `O_EXCL`，并核对源 dev/inode/size/SHA 的前后稳定性。
- stage 的 15 个文件严格等于白名单，0 symlink、0 extra；VCS 私有 work 中的工具 symlink 可以安全留在白名单之外。
- M803 三个 RTL、SVA、两个 TB、两个 VCS filelist 与 DC filelist SHA 都与合同一致。
- M837 与 M848 的 attack/equal-bandwidth 命令和 PASS gate 字节完全一致，片段 SHA 为 `261d47f0...`；五组周期仍为 `51/53, 131/133, 486/499, 1231/1246, 14/14`。
- M846 双封失败分类与 M837 已消费 attempt 均校验通过，R23 没有复用或追补旧 work。
- 独立 source dry-run 返回 86，在 live license/VCS 边界前停止；license、VCS、simv、attempt、result、quarantine 计数均为 0。
- 正式 M848 result/attempt/stage 当前均不存在；`docs/359` SHA 保持 `dedde7ce...`。

## 最小修复

不要改冻结 RTL、SVA、TB、filelist、命令、seed 或周期门，也不要放松 `O_NOFOLLOW`。只需在 R23 guard 增加递归 exact-population 的专用 verify/publish API，或以不触发旧 flat-root 断言的方式校验 sealed recursive population；随后增加完整出版路径的单元测试，刷新受影响 SHA，并重新做独立 source hammer。当前身份不能写 release。

本审阅没有查询许可证，没有运行 VCS/simv/DC/任何 EDA，也没有创建正式 attempt/result。
