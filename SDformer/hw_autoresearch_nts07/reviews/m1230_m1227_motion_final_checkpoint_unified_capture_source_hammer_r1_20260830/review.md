# M1230 — M1227 最终 checkpoint 参数化统一采集器独立 hammer

## 裁决

**FAIL（P0），不允许 release authoring。** M1227 的采集与失败取证主合同大部分
是正确的，但最终 checkpoint/config 配对接口与 M1228 的真实输出结构不兼容，
并可被一个双封的混合身份对象绕过。双 seal 只能证明对象没有被修改，不能证明
该对象表达的是同一个被选中候选。

本 hammer 没有修改 M1227、没有访问远端、没有读真实 checkpoint、没有运行
valid825/GPU/capture/EDA，也没有发布 release 或消耗生产 attempt。

## 已经通过的部分

- M1224 review/manifest/outer 三重身份绑定正确；dead 12 是 `.sn_v` ATLIF，
  不是 attention parent。
- 静态 inventory 精确 259 模块/105 ATLIF；每样本 live inventory 精确
  247 模块/93 ATLIF，live 每个一次，12 个 dead `.sn_v` 精确零次。
- 40 样本总门是 9,880 ordered records；attention 独立笛卡尔积 480；
  retained payload 文件人口 640。
- missing/duplicate live call、dead call、wrong category、unexpected sample、
  attention 缺失/重复及 payload extra 都被拒绝。
- per-sample snapshot 使用临时目录、成员 fsync、目录 fsync、rename、父目录
  fsync；状态明确 `FORENSIC_ONLY__NOT_CANONICAL`。注入 rename 前失败后没有
  partial final directory。
- 三个 M1227 namespace 当前互异且均不存在；import 不加载 torch、numpy 或
  M1174 substrate。作者 15 项测试与独立复验均通过。

## P0 反例

M1228 的真实 selection 把 checkpoint 和 configuration 都放在 `selected` 中。
M1227 [capture source](/home/zhumd/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention/entrypoints/capture_m1227_motion_final_checkpoint_unified_hardware_r1.py:639)
却读取顶层 `selection["configuration"]`：

1. 对 exact M1228 形状，验证器直接 `KeyError`，所以新最终结果无法进入 release。
2. 若在同一个递归双封 selection 中补入冲突顶层 config-B，同时
   `selected.configuration` 保留 config-A，验证器会通过，并返回
   selected checkpoint + config-B。即 checkpoint/config 配对被静默拼接。
3. `selection_schema` 由未来 launch JSON 自报，selection status 未校验，也没有
   强制最终 selection 的独立 result hammer。反例本身的 status 明确写着
   `HARDWARE_REBIND_NOT_AUTHORIZED`，补顶层 config 后仍被接受。

## 最小修复

新建 additive successor，不覆盖 M1227：

- 对 M1228 schema 只从 `selected.checkpoint` 与 `selected.configuration` 读取，
  并拒绝顶层 `configuration`；
- 在 source/contract 固定 selection schema 与独立 result-hammer authority，
  不接受 launch JSON 自报 schema；
- 严格检查 selected candidate/epoch/profile/checkpoint/config 的 key、类型、
  SHA/size/mtime，并让 result hammer 明确授权该 pair；
- 保持已经通过的 live/dead、atomic snapshot 与 9880/480/640 门不变。

修复后必须新 hammer。当前只允许 repair source authoring；不允许 release
authoring，更不允许生产运行。
