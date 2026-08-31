# M719 / M533 r9 monitor-local repair 作者交接

日期：2026-08-28  
状态：**SOURCE_ONLY_COMPLETE；FRESH_STATIC_HAMMER_REQUIRED；NO_RUN_AUTHORIZED**

## 结论

已从 consumed r8 runner 建立唯一 r9 source identity：

```text
dc_handoff/scripts/run_vcs_m719_m533_m528_dead_write_only_1rw_r9_exact_sha.sh
SHA256 = 27f2d7c0f6a2a8569b16f161fe5fcadc0722dfdb0735ee36130c3fb29b964604
```

没有运行 runner、VCS、simv 或任何 EDA；没有创建 candidate、`launch_now=true` release、attempt marker 或 result。

## 唯一修复

r8：

```bash
local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0 tmp="${heartbeat}.tmp.$$"
```

r9：

```bash
local output=$1 violation=$2 heartbeat=$3 request=$4 ack=$5 seq=0
local tmp="${heartbeat}.tmp.$$"
```

同时把唯一 attempt identity 移到从未使用的新路径：

```text
results/m719_m533_m528_dead_write_only_1rw_vcs_r9_20260828
```

r8 双封失败包与 M717 fresh hammer 作为 r9 硬前置，在普通 preflight 前和 atomic mkdir 前各验证一次，并绑定进未来 terminal receipt。r9 不删除、不续跑、不覆盖 r8。

## 静态自测

`static_selftest.json` 已通过：

- wrong-old-runner negative：旧 r8 SHA 被新 contract identity 拒绝；
- old same-local isolated reproducer：RC `127`，`heartbeat: unbound variable`；
- new split-local isolated reproducer：RC `0`；
- `bash -n`：PASS；
- 新 result path / attempt marker：均不存在；
- candidate/release/future review path：均不存在；
- r8 失败包和 M717 member manifest / outer seal：PASS；
- VCS compile 开始至 terminal tail 与 r8 byte-exact；
- monitor function 将两行重新归一化为旧声明后与 r8 exact；
- RTL/TB/SVA/macro/binding plan SHA 全部保持 r8 身份。

自测 SHA：

```text
static_selftest.py   60e47558e5455258bcf77683248d84965d47e9f39e9b06473da2884c5a8dc3df
static_selftest.json ba90bfe8396a64d6b0ff7c558e169c9649245691908ad609f4447bd9d68119ad
```

## 冻结功能身份

| Member | SHA256 |
|---|---|
| top r2 | `726039dbfc1fe611de7beee7d0854028f4163e36b814329251a2e77b7fa790e1` |
| SVA r2 | `b9f66febb5578e3c5a792dee42d87edb0ec68a71845b096a4f47c8c7cdde2c7b` |
| TB r4 | `72a6cef71b0014111c176e6baa751e6d0bfa1ea20e50b5c39b4064bbbe8345ff` |
| macro adapter | `8fd008a321a7167f407025b6c5bebe29155860b464d3846203b81e43f458d783` |
| binding plan | `db4075cb9d34323dcc8c9bb04e575104acb9cb97a819b7f0750ce4a2d3976983` |

## Contract

```text
contracts/m719_m533_m528_dead_write_only_1rw_source_only_contract_r1_20260828.json
SHA256 = fca6edc169aaa4d932bdbe506b3452e49f156e20b9d9c9939a30b9665bf76185
```

contract member sidecar 与 outer sidecar 已验证。contract 明确所有 runner/VCS/simv/EDA/GPU/训练/远程运行预算为 0。

## 下一门

只允许 fresh reviewer 产出：

```text
reviews/m719_m533_r9_source_static_hammer_r1_20260828/review.json
```

必须 100/100、P0/P1/P2=`0/0/0`，并独立复验 wrong-old-runner negative、old127/new0、result/attempt absence、r8/M717 prerequisites 和 frozen functional SHA。该 static review 也不得运行 runner/VCS/simv。

在 fresh static PASS 之后，candidate、candidate hammer、release、final hammer 仍需按顺序另行创建。当前没有任何 launch authorization。

`docs/359` 未修改，SHA 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

