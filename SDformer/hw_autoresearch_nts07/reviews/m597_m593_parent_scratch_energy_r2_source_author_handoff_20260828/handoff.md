# M597｜M593 parent-scratch energy r2 repair-source author handoff

## 状态

r2 analyzer 与 source contract 已按 M595 的 `1/1/1` 缺陷修复，当前仍是 **source-only**。只执行了 Python 3.6.8 原生编译与不读取业务输入的 built-in static self-test；没有运行正式 analyzer，没有 result/attempt/launch，没有 EDA、GPU 或远程动作。

作者不能自评。下一步必须由不同 agent 做 M598 fresh independent static hammer；只有其 `P0=P1=0`，root 才能另立 exact runner 链。

## 三项修复

1. **traffic 配对**：all-write 的 cycles/read/forward/write 全部来自 sealed M504 result + hammer。宏读为 `16,490,761`，RAW forward 为 `1,714,628`，不能把 `18,205,389` parent edges 全算宏读。dead-only 宏读同为 `16,490,761`；forward 不收费。
2. **strict identity**：CLI 不再接受业务输入 path 或 expected SHA。analyzer 内建所有 path/SHA/manifest/outer seal，并嵌入 r2 contract SHA `90399b6c...`；本 handoff 双层 seal 反向绑定 analyzer SHA `6896c8a4...`，避免 SHA 循环。
3. **provenance**：future rows 保留 cycle/traffic source、read/forward/write、parent/active、8-bank/144-B multiplier 与 conservation。

能量单位已从 `per frame` 收窄为 `per frozen sampled inference`。`38.2283079189%` 与 `1.2622562287 mJ/sampled inference` 仍只是 M595/author self-test 的诊断参考，**不是 admitted result 或 paper data**。

## 精确身份

- contract SHA：`90399b6c932e28f6eac38f3408af0374b23beb369e1fd4e57e3b98d92d28b1bf`
- analyzer SHA：`6896c8a406dc3274926e6c7d958136aca47b9df9afa3522d6c2539a142ea9cf9`
- docs/359 SHA：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`
