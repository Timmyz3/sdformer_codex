# M724｜M714-r2 repaired one-shot runner receipt-blind static hammer

## 裁决

**FAIL，72/100；P0/P1/P2 = 2/0/1。** 不得运行当前 runner，不得创建 attempt/result，不授权 A800 或远程 capture。

本轮只读 exact-SHA runner、contract、capture、M366、M716 与 M720 证据；没有运行 runner，没有 import/执行作者 capture，没有查询 `nvidia-smi`/GPU，没有调用 EDA 或远程，也没有修改作者文件。

## P0-1｜contract 与 capture 语义不兼容，会先烧掉 one-shot attempt

当前 contract 的 `milestone` 是 `M714-r2-repair-after-M720`，而 exact capture 硬要求 `M714-r2`。同时 contract 的 identity 有六个 key，新增 `m720_failed_static_review`；capture 仍要求精确的五 key 集合，会拒绝该额外 key。

runner 在 idle 前只核对 contract/capture SHA，不调用 contract 语义 validator。控制流是：验 static review → 检查 result/attempt 不存在 → 四次 idle → `mkdir ATTEMPT` → 启动 capture。因此这两个不兼容项会在 **attempt 已消耗、M366/CUDA 尚未进入** 时失败，得不到任何合法 M714 结果。

## P0-2｜compute-app 查询对命令失败是 fail-open

`GPU_APPS` 使用：

```bash
nvidia-smi --query-compute-apps=... || true
```

查询失败会被转成空字符串，后续 `-z GPU_APPS` 把它当作“没有 compute app”。若 utilization/memory 查询仍可解析且低于阈值，一个不属于 train/eval/valid/profile 命名族的 GPU 进程可能被漏过。修复必须把 query 退出码非零与“成功查询且列表为空”区分开。

## M720 进程命名修复：所要求正反例已过

独立从 runner 源文本提取 regex 并重放，未扫描真实 `/proc`。

- 应命中的真实命名：`profile100.py`、`valid825.py`、`validate.py`、`trainer.py`、`trainonly.py`、`evaluation.py`、`training.py`、`run_date11_ft5_and_valid825.py`、`run_h67_ep35_profile100_bit_trace.py`，全部 PASS。
- 无关名字：`retraining.py`、`invalid825.py`、`evaluate.py`、`profiler.py`、`trainable.py`、`profiled.py`、`validity.py`、`data_profile100extra.py`、`evaluationReport.py`、`mytrainer.py`，全部不命中。

四次 idle 的排序也成立：四个 sample 在 attempt 前，中间有三次 5 s 间隔。但 P0-2 意味着 GPU-app 查询本身仍不是 fail-closed。

## M716/M720 其余复核

| 项 | 结果 | 静态证据 |
|---|---|---|
| exact-SHA 身份 | PASS | runner/contract/capture 为指定 SHA，M366/M716/M720/docs359 身份一致 |
| contract/capture 语义 | **FAIL P0** | milestone 与 identity key set 各有一处精确不匹配 |
| M366 人口/数值门 | PASS static | 10 samples、105/81/45/36 sites、450 calls、dead-called empty 和四项零数值门都在 PASS 前 |
| one-shot lifecycle | PASS static | idle 在 attempt 前；staging、failure quarantine、seal、publish rename 和 terminal reverify 顺序成立 |
| output miter | PASS as boundary | `real_output_miter=false`，pattern capture 不冒充 accelerator equivalence |
| cycle/config | PASS | Fixed=`17N+12`；build=`+64/call`；direct=`+23 beats/call` |
| resident-45 税 | PASS | P1/P2/P4/P8 = 23/46/92/184 macros，46/92/184/368 KiB |
| pattern 守恒 | PASS | tile/bitplane、histogram、distinct/nonzero、port monotonic、per-site=aggregate |
| chunk boundary | PASS | `column_base%16==0`，只有 final chunk 可 pad |
| algebra selftest | PASS | 明确标记 deterministic randomized smoke，`exhaustive=false` |
| claim boundary | PASS output / **P2 source wording** | JSON 与 contract 都是 ideal-resource lower bound，但源 docstring 仍写 conservative issue schedule |

## 独立算术

- subset table：`2×32×10×11 = 7040 bit = 880 B`。
- M518 Fixed：N=1/4 分别 `29/80 cycles`。
- direct table load：`ceil(7040/256)=28 beats`，相对已含的 5 beats 只增 23。
- resident-45：P1/P2/P4/P8 分别 `23/46/92/184 macros`，容量 `46/92/184/368 KiB`。

## Claim boundary

本 FAIL 不授权 runner、remote/GPU capture、attempt/result、M714 pattern 数字、executable cycle、real-output miter、RTL/VCS、Synopsys PPA/energy、accuracy、system speedup 或 paper headline。`docs/359` 未修改，SHA256 仍为 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
