# M649｜H67 decoder ConvTranspose 输入 typed numeric audit 作者交接

## 当前裁决

`STATIC_AUTHOR_HANDOFF_ONLY__FRESH_HAMMER_REQUIRED_BEFORE_GPU`。

本轮只新增独立 M649 数值审计源码、合同和 CPU 定向测试；没有运行 GPU/model/checkpoint forward，没有生成 M649 result/staging，没有再次运行或重置 M511，也没有运行 EDA。M511 冻结 producer/contract/runner/verifier 均未修改，已消费 one-shot 仍保持消费状态，失败 staging 的 `FAILED.json` 与 d0 partial bitpack 均原样保留。

## 关键结构纠正

冻结源码给出的 decoder 拼接次序不是“前 C−2 binary、末 2 flow”：

- `Spiking_STSwinNet.py`：`skip_ftn(predictions[-1], x, dim=2)`；
- `model_util.py`：`skip_concat` 返回 `torch.cat([x1, x2], dim=dim)`；
- 所以 d1–d3 的源码预期布局是 **前 2 通道预测 flow、后 C−2 通道 binary**。

M649 不依赖这个推断直接给 GO。它会对全部 10×4 hook 同时测两套假设：

1. source-ordered：first2 finite/observably-nonbinary flow + suffix C−2 exact binary；
2. pre-audit hypothesis：prefix C−2 exact binary + last2 finite/observably-nonbinary flow。

第二套完整保留在结果中，但只作 diagnostic；typed-split admission 默认绑定冻结源码顺序。若实测不支持 source-ordered 布局则 `NO_GO_EXACT_TYPED_SPLIT`，不得阈值化、交换通道或把模拟 flow 强转 bitpack。

## 审计内容

每个 sample/module 记录：

- input shape、dtype、device、contiguity、stride；
- 全 tensor 与每 channel 的 exact zero/one/binary/integer/nonfinite counts；
- d0 全通道 exact `{0,1}` gate；
- d1–d3 first2/suffix 与 prefix/last2 两套 partition gate；
- 两套候选 flow 通道各自的 finite min/max、zero/one/integer、NaN/+Inf/−Inf，以及 finite-only float64 sum/abs-sum/square-sum/mean/mean-abs/RMS；
- 精确 40-record、float32、S10/module order、checkpoint exact load、BN policy、CUDA fence 与 raw source identity gate。

安全 aggregate 只把两个候选模拟通道有界搬到 CPU float64；不保存原始 activation，不复制/序列化全 tensor，JSON 禁止 NaN/Inf token。

## 新文件身份

- launcher：`neuron_experiments/H9_bipolar_self_attention/entrypoints/audit_m649_h67_convtranspose_typed_numeric_inputs.py`
  - SHA256 `b5a4f71a5eb12bd21825a71fa1a80cd12d732711b52866c303aad7b674d3e74f`
- contract：`hw_autoresearch_nts07/contracts/m649_h67_ep35_convtranspose_typed_numeric_audit_contract_r1_20260828.json`
  - SHA256 `651dddb3afa7a794070a1527474b9f8660ded8f12b91338efc41805554436340`
- tests：`hw_autoresearch_nts07/system_simulator/tests/test_m649_h67_convtranspose_typed_numeric_audit.py`
  - SHA256 `5b3c8bc3b73a185ae00b6bcbff332fab317a6ed7e18c2c60436bc30e3267ceda`

冻结根：M511 producer `e16a454d...`、M511 contract `e556743d...`、checkpoint `4f33e086...`、config `8be3f7bb...`、`docs/359` `dedde7ce...`。M649 还显式绑定 `Spiking_STSwinNet.py` 与 `model_util.py`，以封住 typed channel order。

## Fail-closed 事务

- 在 import M511 producer 之前先核验 M649 全部输入；import 后再次核验 producer SHA。
- 所有 CLI、contract、M511 失败状态、output/quarantine 路径都在 `resolve` 之前检查原始路径链；拒绝任一 symlink、dangling leaf 与 `..` traversal，再做 canonical equality。
- 开始与结束都重验 M649/M511 输入、checkpoint/config、raw DSEC source、失败 one-shot receipt/seal/staging population 和 `docs/359`。
- M649 只写新的 `results/m649_h67_ep35_convtranspose_typed_numeric_audit_s10_r1_20260828`，通过同父目录 staging、double seal、atomic rename 发布；post-publish 验证失败会把 M649 canonical 移入唯一 quarantine。
- M511 canonical 必须保持 absent；M511 consumed attempt 与失败 staging 仅只读，绝不删除、rename、覆盖或作为新 bitpack payload 复用。

## 静态验证

系统 Python 3.6 `py_compile` 通过；固定 PyTorch Python 运行 14/14 CPU-only tests 通过。测试覆盖 exact inputs/failed state、d0 binary、first2 flow/suffix binary、错误 last2 假设、suffix 污染、nonfinite、dtype/population fail-close、double-seal tamper、dangling canonical output、runtime input symlink alias 与 parent traversal。

## 运行边界

GPU 命令目前**不授权**。只有 fresh independent hammer 返回 P0=0、P1=0 并明确 GO 后，才能使用 contract 中唯一候选命令。GPU audit 即使得到 typed-split GO，也只授权新 mixed-type decoder capture/simulator contract；不授权 cycle、speedup、RTL、VCS、Synopsys、energy、PPA、system speedup 或 DATE headline。
