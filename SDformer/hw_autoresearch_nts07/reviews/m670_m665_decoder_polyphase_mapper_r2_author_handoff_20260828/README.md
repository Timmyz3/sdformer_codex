# M670｜M665 decoder polyphase mapper r2 修复作者交接

## 状态

`STATIC_R2_AUTHOR_HANDOFF_ONLY__FRESH_HAMMER_REQUIRED`。

本轮只修复 M667 封存的三个 P1 和一个 P2，不改 M514 polyphase 坐标、phase/tap/K 顺序，也不读取 canonical M660 payload。作者验证不构成独立准入；只有新的 fresh hammer 返回 P0=0、P1=0，r2 才能进入 M660 payload 集成。

## r2 身份

- mapper：`hw_autoresearch_nts07/system_simulator/scripts/map_m670_decoder_convtranspose_polyphase_workload_r2.py`
  - SHA256 `875b31ed1994729cc29321af0053fcea5586077aa468398d31eb4fe0fdb1596b`
- pytest：`hw_autoresearch_nts07/system_simulator/tests/test_m670_decoder_convtranspose_polyphase_workload_r2.py`
  - SHA256 `39c4fd847e04e3d3089e9bea614339a4c5a2ab2c9dd49a68cb66a343d9e8e31e`
- dual-Python smoke：`hw_autoresearch_nts07/system_simulator/tests/m670_decoder_polyphase_r2_dual_python_smoke.py`
  - SHA256 `8c77912aea6d894c953a72273e3e4c03f2f5f79392a4a5b089ed639a8e47d1b0`
- contract：`hw_autoresearch_nts07/contracts/m670_decoder_convtranspose_polyphase_workload_mapper_r2_contract_20260828.json`
  - SHA256 `b4e0c7ab83059cf8466a7c4485180e515b2a15f55aec7d4efb8e4262ec1d1975`

M667 review 外层封文件 SHA256 保持 `8d65d026...`，r1 mapper/tests/contract 保持 `07dd6474...`、`736056ea...`、`52eb24ec...`，`docs/359` 保持 `dedde7ce...`。

## 四项定向修复

1. shape 元素数改用 Python 大整数逐维累乘；单维、总元素、空间 plan 与 K 均有显式上限，越界前即拒绝。源码不再使用 `np.prod`。
2. manifest 与 payload 必须给出显式绝对 trusted package root。词法 containment、逐 component `lstat`、父/叶 symlink 拒绝和 resolve 后 containment 四层同时成立；manifest 必须是 package root 的直接成员。
3. record 绑定完整 S10×4 lattice：D0/D2/D3 只能位于 `d0_d2_d3_binary_records`，模块为 0/2/3、name 精确、route 为 `EXACT_BINARY_BITPACK`、身份取 `row.input`；D1 只能位于 `d1_records`、模块 1、name 精确，只有 `EXACT_SCALED_BINARY_BITPACK` 取 `theta_binary_candidate`，fallback 只验证而不 bitpack。交叉、重复或缺失均拒绝。
4. shape、phase/bank、tile_m、output_channels、spec、sample/module、packed_bytes 等整数入口统一拒绝 `bool` 与 `np.bool_`。

## 作者验证

- PyTorch Python：`32 passed in 1.06s`。
- Python 3.10.18 / NumPy 2.1.2：`PASS_M670_R2_DUAL_PYTHON_STATIC_SMOKE`。
- Python 3.6.8 / NumPy 1.19.5：`PASS_M670_R2_DUAL_PYTHON_STATIC_SMOKE`。

Pytest 继续逐元素对齐 PyTorch `ConvTranspose2d`；dual smoke 不导入 Torch，在两套 Python 下覆盖 bounded product、bool、little-bit、polyphase、trusted-root/symlink/traversal、完整 lattice 与交叉 route/缺失攻击。

## Claim boundary

没有运行 GPU、VCS、DC、PTPX 或其他 EDA，没有产生 cycle/speedup/energy/PPA，也没有修改 r1、M667 或 `docs/359`。本包只是 r2 作者候选，不能自评、不能授权 M660 集成。
