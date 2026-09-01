# M1558 reduced-binary producer source 作者收据

状态：**source + synthetic binary roundtrip PASS；仍需独立 source hammer。未授权 GPU、SSH、checkpoint load、capture、release 或 RTL。**

M1558 将 M1552 的 474.72M 逐 token JSON 路径拆成两个有限对象：

- PATCH 的 430.08M token 只做 chunk-vectorized S1 histogram/debt 累计，不写逐 token payload；
- 24 个 FC1/FC2 层的 44.64M token 写独立 zlib binary frame。每帧依次保存 support、sign、nonunit bitmatrix，逐 token little-endian `uint16 nnz`，以及 row-major 非零 signed-int8 code；全零 token 仍保留一行 bitmap/nnz。

M1458 的 `input_elements/input_active` 给出 FC 原始 payload 上界 `7,528,535,874 B`，加独立 zlib frame、固定 header 和 64 MiB auxiliary allowance 后为 `7,598,737,368 B`，严格低于 12 GiB。Producer 构造必须消费一次性 preload permit；permit 绑定 resolved output、32 层 inventory SHA、estimate 和 free-space receipt，并要求估算后严格剩余超过 16 GiB。运行期 raw/disk 写入也有 12 GiB hard cap。

Synthetic fake-hook 在当前 Python 与 CPython 3.6 均通过：3 samples、6 binary frames、18 FC tokens（含 6 个 zero token）、36 个 nonzero code、3 条 PATCH histogram；21 项攻击覆盖 permit 伪造/复用/目录、hook/sample/order、tail bit、nnz/sign/nonunit、frame truncation/header/size、静态 axis inventory、runtime cap 和 release 阻断。

固定点仍只是在 captured diagnostic codeword/contributor 范围 exact；`hardware_quantization_authority=false`、`model_bit_exact=false`。M1554 的 99.2% global-capacity drop 未进入本 source 的准入门，S2 仍需 capture 后以 activity-relative safe reference 重筛。
