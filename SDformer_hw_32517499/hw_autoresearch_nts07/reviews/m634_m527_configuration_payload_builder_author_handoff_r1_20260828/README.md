# M634｜M527 五档 executable configuration payload builder 作者交接 r1

## 本次关闭的缺口

M625-P1-01 指出：M624 的 R6 没有独立生成并 SHA 绑定 M527 common-resource manifest 与 B0/B1/B2/B3/Ours 五档 executable configuration manifests。本次新增：

- `system_simulator/scripts/build_verify_m634_m527_configuration_payload.py`
- `system_simulator/tests/test_m634_m527_configuration_payload.py`

生成器/验证器 SHA256：

- builder：`b53429d9444e44f33cb9a240f696a3d847323da1af7929ed43e473e87fa564fa`
- test：`bc60e3bd5f6689677883b9f04d3b860b4e0774c0846fbd7a0c6c8cdbcddaa5e1`
- frozen M527 r3 contract：`83ea25e43b53d12800ac64e971069a682e3077411ff10851a7861636ef77355b`
- frozen docs/359：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

## Fail-closed 输入

`build` 只有在以下输入均为 repo 内非 symlink 普通文件、SHA 非空且现场重哈希一致时才会原子写新目录：

1. measurement-identity binding，逐一绑定 complete-trace、sequence-population、aggregation-weight 三个 manifest、checkpoint、frame definition、density metric/bins 和完整 operator-id universe；
2. unified simulator source；
3. common-resource source；
4. 恰好五个 configuration source：
   - `b0_dense96_fixed_t10`
   - `b1_project_defined_ptb_like_structured_k1x8`
   - `b2_exact_bit_sparse_k1`
   - `b3_exact_bit_sparse_k1x8`
   - `c123_ours_exact`

每个 configuration source 必须把 frozen operator-id universe 精确分割为 optimized 与 same-model fallback 两部分。漏算、重复或交叉均拒绝。输出目录必须事先不存在；任一预检失败不创建 output。

## 公平性与资源账

五档都逐字段绑定同一个 physical resource tuple：28 nm、3 ns、96 source lanes、一个物理 K1x8 service pool、240 KiB SRAM、64 GB/s decimal DRAM、192 B/3 ns cycle、Acc24，以及同一组 queue/bank/port 参数。B2 K1 只把 `execution_service_limit_sources_per_cycle` 限为 1，不能删除另外七路物理资源逃避面积/能量账。

所有 `charge_policy` 字段必须为 `true`。fallback 必须在同一 unified model 中执行，并对 cycles、traffic、energy、area 全收费；unsupported operator IDs 必须显式列举。工具拒绝未知 manifest 字段，因此即使攻击者重做内外 seal，也不能偷偷加入 `system_speedup_admitted=true`。

## 输出与 seal

成功时输出：

- `common_resource_manifest.json`
- 五个 `<configuration_id>.json`
- `registry.json`
- `verification_receipt.json`
- `SHA256SUMS`
- `SHA256SUMS.seal.sha256`

`validate` 会复核双 seal、精确 member set、所有 live source SHA、五档 ID/顺序、共同资源 tuple、charge/fallback policy、trace/simulator/common-resource SHA 和 claim boundary。

## 当前边界

本次没有生成 production payload，因为 decoder-complete trace、safe unified simulator 与五档配置源尚未全部冻结。测试只在临时 synthetic fixture 上验证 fail-closed 逻辑，退出后自动删除。

即使未来 production payload 通过，仍只代表 `configuration_registry_payload=true`。M527 r3 自身三个 admission gate 没有被修改：fixed numerator、unified measurement、system speedup、waterfall、effective GOP/s 与 paper headline 继续为 `false`。complete-trace 的上游 schema/population/decoder 语义仍必须由 superseding availability analyzer 独立验证，M634 不把“文件哈希存在”冒充“全网 trace 语义闭合”。

## 定向验证

- Python 3.6 `py_compile`：PASS
- `unittest -v`：9/9 PASS
- 覆盖：完整生成/复验、null SHA、trace mutation、operator partition gap、资源宽度偷换、未收费资源、payload mutation、重封后未知 overclaim 字段、live config-source mutation。
- GPU/EDA：未运行
- `docs/359`：未修改
