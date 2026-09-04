# M1290｜M1281 decoder surrogate production adapter additive repair

## Outcome

**PASS source-only repair，等待异作者 receipt-blind hammer。**旧 M1281 source、test、
contract 与 receipt 均未修改；本轮未打开 M1111DR2 live work prefix 或 canonical
result，未执行真实 calibration、EDA、GPU 或远端任务。

M1286 的全部 P0 已转成可执行边界：

1. Fixture 与 production API 已拆开。`calibrate_fixture` 要求
   `type(synthetic_fixture) is bool` 且值为 `True`，其 analytical annex 永远为 false；
   `calibrate_production()` 为零参数，拒绝 caller path、PASS bool 和裸 SHA authority。
2. Production 入口内部验证 M1111DR2 exact 三个 payload 文件、nested manifest/
   outer seal、每个 member hash，以及异作者 M1291 result-hammer 的 flat manifest/
   outer seal。Hammer review 必须密码学绑定 result manifest、outer 和三个 payload SHA。
3. 120 rows 直接投影 sequence/sample/module/configuration、三个 transaction digest、
   kind-summary digest、diagnostic cycles 和六类 summary；不接收新的自报 group/term/
   traffic schema。
4. Group count 只能由 descriptor/weight/psum-read/compute/psum-write 五类共同 count
   得出；active terms 只能由 weight traffic/16 得出，并要求
   `group <= active_terms <= 8*group`。Commit bytes 逐 call 取
   `output_commit.traffic_bytes`，D0/D1/D2/D3 分别核对 13,824,000 / 27,648,000 /
   55,296,000 / 221,184,000；不存在旧的 288 B/call 常数。
5. 必须有 30 个 distinct sample identities、3 sequences、4 modules、每层 30 个
   distinct observations。任何层内退化为重复 observation 都 fail closed。

## Tests

10 个 synthetic production-shaped unit tests 全部通过，覆盖：裸 authority、非 Boolean
fixture flags、result/hammer seal 损坏、digest substitution、协调字段伪造、term/group
越界、sequence/module swap、per-layer diversity collapse 和四层 commit 值。Fixture
即使零拟合误差也不能启用 annex。

这些测试只在临时目录构造与 M1111DR2 schema 同形的数据；没有读取 growing prefix、
canonical result 或未来 M1291 hammer。Python source/test 仅内存 compile，未生成新
pyc 作为证据。

## Claim boundary

当前仍是 `calibration_only=true`、真实 120 双封未验证、
`analytical_cycle_annex=false`、`speedup=false`、`system_speedup=false`、
`paper_ppa_ready=false`。Traffic 仅可称 M1111DR2 exact diagnostic projection，不是
新的性能模型；cycle equation 仍是待真实双封校准的 analytical hypothesis。

`docs/359_DATE终局冻结_20260813.md` 未修改，SHA256 保持
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。
