# M1102 C1 legal-work8 additive source author receipt

结论：**SOURCE GO；launch/attempt STOP；M1095 DO_NOT_RETRY。**

本次只新增合法 work8 域适配与新的原子库命名空间。语义是唯一被 M1100/M1101 允许的修复：`work=0` 沿用 M1086 的无事件、无状态修改行为；所有 `work>=8` 原样委托冻结 M1056；`1..7` 严格拒绝。canonical provenance 额外要求 `work%8==0`，没有引入 bank subset、event compression 或 minimum-15 padding。

## 已实际执行的只读门

- 穷举 `812,160 tasks × 3 designs = 2,436,480` 个 work 值，full provenance coverage PASS。
- 三个设计各为：zero `74,106`、work8 `4,174`、positive>=16 `733,880`。
- 总计 `12,522` 个真实 work8 occurrence；每个均执行 frozen M1056 fresh 与 delayed-RAW 两档回归，两档都是 `12,522/12,522 PASS`，RAW dependency `12,522/12,522 PASS`，最小 dependency delay 为 `0`。
- generic API 的 8..14 均与 M1056 bit-identical；canonical 边界拒绝非格点 9..14；1..7/bool/negative fail closed。
- 没有调用 full cycle iterator，没有生成新周期或 speedup，没有创建 attempt/result/lock，也没有启动 EDA/GPU/remote。

## 身份

- semantic source SHA256：`95bd50aebcc473ab69cdea6ccf27d54743c89926c5e0f31199dc469ced9bf7cc`
- atomic library SHA256：`0325a4c901e945656ad6d74b12cae6b066f5b75bb426326143f8b0a8f24d1157`
- contract SHA256：`fad9c381fc1e55fc78d6cf4b95ad0959b5a7089989a7acce1ccfafa73714db6e`
- contract outer-seal-file SHA256：`b17774b1b3fad06f104081b2ab2b0de4b3b539c72fd9e6adcb2171a46d55770c`
- docs/359 SHA256 保持 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`。

下一步必须由不同作者对 source/contract/atomic library 做独立 hammer。通过后，仍须由不同作者生成零参数 launcher 并再 hammer，才允许唯一一次新的 CPU attempt；不得复用或重试 M1095。
