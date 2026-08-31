# M316：M312/M313 valid825 两阶段流程安全 preflight

结论：当前 **不能启动 M312**；M313 只能等待修复后的 baseline、生成新 exact-SHA candidate contract 后再次独立复审。

M312/M313 已能拒绝独立 token 形式的重复参数：M312 为 8/8，M313 为 11/11。但 `unique_option()` 只识别完全等于 `--option` 的 token；`--option=value` 不会计为重复，下游通用 M284 argparse 又采用最后一个值。因此 contract、config、checkpoint、path-results、distance、max-samples、BN、dump 和 baseline 参数仍可 last-argument 覆盖。

最严重的是 `path-results` 双根：外层保存第一目录，通用 wrapper 可被后置 `--path-results=第二目录` 驱动。GPU 输出写入第二目录，外层却可以检查并封存预先布置的第一目录。M310 的 duplicate-argument P0 因而没有闭合。

第二个根问题是合同自认证：合同记录 launcher SHA，launcher 检查该字段，但没有独立 seal 指定“本次必须使用哪一个合同 SHA”。保持相同 schema/milestone、填入真实 launcher SHA 的克隆合同仍可重新绑定 checkpoint、config、输入和 gate。

M313 的方向是正确的：源码计划 exact-SHA 绑定 baseline receipt、profile、per-frame、launch receipt、manifest、seal，并固定唯一 `[0,2,3]`。但 baseline 尚未生成，M313 contract 也不存在；同时 manifest 内容、nested order/identity、finite Decimal AEE 和 candidate receipt 身份仍未闭合。因此它只能在新合同生成后再审，不能提前准入。

修复顺序：

1. 单次标准化解析，`allow_abbrev=false`，拒绝所有语法形式的重复 destination；禁止转发原始 `sys.argv`。
2. 用独立 launch manifest 同时 exact-pin launcher 和 intended contract。
3. 强制 `per_frame.csv` 与其它产物位于唯一、空的 canonical result root。
4. strict JSON、有限 Decimal AEE、完整 nested identity/order/admission 重算。
5. manifest 必须恰含四个 canonical 条目，生成后立即 replay，并校验 seal 的 hash 和文件名。
6. 修复 M312 并重新生成合同、复审、运行 baseline；随后生成 M313 candidate contract，再做一次独立 preflight。

评分 `54/100`；`P0=3, P1=4, P2=2`。M316 未导入 evaluator、未运行 GPU，也未修改主源码、合同、RTL 或 docs/359。
