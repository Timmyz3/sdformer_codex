# M795｜M793/M785 decoder production source fresh hammer

结论：**NO-GO true release，72/100，P0=0、P1=3、P2=2。** 本轮只做 source hammer；未创建 true release、attempt、production result，也未生成 cycle/speedup。

## 通过项

- Candidate、driver、runner 的成员 SHA、manifest 与 outer seal 全部独立复核通过。
- `py_compile`、`bash -n`、synthetic self-test、candidate validation 通过；后两者明确返回 `production_cycles=null`、`production_speedup=null`。
- M785/M786/M790 身份链和 M686 40 records、M699 120 records 的双封输入均被重验。
- analyzer、tests、storage oracle、M785 contract、M790 review、M686/M699 manifest 的单字节篡改全部 fail closed。
- duplicate member、extra partial file、symlink、outer-seal bit flip 全部被 sealed-directory verifier 拒绝。
- 源码循环确实是两个 population 分开执行、40+120 records、T10、三种 config；每个 record/timestep 新建 scheduler，不给跨 record/population overlap credit。
- 固定资源为 96 lanes / 245,760 B / Acc24 / 3 ns / 192 B/cycle；唯一合法命名比较是 typed signed K8 对 equal-service K1x8。

## 阻断项

1. **D1 实际进入 headline ratio。** `headline_eligible=false` 只是行标签；`per_config.total_cycles` 仍累加 D1，population 的 K1x8/K8 比值直接使用该总数。必须另算排除 module 1 的 `headline_total_cycles`，并用它生成唯一 headline ratio。
2. **canonical result 的碰撞发布不安全。** `mv stage result` 在 `result` 已被并发创建为目录时会返回 0，并把 stage 嵌入 `result/stage`；当前 post-check 仍为真，但 canonical root 没有 `result.json`。必须改为原子 no-replace，并检查 canonical root 四件套。
3. **M795 authority 未绑定 exact candidate SHA。** runtime 仅检查 review 的 status/score/severity；必须同时核 M795 中 candidate/driver/runner SHA 与 release binding。
4. `strict_json` 接受 duplicate key，需要 object-pairs duplicate rejection。
5. 封存的 M794 request 把 M699 outer-seal SHA 写成 58 位；candidate 中 64 位真值正确，需另发 corrected request，不能改封存 request。

## 修复门

修复以上 P1/P2 后必须冻结新的 driver/runner/candidate SHA，重新做 receipt-blind source hammer。当前 M793 不得生成 true release，更不得运行 production；C2/decoder cycle、speedup、Table-A、system speedup 均仍为 false。
