# M806 / pinned M805：M798 decoder 最终 launch hammer

结论：**PASS 100/100**，P0/P1/P2 = **0/0/0**。M798 true release、candidate、M799 source hammer、driver 和 one-shot runner 的 exact-SHA 链闭合。根代理现在只被授权执行请求中冻结的那条命令一次；本 hammer 没有运行 one-shot、没有消费 attempt，也没有产生任何周期或加速比。

## 1. 身份与双封

- true release：`cebcfe1f...813b892`；member manifest file：`2b73500b...3a952`；outer seal file：`69d94827...6a6bc`。
- candidate：`db787cb6...1fed7`；member manifest file：`06a80110...c577`；outer seal file：`4f0f1f1d...1ed36`。
- M799 source review：`8399ce60...52d9`；manifest：`dc38106f...5f87`；outer seal：`9f4e249c...b3bf`；状态为 `PASS100_M798_SOURCE_CANDIDATE__AUTHORIZE_TRUE_RELEASE_ONLY`。
- release 的 `source_identity` 与 candidate 完全相等；`reviewed_source_identity` 精确绑定 candidate `db787...`、driver `44b7...`、runner `daf559...`。
- release、candidate、source review 均用拒绝 duplicate key 和 NaN/Infinity 的严格 JSON 解析器重新读取；重复键、NaN、Infinity 三种负例均被拒绝。
- `docs/359_DATE终局冻结_20260813.md` SHA 仍为 `dedde7ce...bdfc4`。

## 2. 唯一允许执行的 preflight

只调用了 driver 的 `--validate-release-preflight`，返回：

`PASS_M798_TRUE_RELEASE_PREFLIGHT__ONE_SHOT_UNCONSUMED`

release SHA 与 candidate SHA 分别精确为 `cebcfe1f...813b892` 和 `db787cb6...1fed7`，`production_replay=false`、`production_cycles=null`。全过程 canonical result、attempt、failed-or-incomplete quarantine 均不存在。

## 3. 冻结语义

- M686 40 records 与 M699 120 records 分开，T=10；三配置为 A1-OSG、equal-service K1×8、typed signed K8。
- 资源严格为 96 lanes、245760 B macro-rounded SRAM、Acc24、3 ns、192 B/cycle。
- D1 的周期进入 `total_cycles`，但 module index 1 不进入 `headline_total_cycles`；唯一合法 headline 比值是 K8 / equal-service K1×8，K8 / A1 禁止作为 headline。

## 4. Pre-attempt 攻击

以下七项均 exit 1，且均未创建 result/attempt：

1. duplicate release key；
2. release source SHA 篡改；
3. M799 review SHA 篡改；
4. D1 headline exclusion 篡改；
5. 资源 tuple 篡改；
6. caller 提供错误 release SHA；
7. caller 缺少 release SHA。

## 5. Runner 边界

静态复核确认：runner/release SHA 环境绑定在前；release preflight 与 disk/memory/commit resource gate 均在 attempt `mkdir` 之前；attempt 只有一个消费点并在 production 前双封；stage 是 canonical result 的 sibling；发布使用 `renameat2(RENAME_NOREPLACE)`；碰撞和中途失败进入 no-clobber quarantine；canonical root 只接受 `result.json`、`detailed_rows.json`、`SHA256SUMS`、`SHA256SUMS.seal.sha256` 四成员。

## 6. 授权与红线

只授权根代理执行以下 exact command 一次：

```bash
env -i PATH=/usr/bin:/bin LANG=C.UTF-8 LC_ALL=C.UTF-8 M798_EXPECTED_RUNNER_SHA256=daf55986d22c7c0250cb39c0dfd9cf4f35fa01f654bcafbc00743a7a12886fb5 M798_EXPECTED_RELEASE_SHA256=cebcfe1f65d49396fb8ea08c01dd0027cb313c957a98c00d4836b1d4b813b892 /home/zhumd/work/sdformer_codex/SDformer/hw_autoresearch_nts07/system_simulator/scripts/run_m798_m785_decoder_physical_residency_one_shot.sh
```

生产 raw result 必须再经 fresh result hammer 才能引用。当前不得写 decoder complete、Table-A、系统加速、RTL/VCS/EDA/能量/PPA 主张。
