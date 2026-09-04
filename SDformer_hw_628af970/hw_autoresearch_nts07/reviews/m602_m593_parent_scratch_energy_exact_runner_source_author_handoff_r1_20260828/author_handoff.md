# M602｜M593 parent-scratch energy exact-runner source author handoff

## 状态

M602 exact runner 与 `launch_now=false / release=false` source candidate 已完成。当前只执行了 Bash syntax、无业务结果的 `--preflight-only`、缺失 future authorization 的 fail-close 和 `/tmp` no-replace collision self-test；没有正式 analyzer run、result、attempt、EDA、GPU 或远程动作。

本作者曾完成 M599 review，因此不能自评 M602。下一步必须由不同 fresh agent 执行 M603 runner static hammer；只有 M603 `P0=P1=0`，root 才能另立双封 M604 true-launch admission。

## 关键关闭项

- runner 冻结 M597 analyzer/contract/handoff、M599 review，以及 M504/M528/macro-map/M595/docs359 全部 path/SHA/manifest/outer。
- canonical result、attempt、consumed attempt 与 runner/analyzer staging 均位于同一个 `results/` parent；`lexists/lstat` 拒绝 symlink、dangling link 与既有目录项。
- EXIT/INT/TERM/HUP trap 在 attempt `mkdir` 前安装；失败时 attempt、runner staging 和 analyzer 内部 staging 一起搬入 unique quarantine，递归 member manifest + outer seal，再用 `renameat2(RENAME_NOREPLACE)` 发布。
- success 路径对静态身份、future authorization、analyzer output schema/conservation/claim、terminal receipt、最终 manifest 做 pre-publish 与 post-publish 直接重哈；attempt 单独封印并 no-replace consume。
- 当前 runner 的 `--execute` 必须拿到固定坐标的 M604 true admission；该 admission 又必须绑定 fresh M603 review 且 `P0=P1=0`。M604 当前不存在，所以不会误启动。

## 身份

- runner SHA：`6a54d938f598835114c2e463e56eb03f4e0754947dbbeb0b33f03fd04e569b2c`
- source candidate SHA：`4261d4a4409e37e580b930afd239a3d4d8d65a851cdd4c78ebe3d86e568c0574`
- M597 analyzer SHA：`6896c8a406dc3274926e6c7d958136aca47b9df9afa3522d6c2539a142ea9cf9`
- M597 contract SHA：`90399b6c932e28f6eac38f3408af0374b23beb369e1fd4e57e3b98d92d28b1bf`
- M599 review SHA：`56ac7aafd7b603d437efe267ee2875909a365072181d0abd9101fd5d601497b1`
- docs/359 SHA：`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

`38.2283079189%` 与 `1.2622562287 mJ/frozen sampled inference` 仍只是 M599 诊断，不是本 handoff 产生的结果或 paper data。
