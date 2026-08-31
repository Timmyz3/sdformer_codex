# M655｜M649 strict-S10 retry 作者交接

## 当前裁决

`STATIC_RETRY_HANDOFF_ONLY__FRESH_HAMMER_REQUIRED_BEFORE_GPU`。

M653 放行的 M649 首次 GPU 数值诊断已经执行。模型与 checkpoint exact load 通过，10 个样本、4 个 decoder hook 共 40 条记录均在内存中完成；但是原循环在第 10 次循环体结束后向 DataLoader 请求了第 11 项，缺失的 `zurich_city_09_a_0101.npy` 触发 `FileNotFoundError`。事务按设计 fail closed：canonical 结果不存在，只保留一个 544-byte `FAILED.json` staging，内存记录未发布、未复用。

## 唯一修复

- 新增 `take_exact(iterable, 10)`：只调用底层 iterator 十次，生成器结束时不探测第 11 项。
- 新增定向 iterator：第 11 次 `next()` 会立即抛错；15/15 测试证明调用数严格等于 10。
- 合同把首次 M649 失败 staging/receipt 加入第 24 个冻结输入；重试开始、结束均核验其目录 population、544 bytes、SHA、完整 JSON 语义和 canonical absent。
- 首次失败 staging 只读保留，不删除、不改名、不补写、不当作结果。

没有改变数据、样本、checkpoint、模型、hook、numeric gate、输出 canonical 或候选命令。没有运行第二次 GPU、M511、EDA 或远端任务。

## 新身份

- launcher `d9e362eea0627b0e7b8d84e9ed339142366362cb3382298aee4a438ef5087dfd`
- contract `580ddee0e52ef325df5ba73ed799dcd4a6b6fb25e94123428230cf752f405b5b`
- tests `6808105e0f5058dfbbd38f538724325dc772d9a18625c18d4b979dcf47abd422`
- first failed M649 receipt `28906352856590248383c939db05ba0023a7a99dedaa594d1851481ba375d59c`, 544 bytes
- checkpoint `4f33e086070bb92524d94727c6e39cdb7296441c2660e70f1d7be29467645158`
- `docs/359` `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`

## 运行边界

第二次 GPU 运行尚未授权。fresh independent hammer 必须重验失败 staging 未漂移、canonical absent、严格十次 iterator 回归、24 项 M649/21 项 M511 identity、路径攻击和全部 claim boundary；只有 P0=0、P1=0 且明确给出合同内唯一命令才可重试。即使结果成功，仍须 fresh result hammer，且不自动授权 payload、cycle、speedup、RTL、EDA、PPA 或 DATE headline。
