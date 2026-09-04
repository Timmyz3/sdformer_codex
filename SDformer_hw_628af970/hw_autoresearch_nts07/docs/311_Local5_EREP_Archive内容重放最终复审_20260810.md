# Local5 EREP Archive 内容重放最终复审

## 1. 最终裁决

独立 DATE 风格第三次复审结果：

```text
评分：4.5 / 5
裁决：Accept-Synthetic
P0：0
P1：2
P2：0
```

该裁决只接受 `[synthetic-contract]+[代码审计]` 的 archive/parser 合同，不接受
正式 `[rtl]`、workload 性能或 DATE PPA 声明。

最终结果目录：

```text
results/local5_erep_ledger_replay_v4_reviewfix_v3_20260810
```

48/48 测试通过，result、receipt 和 source-input SHA 全部通过。

## 2. 已关闭问题

1. parser 在 `np.load` 前读取原始 ZIP `infolist()`，冻结成员名、顺序、数量和
   唯一性；重名成员不能经字典折叠绕过；
2. 目录项、archive/member comment、member extra、非零 flag bits 和非
   stored/deflated 编码全部拒绝；
3. mutation 覆盖重复成员、乱序、目录、archive comment、member extra 和 BZIP2；
4. raw event 独立重建 phase/head/window ledger，再与上层 ledger canonical SHA
   对照；
5. expected/actual Acc32 mismatch 由 parser 逐元素重算；
6. runner 生成确定性 `source_bundle.tar.gz`，固定 tar 元数据并使用 `gzip -n`，源码包
   纳入 result/receipt SHA；
7. 文档复现命令、最终目录、测试数和实际结果一致。

独立审稿人重建的 source bundle SHA 与结果包一致，说明本地未跟踪源码可以从结果包
恢复。它仍不替代 Git commit/tag 或外部签名。

## 3. 保留的两个 P1

### 3.1 正式规模正路径

当前没有实际通过以下完整规模：

```text
1200 window
13800 input head
462600 phase
198720000 Acc32 scalar
```

formal runner 仍在 admission receipt 缺失处 fail closed。正式数据到达后还必须验证
大文件是否触发 ZIP64 `member.extra`；当前 canonical contract 拒绝任何 extra。若正式
成员超过 ZIP32 边界，应先评估分片 archive，而不是静默放宽 parser。

### 3.2 双来源 provenance

当前 parser 能证明 expected 与 actual 逐坐标一致，但不能单独证明：

- expected 来自冻结的独立软件金参考；
- actual 来自目标 DUT、规定 filelist 和规定仿真命令；
- 两者没有被同一 adapter 同步复制或错误重排。

正式 adapter 必须分别绑定软件金参考生成器 SHA、DUT RTL/filelist SHA、仿真器版本、
仿真命令和各自的原始输出，再由第三段只读合并器构造 miter archive。这个来源链未完成
前，formal G0 保持 DENY。

## 4. 下一步

Local5 当前不再扩 synthetic parser。等待正式 profile100 producer 完成后，按顺序：

1. 运行已接受的 HxH preflight；
2. 实现 software-expected、RTL-actual、read-only merge 三段式 adapter；
3. 执行正式规模 archive replay，检查 ZIP32/ZIP64 和内存峰值；
4. formal G0 通过后才允许生成 EREP candidate RTL。

等待期间转回 Motion 线，只推进一个不依赖 GPU 的最高优先级证据缺口。
