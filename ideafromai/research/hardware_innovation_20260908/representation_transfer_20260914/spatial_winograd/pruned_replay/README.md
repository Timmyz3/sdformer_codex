# 两个冻结剪枝函数：原普通／通用 Winograd 完整回放

已用父目录锁定的同一版 RTL、TB 和实际 FP32 identity→J20→wide64→I24 消费者，分别回放 moment 与 native_tap。没有增加或改变 controller。**moment 的通用四 M Winograd 在新增36跨序列输入上恢复到7.376%冷收益；native_tap自己的普通分解更快，接通用Winograd反而慢10.295%。** 这是各自函数内部的实测，两个剪枝函数之间必须另结合质量比较。

| 函数／输入 | 普通m0冷ready | 通用m1冷ready | m1改善 | 普通m0冷BP | 通用m1冷BP | m1改善 |
|---|---:|---:|---:|---:|---:|---:|
| **moment／18序列36tile** | **980,690** | **908,354** | **7.376%** | **1,025,097** | **953,063** | **7.027%** |
| **native_tap／18序列36tile** | **849,931** | **937,429** | **−10.295%** | **892,884** | **983,159** | **−10.110%** |
| moment／held64 | 2,334,322 | 1,975,186 | 15.385% | 2,418,895 | 2,060,295 | 14.825% |
| native_tap／held64 | 1,942,296 | 2,066,038 | −6.371% | 2,023,740 | 2,153,800 | −6.427% |
| moment／disjoint64 | 2,510,814 | 2,107,026 | 16.082% | 2,598,908 | 2,195,989 | 15.503% |
| native_tap／disjoint64 | 2,095,294 | 2,202,605 | −5.122% | 2,180,333 | 2,293,713 | −5.200% |

这是实际配置、逐tile source/origin/start和最后I24全包含的冷流累计周期；模型只在第一命令加载。暖流为第二遍无reset回放：m0表中各数减1176、m1减1368即可得到实际测得的暖服务，完整原数见 [comparison.json](comparison.json)。source/W/raw/FP identity/output采用父相同BP日历；每个命令必须等I24完成才配置下一source。18序列每首帧固定edge128/interior9664；两64仍为原同帧不同区域，不能作为新序列数量。跨序列moment仍有15/36个tile变慢（21赢），native_tap的通用Winograd则36/36变慢，逐tile数据在各函数的 `sequence_results.jsonl`。

moment的192个(rank,N8)组全满足 `g1=g0+g2`，因此四项U中的U2整向量为零；原生三tap没有整N8零组。通用m1确实通过既有weight/support路径跳过这个M的乘法及W读，但仍计算完整四D、初始化四M并执行完整恢复。native_tap删192/576个原生向量，另外出现106/768个U零向量；普通m0同样有权利用它的零组。选择、约束与导出证明见 [冻结函数报告](../../spatial_winograd_pruning/README.md)，不能把25%变换零组与三分之一原生零组当相同算量或相同质量。

| 跨序列36tile | 普通Q2向量MAC | 通用Q2向量MAC | 少用MAC拍 | 固定变换/恢复新增 | 冷额外配置 | 最终净省拍 |
|---|---:|---:|---:|---:|---:|---:|
| 原母Q11（父锁定结果） | 391,776 | 287,004 | 104,772 | 101,952 | 192 | 2,628 |
| moment | 391,776 | 217,296 | 174,480 | 101,952 | 192 | 72,336 |
| native_tap | 261,017 | 246,371 | 14,646 | 101,952 | 192 | −87,498 |

所有tile不含BP的式子仍精确为 `普通周期−通用周期 = 普通MAC−通用MAC−2832`，冷流另减一次192。moment零U提升了实际跳零机会，但固定恢复税一拍未消除；native_tap已经让普通原生路径少做许多乘加，通用Winograd剩余机会不足。这解释了本次正负结果，而非仅引用系数零率。后续三M专用恢复由独立 `spatial_moment3` 叶承担，这里的失败/通用臂保持原样供它比较。

跨函数直接比较也应保留：moment通用m1比native_tap普通m0，冷ready在跨序列/held/disjoint分别仍慢58,423／32,890／11,732拍。根独立diverse10质量为 [moment AEE1.210152895772627](../../quality/moment/deployed_diverse/spatial_integer_summary.json) 与 [native_tap AEE1.2981784301740258](../../quality/native_tap/deployed_diverse/spatial_integer_summary.json)，母Q11为1.263526665171576。这只是10帧质量起步，完整825由根独立执行；本报告不拿母Q11或另一函数的AEE填补未完成结果，不以周期差宣称等质量优势。

raw-only也独立实跑。跨序列冷ready moment893,366→821,030，native_tap762,607→850,105；held64 moment2,179,098→1,819,962、native_tap1,787,072→1,910,814；disjoint64 moment2,355,590→1,951,802、native_tap1,940,070→2,047,381。真实消费者ready每tile在raw之后增加2425拍，冷参数另24拍，与父同接口恒等式一致；完整消费者表来自实际连线运行。

资源严格沿用 [父合同](../resource_contract.json) 和 [本次实施前PLAN](PLAN.md)：单context，8×32ALU、8×19×13 multiplier、source1920B、原地Z1280B、p_mem15360B、同单W/Z/psum服务、同8×64消费者ALU与8×32×32 multiplier。两臂共9984B单Q2表、416B qcache、96B额外三M和32B变换尾holding；父相对原factor列出的2754B额外数据/支持仍全部存在，不能因零U而声称已节省这些面积。m0/m1仍实际加载Q1+Q2 1152/1344拍，完整consumer再24拍；所有零向量配置与cache零写均付费。m1实际跳过零W的数据读取/MAC，没有免费删输入变换、输出恢复或新增端口。两函数新bounds见 [comparison.json](comparison.json)，均在现有D16/U13/M32及wide64范围内。

[prepare.py](prepare.py) 从父的178个fixture只借原source/origin/FP32 identity和相同Q1的Z/D，原输入通过链接引用；raw/wide/I24实际重新生成。每函数分别独立核Q1/Q2、展开W、完整及分stripe Winograd偶数重建、J20和最终RNE/sat。各135个已有真实tile与该函数export gold逐值相等；36跨序列输入没有借母Q11的raw或I24当gold。moment相对母模型实际550,651个raw和550,221个I24改变，native_tap为550,655/550,547，证实本回放使用各自新函数。根的独立新函数整网capture核验属于后续质量证据。

实际覆盖每函数15small（8真实、全零/全一/随机/tail/rank正负/图外poison）、held64、disjoint64和36跨序列，两mode、ready/BP、两遍连续换源。每函数每kind1432命令；两函数raw/完整consumer合计 **5728命令**，全部PASS。总raw21,995,520值，实际J20/wide64/I24各10,997,760值；Z7,331,840、D3,665,920值，逐状态/端口/配置/重复核验684,560项。见 [verification.json](verification.json) 与各函数 `raw_verification.json`、`consumer_verification.json`。M仍为前缀溢出断言＋静态路径审阅，未加逐M monitor；输出原序/末包/BP保持全核。模式各在独立reset/config进程，未测试无reset跨mode换布局。

复现：以 `/opt/anaconda3/bin/python3.12` 执行 `prepare.py`，再 `run.py --kind raw --stage small` 和 `run.py --kind consumer --stage small`，它们在本子目录用父锁定源码独立执行Verilator4.028 `--cc --exe`＋make。small通过后，对held/disjoint/sequences逐stage运行两种kind并加 `--skip-build`；最后 `verify.py`、`summarize.py`。默认两函数都跑，可用 `--arm moment/native_tap` 单独回放。没有拷贝或改父SV/结果，逐命令JSONL一行一条。无EDA、时序/面积/能耗结论、训练或生产/main.tex改动。

后续质量补齐：根代理已完成各函数独立825评估和真实网络36位置重捕，全部Z/raw/J/wide/I24与这里独立gold相等。moment AEE为1.296504，native_tap为1.285953；自由U零项控制为1.258343。见[完整质量表](../../quality/QUALITY_REPORT.md)。早期diverse10和本目录局部误差不是最终择优依据；这些质量补证不改变本报告原始RTL周期。三项执行和box2适配分别在兄弟目录独立实现，未回写本锁定双臂。
