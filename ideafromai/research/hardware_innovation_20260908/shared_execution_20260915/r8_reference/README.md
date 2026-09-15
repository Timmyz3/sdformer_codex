# R8在共同双context资源上的参考

冻结flatR8的native4P、borrowRR、count21、bitmap7已在统一source/origin协议上完成96个命令、2944次tile执行，raw/J/wide/I24各11304960个值全部通过。新18序列36tile仅使用共同原始source与FP32 identity，R8 gold由旧R8参数独立计算，未借空间phase3输出。

下表是完整服务周期，**service=RTL total_cycles＋1个go拍**。cold首次配置，warm第二遍同模型不reset；BP使用共同确定性日历。两个64集是首帧不同tile；36tile来自18序列每序列首帧的edge128/interior9664，不是新增网络质量评测。

| 输入 | R8控制 | cold | warm | BP cold | BP warm |
|---|---|---:|---:|---:|---:|
| held128..191 | native4P/2 | 787604 | 785756 | 863410 | 861125 |
| | borrowRR/4 | 786422 | 784574 | 861958 | 860363 |
| | count/21 | 772612 | 769866 | 849135 | 846440 |
| | bitmap/7 | 750650 | 748802 | 829035 | 826263 |
| disjoint4000..4063 | native4P/2 | 831997 | 830149 | 909390 | 907583 |
| | borrowRR/4 | 830887 | 829039 | 908363 | 906823 |
| | count/21 | 810789 | 808043 | 888295 | 885835 |
| | bitmap/7 | 792116 | 790268 | 870475 | 868983 |
| 18seq36tile | native4P/2 | 376567 | 374719 | 414728 | 412520 |
| | borrowRR/4 | 376447 | 374599 | 414068 | 412373 |
| | count/21 | 369392 | 366646 | 407775 | 405003 |
| | bitmap/7 | 357261 | 355413 | 395613 | 393955 |

同源输入已与 `../spatial_rr` 三主集合按序逐字核对，source251904字、FP32 identity629760字、origin328标量全相等。两者替换同一个 `sttmultires_unet.encoders.swin3d.patch_embed.residual_encoding.resblocks.0.conv2.0`，source是C96×4×4、低10位T门字，物理origin=2×输出tile坐标−1；identity是原resblock输入。两网络评估器均使用旧 `data/model_access.py::load_parent` 相同前缀。首帧新捕获tile128/9664与旧全帧source/id和重算R8完整输出逐字一致。原137个参考fixture与既有R8fullgold一致；36新tile只借source/id，其raw由native因子和expanded-W两种CPU形式独立核等。

这是不同有损函数的同资源参考：flatR8自己的valid825 AEE为1.327635022608，自由U3M为1.258343102157（各自已完成的网络评估）。空间质量较好不表示其执行一定更快；旧单context百分比不用于本表。最终空间共同执行结果由其目录给出。R8小集12、空间小集15，不能比较小集合计周期。

本次仅改变统一加载和可用状态容量：每ctx实际Z8×40×52=2080B，总4160B，仍同一个416bit服务；R8算法用前10行。cache声明扩大为384B；单份producer8×32ALU、8×19×13mult，单份consumer8×64链、8×32×32mult及真实borrow仲裁保留。R8 bitmap额外8个pop16树、2592B plane、活动状态，count单份class/rep及配置均显账。共同静态/holding/cache并集与candidate独立审阅见 [RESOURCE_REVIEW.md](RESOURCE_REVIEW.md)；未测综合、频率或PPA，不宣称所有未用许可均物理保留在裁剪网表。

参数cold共1848拍（Q1 864、Q2 96、k-live 864、consumer24）；count另加class864＋rep32＋group1，mode21另加排列1，总2746。bitmap plane在原Q1加载时按位转存，无额外外部加载拍，但存储/线路显式存在。各mode享同驻留配置权限；额外跨mode2→7、7→21、21→4、4→7的ready/BP回放首次进入count才付898拍，所有回放均不reset。加载当前模型后的不同source/origin按manifest虚拟tile索引处理，只有I24退休后回收。模型换参失效接口尚未实现。

原两64的16条ready记录对旧bitmap_rr核心、消费者、所有grant/仲裁计数及total_cycles逐项相等。BP新增全1536源字真实握手与origin背压，边界padding也付加载，故不照搬旧BP周期。每个返回请求/输出在停顿时保持；zero/one/random/poison与真实小集、非连续跨序列/跨row、跨mode均通过。prepared176fixture，其中175种实际用于此次RTL回放。

独立 [verification.json](verification.json) 共1722798项检查，通过native raw、FP32→J20、wide、I24RNE，class合法性/代表恢复/8bit计数界、bitmap signed3分解及全部source/W/Z/psum/ALU/wide服务恒等式；最高count158。CPU profile独立核175fixture共672000个值/阶段。R8本次没有增加latent monitor，latent由CPU两种形式和原核断言支撑，不称逐RTL latent值已测。对空间candidate的Z/D实测覆盖应引用其报告。

复现使用 `/opt/anaconda3/bin/python3.12 prepare.py`、`implement.py`、`run.py --stage small`（Verilator4.028 --cc --exe后make），随后对held/disjoint/sequences/swap使用 `--skip-build`，最后运行 `cross_inputs.py`、`verify.py`、`summarize.py`。原始每命令一行在 `results_*.jsonl`，汇总和拥塞/配置拆账在 [comparison.jsonl](comparison.jsonl)。所有旧目录只读，无GPU/EDA/训练/生产/Git改动。

空间shared结果的解释：同held64，自由U3M borrow完整冷service1433162，较R8 bitmap7的750650慢90.923%。空间Q2为634068次8lane MAC，R8为198720，约3.191倍；disjoint为660936/210384≈3.142倍，36seq为217296/69276≈3.137倍。空间将更多邻域收缩留在多位连续Q2，flatR8把3×3邻域留在binary Q1，二者还存在rank、位宽与质量差异。不能把空间相对其自身expanded OS的收益移到R8强控制分母。借用consumer64只迁走部分Q1更新，held空间不借→借仅1433737→1433162，未解决主要Q2发射和Z/ALU争用。其余空间最终汇总以 [spatial_rr](../spatial_rr/) 为准。

本轮接入后没有另加调度或算法。下一条有区分力的迁移接口与完整度边界见 [NEXT_INTERFACE.md](NEXT_INTERFACE.md)。
