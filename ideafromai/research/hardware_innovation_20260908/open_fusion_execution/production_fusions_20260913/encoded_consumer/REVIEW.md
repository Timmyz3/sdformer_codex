# 独立审阅：真实编码与连续消费者

审阅对象为 `NEXT_INTERFACE.md`、既有 `literature_owned/probe.py`、表示层 `quantizers.py`、原 Machine 整数操作，以及本目录新实现。审阅只读根实现与已有数据，不重复实验。已静态读 `binding.py`、`predictor.py`、`kernel.py`、`run.py` 与完整带spill的 `latent_cse.py`，并核对共同 H8 源驻留下最终35条实际记录。未发现未解决的数值或共享资源违规；该结论限于本目录CPU有限资源模型和下述固定边界。

## 不可跨越的数值边界

同一组既有参数的函数是 `b=Dg+c; q=clip8(RNE((x-b)/step)); xhat=sat24(b+step*q)`。x 必须是当前 Machine 实际产生的 updated I24，g 必须是当前真实 projection 门；压缩臂的实际码必须写 SRAM 后再读。fixed、affine、full-D 各自与其同函数 expanded-I24 控制比较，不能把不同函数的精确性合并。

原 `IRNE` 仅执行 ties-to-even 右移或精确左移，没有 sat24；`ISAT` 才是独立 signed24 饱和。源程序的 `ISOURCE norm24` 将两者融合，不能替代编码残差的原 IRNE。编码必须先对宽残差舍入、再有费 clip8，不得把预测值或未舍入残差先裁成 I24。U 所有项合并后经过原 U RNE/sat；V 保留原 RNE/sat、bias 与最终 sat。

分解消费采用 `U*xhat = step*(U*q) + D*(U*g) + c*sum(U) + U*e`，其中 `e=sat24(b+step*q)-(b+step*q)`。仅当对当前参数全部 1024 个 T10 门字、所有 signed8 码作参数范围证明时，才可删除 e；观察某一 halo 未饱和不够。预测、重建及 U 各中间值也须落在模型 signed48 中，不能借 NumPy int64/float64 的更宽范围。fixed 的纯尺度可以合法调整 U 指数；带加项的 affine/full-D 不可分别舍入 Uq、Ug 再相加。

真实 PED U 与 projection 卷积矩阵不同，已经有门不意味着已经有 Ug。Dg 编码预测与 D(Ug) 消费还原是两项不同费用。当前 full-D 常量为 signed19；二值门的 Dg 可以按条件加 D 执行，但多位 Ug 的 D(Ug) 不能冒充一次 signed16×I24 乘法。必须用实际窄乘分解与移位/加法，或适配现有资源并付控制 ROM 的常量 CSE；新增两槽写回操作只是有限资源 CPU 模型假设，尚无 RTL 时序证据。

## 容易遗漏的费用与共享资源

- 编码端真实 updated 与门的 SR 请求、响应收集、RF load、预测常量 CR/cold CW、预测运算、残差、RNE、clip8、打包与 SW，以及消费者重新 SR/解码与等待。
- expanded 控制同样先真实产生码，再实际重建 xhat 与写读 I24；允许在编码 RF 内立即融合重建，不强制额外 Q8 写读。这是更强的合法同函数控制。不能沿用旧 U 入口实验直接注入码/xhat 的免费入口。
- Ug、Uc cold fill/CR/load/add、宽 D 分解、所有真实分块造成的系数重读和 DMA 外送必须计入。`Uc=c*rowSum(U)`只依赖部署常量，可离线折叠；不要求运行时重算它。允许普通臂同样使用 P1 留 Z、缓存、固定常量编译和合法旁路。
- 一份 96×8×48 RF、一个 ready/写回队列与时钟、原 SR/SW/CR/CW 仲裁继续成立。Python 读取仅可作最终观测或参数绑定断言，不能成为未计费活操作数、第二 collector、第二 CR cache 或隐藏预测阵列。
- 高地址码与 scratch 必须处于原 128KiB SRAM 内，并在实际生命周期上不覆盖仍活的 PROJ；新系数 cold fill 与旧低地址镜像共用原 128KiB 系数池。

所有臂采用 P1/H48，并共同获得真实 H8 源驻留：Uq RF0..29、full-D Ug RF30..59、当前T10×H8源 RF60..69、门92/93/95。full-D U阶段最多73活向量，编码预测40..49与门50先死亡。按h0=0,8,…及块内k=h0..h0+7执行，保持原96个k的累加顺序；raw/expanded每t真实读取24B I24，code8每t真实SR64与signed8 ILOAD，再以同一个acc+source两RF读MAC消费。该普通供数加强不能计作门专属X。

Uc 连续48B向量从实际 CR 经原 cache48 装入临时 RF94，加至 U 后立即死亡。进入CSE时源60..69与门RF已死，可复用RF60..95并回收本hg Ug；CSE最大工作46向量加Uq30与其他hg Ug20，合计96。D归约结束后Ug死亡，V可占30..89。若Uc与D常量读取交错破坏原单字CR命中，应计实际额外读取。

不能沿用旧 U-only 实验的跨16个P1免费 Uc 常驻。实际局部链每 halo 有8个 P2 callback，间隔的 sparse-U/F/projection 覆盖 RF0..79。按当前连续缓存路径，一次30向量 Uc refill 为45次 CR×2槽加30次 load/等待×3槽，即180槽；H48比H24多8次 reload共1440槽，V少16×96=1536槽，静态净差约96槽/halo。临时 RF94流式 Uc 免去30RF常驻，保留 H48，无需据旧入口布局增加分块扫参。此估算不是实测结果。

## 已读实现的结论

`binding.py` 从旧 R24 ordinary/lifting_raw 导出执行真实 source、完整 K864 preview/sn2、Conv2/F/merge、projection gate，并只替换16个 anchor 的 U/V。回调参数没有 gold；gold 只在执行后用于输出比较。真实 updated 与门始终核原导出。返回 PED 必须与实际 SRAM 相同，再经过原 SR/DMA 外送，并将外送字节同返回值逐位置核对。未见从 gold 供应回调操作数的路径。原 P1 smoke 显式落盘 U，只用于绑定核验，不是随后 P1 留 Z 的性能分母。

旧父绑定对34项 live、除 PED U/V 外全部部署常量、geometry/source/preview/updated/projection捕获做逐项一致性检查，并实际执行源程序要求门0diff。preview历史FP32差分的 RMS 汇总最多允许2ULP，但整数端点与差分计数不放宽。这不等价于编码结果已通过。

`predictor.py` 的计算来自实际 CR 响应、门 RF 和累加 RF；绑定只保留静态非零坐标。H8门以两次真实 SR64 收集到原 staging 前16B，再有费 load 到 RF50。每个门列的空判定也付 issue；`IADD_GATE_CONSTANT` 使用累加与门两个 RF 读，实际 CR 提供 signed19 常量，结果沿原两槽 WB。D=0时不读门。该条件加法适用于二值门预测，不可复用为宽 Ug 乘法。该文件没有改原 IRNE，参数范围 helper 的无sat与48bit界已由运行入口实际断言。

`kernel.py` 中，真实输入经 `load_i24`、宽 `ISUB_REG`、原 `IRNE` 和 `ICLIP8`，压缩臂以真实 SW64 写 Q8；消费按原 k-major 顺序运行，每逢k%8=0通过 `gather_h8` 读取各t真实SR64并装入RF60+t，随后选择lane k%8。expanded 臂在编码RF立即scale/add/sat并写I24，再沿共同H8源驻留读取。实码地址、signed8解码、wait_reg/drain与RF寿命已核对，未见gold初始化码或重建输入的路径。

Uq 与 Ug 共用实际 U 系数响应，门经真实 H4 SR/ILOAD、付费源判定、T10 unpack 和逐tick判定后才发 IAAC；未将既有 projection 结果当作 Ug。连续 Uc 阶段先完成逐向量 scale/load/add，再调用 latent CSE，保留单字 CR 缓存语义。固定零基值臂直接绕过 predictor/cold fill，并合法折 U 指数。U complete、V complete、bias、最终 ISAT 的原顺序保留。Uc的局部48B收集与输入/预测collector生命周期互斥，可映射同64B staging；不能将这段Python局部数组另算为可同时使用的第二缓冲。

`run.py` 已在执行前断言范围 helper 的无sat24与48bit界，且对每臂 deepcopy 同一份实际前缀 Machine 的完整时钟、RF、ready、pending、SRAM与计数；这是同一已执行前缀的独立实验分支，不是把前缀结果作为零成本新状态。每臂最终服务仍含完整前缀。

`latent_cse.py` 已将两父完整D图适配为实际带spill计划。沿原完整算术图与固定顺序，使用确定的最远下次使用策略，保存与重载signed48向量；没有删公共节点或把中间值藏进host数值字典。旧 `required_RF_for_this_order=127/120` 仅描述同顺序无spill的容量，不能再写成完整图未适配，也不是图的必要RF下界。

每次spill通过原64B staging前48B、一次RF读取issue与6次SW64；重载通过6次SR64、collector issue、ILOAD与真实写回等待。工作RF映射最多36scratch加当前hg10个可回收Ug，其他hg的Ug保持到各自消费。运算由实际CR256中的128bit控制记录解码为原 `ISOURCE addsub`，每条只有两个RF读并检查移位操作数与结果的signed48界，没有signed19宽乘捷径。

| 完整CSE父 | 算术节点+输出累加 | 128bit控制记录 | 控制填充字节 | 每hg store/reload | spill SRAM字节 |
|---|---:|---:|---:|---:|---:|
| ordinary | 289+10 | 373 | 5984 | 36/38 | 1584 |
| lifting_raw | 283+10 | 355 | 5696 | 31/31 | 1248 |

已核读计划内置的定向smoke记录：每父240个结果0diff，保留未来hg输入；这不是独立重跑或完整halo通过。coef控制从73728开始、SRAM spill从98304开始，均与现有PROJ、Uc、CODE、RECON不重叠且位于原128KiB池。`load_constants` 已实际DMA cold-fill控制并绑定spill地址，每臂首次callback一次，随后三个hg每条控制都经同CR端口，不增加第二ROM。原source ROM容量与程序不变。

已向实现方指出编码端核验缺口：U/PED 精确比较不能单独证明 Q8/RECON 逐元素正确，因为 U24×96有零空间。后续 kernel 已补观察实际 x/g、按独立 reconstruct 计算参考，并逐P1比较 CODE或RECON SRAM的全部960值；这些观察数组未作为 encode/U/V 的输入，不增加硬件数据存储。最终每有损臂记录15360个 `observed_quantizer_values_compared`，编码端检查缺口已关闭。

## 最终35条数据核读

核读 `ordinary_corner_final.json`、`ordinary_interior_final.json`、`lifting_raw_corner_final.json`、`lifting_raw_interior_final.json` 和 `ordinary_interior_final_stress.json`：四份ready各7臂，加一份压力7臂，全部 `complete=true`。早期25条逐k gather供数记录不混入本表。

每条updated/projection gate/U/PED差分为0，实际PED SRAM外送字节检查开启，阶段槽数和等于总时钟，完整前缀与消费者槽数之和相同，SR/SW/CR/CW计数乘字宽与报告端口字节一致。每有损臂实际比较15360个码或重建I24值；每code8臂均1920次Q8 SW64及1920次 `common_Q8_H8_source_RF_load`，旧逐k gather计数为0。这些实际计数证明最终同用H8驻留实现，而非拼接旧性能表。

每full-code臂有48次CSE H8调用。ordinary实际17904条控制记录、10368次spill SW64、1824次reload ILOAD；lifting_raw为17040/8928/1488。算术节点与输出累加计数逐项等于整张计划乘48。cold-fill控制为1122/1068槽，ready完整CSE阶段为73872/65664槽，ordinary压力为85866槽；实际spill、重载、CR取指和等待都进入服务数。

以下均为包含真实共同前缀的总服务槽数：

| 父/halo | raw | fixed展开 | fixed码 | affine展开 | affine码 | full-D展开 | full-D码 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ordinary/corner ready | 2007870 | 2053950 | 2030910 | 2068632 | 2045862 | 2074612 | 2140493 |
| ordinary/interior ready | 2700748 | 2746827 | 2723787 | 2761510 | 2738740 | 2768652 | 2839011 |
| lifting_raw/corner ready | 1843800 | 1889880 | 1866840 | 1904562 | 1881792 | 1910000 | 1965835 |
| lifting_raw/interior ready | 2476800 | 2522879 | 2499839 | 2537562 | 2514792 | 2544476 | 2605712 |
| ordinary/interior stress | 2974170 | 3035450 | 2994906 | 3035802 | 3013690 | 3045050 | 3128282 |

每个ready halo中，fixed码内消费相对同函数展开省23040槽，affine省22770槽；压力分别省40544和22112槽。full-D码内消费相对自身展开反而多55835至70359槽，即2.41%至3.18%；压力多83232槽，即2.73%。同样采用完整CSE并实际收费后，当前full-D码内放置未获得同函数收益。普通码臂完整编码收费后仍比raw慢；raw与量化函数不同，不能据此宣称无损替代或精度Pareto关系。

最终所有halo的clip8计数均为0，重建sat24未触发；当前端到端数据不覆盖发生clip8的输入分支。clip8定义、原舍入顺序与参数全码域无重建sat24证明已静态核对，不因此声称对任意输入的全链测试完备。

## 未测范围与裁决限制

旧 probe 直接预填码和 xhat，只覆盖 U 入口；其普通 affine 4.31%不可搬作本阶段端到端收益。当前计划边界为真实局部生产者后完整16位置 PED U/V与输出，不含 native/globalBN/join，不是整层或整网，也没有新 AEE、训练、RTL全链、EDA/PPA或生产修改。

本阶段已实际执行完整常量CSE与有费spill，不能再以“未适配”代替负结果。但仿射折叠、H8驻留、P1留Z、常量CSE与spill本身均是普通执行技术；普通fixed/affine也获益，不能换名成为门专属新意。当前证据支持停止这份full-D码内放置的加速贡献句，保留普通Q8结果；不足以否定门预测或低位表示家族，更不补作未测AEE结论。

下一接口须改变已有证据中的实际成本：Ug累计、宽D图及其CR控制/spill，或编码与消费的真实边界；不应再以队列、比例或同放置粒度扫参代替改变瓶颈。若改变表示参数，须重新建立自身数值/质量与服务对照，不继承其他父或学生的AEE。
