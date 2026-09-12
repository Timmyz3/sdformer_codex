# 一个实际 phase 消费接口：固定双臂结果

当前 signed9 z 跨 SRAM 的放置没有收益。ordinary/corner/ready 的完整真实局部链中，phase 臂比同表、同 H8 融合的强控制多 **38,400 槽**，消费者服务增加 **6.5651%**。全部 CODE/RECON/U/PED 检查精确。结论只针对这次有物理16bit码存读的接口，不否定条件表示家族，也不追加无 spill 或其他格式配置。

本轮先实现后运行，仅两个新臂，无失败重试。使用旧 ordinary R24+onepass、已有 full_g_q8 参数及原 Q16 U/V，真实 `binding.run_prefix` 执行 I24→源门→完整 K864 preview/sn2，再同 Machine 运行完整 Conv2/merge/projection 门、实际量化、U/V 和 PED/gate 外送。native、global BN、join 未计；无新训练、AEE、RTL、EDA 或生产修改。

## 同函数强控制与归因

两臂共用一张完整表、g=0 的 c 旁路、同 H8 源驻 RF、原 U MAC 与 P1 keep-Z/H48 V。共同融合使刚重建的 I24 直接留在 RF60..69 被 U 消费，**两臂都不写 RECON SRAM**。

1. `table_RF_reconstruct`：实际 x 减实际表 b，原 RNE((x−b)/2048) 与 clip8 得 q；同 RF 执行 2048q+b 和 sat24，直接接 U。
2. `phase_z16_SRAM`：同一 q 编码后，从实际 b 向下除2048得 a，执行 z=q+a；z 必须按通用 signed9，以物理 signed16 写 CODE SRAM 再实际读取。同 H8 的 b 保留，取 phi=b&2047，RF 内执行2048z+phi及sat24，再接同 U。没有第二遍表查，也没有将两条 U 矩阵当作必要条件。

原 `ordinary_corner_final.json` 的 `full_expanded` 同函数强控制保留 Dg 预测和 RECON SW/SR；原 `full_code8` 保留完整 CSE。这里只读取它们的已有结果，并逐项确认新执行 prefix 的服务、counts、checks 与该文件一致，不重跑、不改其结果。因此新 table-RF 相对旧 Dg 的变化同时包含**查表供数与删除 RECON 存读的共同融合**；不能将它称作 phase 专属收益。phase 的增量只能用本轮同表 table-RF 分母裁决。

## 固定布局与付费模型

| 资源 | 实际分配 |
|---|---|
| coefficient SRAM | 原低地址镜像0..41023不改；LUT=[98304,131072)，1024行×32B，每行10×signed20 b共200bit |
| LUT供数 | 每H8先实际读取row0、十次广播c到RF40..49；真实gate由两个SR64响应、collector与ILOAD写RF50。逐lane收费选择/零检测；非零gword经实际CR256，逐t提取signed20并写对应RF lane |
| CODE SRAM | [114688,116608)，一P1的10×96×signed16；每8码是两次SW64及两次SR64，未凭实测signed8范围缩位 |
| U阶段RF | 累加RF0..29、b RF40..49、gate RF50、临时RF51、当前H8 x/q/z/重建RF60..69，最多52个同时活向量 |
| V阶段RF | 原U RF0..29保留；直接调用原 `kernel.v_stage`，H48输出RF30..89，共90个活向量。此时b和H8源已死 |
| staging | 共64B；原source区域[0,24)，gate collector[0,16)，CODE pack/collector只用[24,40)，保存断言检查两侧未改 |
| 原池与端口 | 96×8×48 RF、128KiB state、128KiB coefficient、SR64/SW64/CR256各一端口、单issue；原512×128 ROM，LUT未放进ROM |

新增 `ILUT20_BROADCAST`、`ILUT20_LANE`、`IFLOOR11`、`IPHASE11` 四操作，分别承担真实CR字段提取/符号扩展/广播或RF lane更新、负数向下取整、低11bit提取。每次沿原共享issue与RF写回模型，断言写回标记为issue+2；依赖等待与仲裁实际执行。**两槽是写回时延假设，不代表每个串行依赖操作只占两服务槽**：原 `wait_reg` 会推进到写回真正提交后，因此本次多数串行依赖有一issue和两等待槽。新增位选择和运算逻辑未经RTL/STA验证，不能据此报PPA。

输入、b和所有算术操作数来自实际SRAM/RF/CR响应。预先构造LUT属于参数编译，完整冷fill实际收费；host读取只用于最终检查，不喂码、不喂重建或updated。原 q 的RNE/clip8先执行，随后才加floor(b/s)，因此未触及直接z编码的half-tie parity陷阱。两臂保留原U/V的RNE、bias、sat边界。

## 已执行结果

真实共同 prefix=1,472,190 槽；16个anchor、192个H8消费块。

| 指标 | table-RF直接重建 | z16 SRAM＋phase | phase增量 |
|---|---:|---:|---:|
| 同 Machine 局部总服务 | 2,057,099 | 2,095,499 | +38,400（+1.8667%） |
| 完整消费者服务 | 584,909 | 623,309 | +38,400（+6.5651%） |
| SR64字节 | 2,727,360 | 2,758,080 | +30,720 |
| SW64字节 | 1,043,360 | 1,074,080 | +30,720 |
| CR256字节 | 3,018,048 | 3,018,048 | 0 |
| CW256字节 | 198,336 | 198,336 | 0 |

实际表冷fill为6,144槽、32,768B。两臂均有192次c广播行和261次非零lane行选择，共453个逻辑行事件；原一响应缓存使物理表读取为417个CR256。1,536个真实gword中1,275个经收费零检测后旁路，c广播与非零lane写值计数相同。新操作写回断言分别检查4,530与8,370次。

phase 的额外费用有完整时间线解释：它用17,280槽进行a/z计算和16bit真存，15,360槽真读并装入RF，23,040槽提取phi并重建；替换了控制原来的17,280槽RF直接重建，净增38,400槽。每个T/H8八元素向量对应净增20个已执行服务槽。最终增量中SR/SW各增加3,840次真实64bit事务；没有把这些端口字节仅做静态代数换算，也没有将净增槽数全归于总线spill。

每臂实际RF q8和RF RECON各15,360值、U3,840值、PED15,360值全0diff；phase额外从真实CODE SRAM读取的15,360个signed16 z也0diff。binding另核验每臂updated/projection gate各61,440值及真实PED外送。完整结果、逐H8时间线、每次表选择/响应计数在 `phase_execution.json`，过程见 `phase_execution.log`。

## 与已有结果的边界

| 已有同点控制（未重跑） | 总槽 | table-RF相对变化 | phase相对变化 |
|---|---:|---:|---:|
| 普通fixed_code8，同Q16 U/V | 2,030,910 | +1.2895% | +3.1803% |
| 普通affine_code8，同Q16 U/V | 2,045,862 | +0.5493% | +2.4262% |
| 同函数full_expanded，Dg＋RECON | 2,074,612 | −0.8442% | +1.0068% |
| 同函数full_code8，完整CSE | 2,140,493 | −3.8960% | −2.1020% |

table-RF相对旧full_expanded少17,513槽，同时减少RECON的46,080B SR和46,080B SW；它的收益归于本轮共同供数/融合组合，尚未单独区分LUT与算子融合的贡献。phase虽胜旧full_code8，却输同表强控制以及已有full_expanded，因此没有phase专属增量。普通fixed/affine同Q16、最佳已测码格式和原公共优化权限都保留；不以较弱的旧全D放置作唯一分母。

停止当前“signed9 z跨SRAM后再phase解码”放置的性能主张，不再运行无spill臂。在现有相同位宽PE上，把z/phase完全合回当前H8 RF会回到两臂共享的精确RF重建边界；本轮没有证据将这种共同融合重新命名为phase贡献。对一般条件表示的其他电路/训练接口，本结果不作结论。

复现命令：`PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /opt/anaconda3/bin/python3.12 phase_execution.py > phase_execution.log 2>&1`。只产出固定ordinary/corner/ready两臂，不含格式、缓存、层或压力扫描。
