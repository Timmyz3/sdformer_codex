# fused_datapath 独立审阅

本次只读实际 `subset_psn.sv`、`tb.cpp`、`implement.py`、PLAN、SUMMARY 及父输入合同，并用 Python 3.12 独立核对 296 行 JSONL；没有重跑 RTL、综合或时序。审阅时本目录尚无 RESULTS/README，数字以 SUMMARY 与原始记录为准。未发现当前固定 A、输入和握手合同下的功能错误。

## 数值与真实执行

`subset_psn.sv:110–115` 确实把两个子集和、前缀和上下界放在一个寄存边界内：GROUP 得到 `−L0−L1`；每个 PLANE 得到 `nv=2v+L0+L1`，同时用 `nv·2^m+N(2^m−1)` 与 `nv·2^m+P(2^m−1)` 判界。当前 plane 的 nv 被比较，没有误用上一拍 v。`tail_delta:90–91` 在 GROUP 初始化 tail，在非末 plane 按 `(tail−P/N)>>>1` 递推；当 m>0 差为偶数，算术右移精确。20 个 tail 值按 t 共享八个 h，依赖的是同组共同 e/m，当前成立。

正 gain 判 `U>=tau`，负 gain 判 `U<=tau`。负 gain 下 lo>tau 才锁 false，hi<=tau 锁 true，等号处理正确；constant 在 GROUP 预锁并优先。full 只在 m=0 用最终值判门，cert 全 80 门锁定才提前输出。cert 的 out_u 只是未完成前缀观察值，不能作为完整 U 消费；本 TB 也只为 full 比较完整 U。

实际 Y 经 320 次 2304bit 接口写入 y_mem，再经 320 次实际行读进入 T10 ybuf；e、符号和逐位表地址均在 RTL 由 Y 形成。A、tau、flags 的单 128bit 供数为 cold503 / warm490 词；两半 LUT 的 64 行和 P/N 均实际构建。没有 TB 赠送指数、plane、表或提前判决。

固定 A 的所有 5 项子集和已经过导出 admission，范围 [−19657,11516]，signed16 截存成立，不能扩大为任意 signed16 A 均适用。48bit 前缀/界比当前量级宽裕。e 按 max-abs 位数定义，signed24 最小负数会得到 e=24，符号头再处理 bit23..0 可精确重建；全零走 e=0 的一拍哑 plane。首批 296 命令只实际覆盖到最大 |Y|=1554；新增 8 个命令已补齐 e=24、signed24 两端点及负 gain tie，见文末复核，不再列为未试缺口。

## 资源与比较公平性

| 显式 RTL 位置 | 核对结果 |
|---|---|
| 80×2 prefix add/sub | 两级串行，每级按 carry-in 实现加/减；前 10 个第一级在表/P/N 构建期复用。 |
| 80×2 bound add | lo/hi 两路并行，接在 nv 和变长移位之后，与 prefix 不分时共享。 |
| 20 tail subtract | 与上述 320 位置独立；GROUP 还使用 20 路变量左移初始化 tail。 |
| 其他组合成本 | 80 路 prefix 变量左移、两路 80 signed less/equality、80 门归约；单份 1280B LUT 具有 20bank×8 个读 mux，另有 H8 和 tau 选择网、运行时指数网络。 |

因此 340 是显式 48bit 加/减**位置**数，不是综合后的精确算术单元数、更不是面积或“同 80 ALU”。删除 dot_hold 480B 和下界 flag 20B 的容量变化与源码一致；Y90KiB、T10 holding2880B 等主存储未消失。新增长组合路径至少包括 LUT mux→两级 48bit add→变量移位→界 add→比较→全锁定归约；本轮没有时序证据。

full/cert 在同一个可配置核上共享这套电路、端口和 cold/warm 合同，full 不再走无用的四个判界状态，所以是本轮有效因果对照。P/N 冷初始化 21 拍对 full 也执行，是共同可切换模型初始化的明确小开销；不能据此声称专用 full 设计也需要该逻辑。旧 80ALU 父核与 native96MAC 都是异资源参照。

## 记录与周期口径

独立核对 296 个唯一任务，逐病例 planes、early_groups、参数/Y/表访问与父核全部一致；被消掉的 SIGN/PREFIX/四个判界状态计数均为零，PLANE 状态数等于实际 planes。总状态和等于 cycles，PINIT..OUTPUT 状态和等于 psn_service，所有 SUMMARY 聚合与 JSONL 相符。gate/Y 各 9,093,120 值、full U 4,546,560 值、上下界 45,317,120 次、指数 113,664 组检查。请求地址/valid 与输出门/U/地址拒绝保持已由 TB 核；Y offer 在接受前持续保持。warm 只验证紧邻 cold 后保留同 A/LUT/P/N，阈值/flags/Y 仍重装。

32 real 是 8 个已投影源 tile×4 H96 切块。ready full/cert 的 PSN service 为 148195 / 97424，节省 34.26%；post-Y cold 总计含 go 为 193411 / 142640，节省 26.25%。父证书 428572→97424 是增加组合资源、改变寄存边界后的结构适配成功，不能把它说成原 80ALU 免费加速。

native96MAC 的 118344 只是原核 PSN 状态分项。97424 比它少 17.68% 周期，**不等于完整链加速**。即使暂只比较这两个 PSN 分项，时间盈亏也只能写
`T_fused/T_native < 118344/97424 = 1.21473`
（等价 `F_fused/F_native > 0.82323`）。没有 native 已达 3ns 的依据；该比值也不包含实际 FC1 交接、打包及其它状态，不能外推完整链。

## 新颖性与唯一值得补的接口

当前独立创新暂评 **3/10**。两半 distributed arithmetic、signed bitplane 前缀、区间证书、tail 递推及 Claude T10 同拍判界思路都是借入；完整 BitL 未迁移。新增价值是把实际 Y/参数供数、H8×T10 粒度、运行时指数与寄存边界接成可计费的 RTL，并明确修复父四拍判界的负迁移。这是有用的实现结果，尚不足以单独支撑新方法标题，频率/面积未知也不能靠周期表抬分。

最有价值的下一接口是实际 FC1 最后写者→此核的 T10 Y 就绪→H8×T10 门结果按原 H96/T 顺序提交：加入真实有限打包存储、占有/覆盖和背压，并让原生 96MAC 与新 full/cert 享有相同 Y 驻留与参数保留。当前 80 门组尚未接原 96 门行消费者，不能免费假定组装；这条完整交接完成前，保留本叶正结果，暂不晋级完整子链性能主张。

## 附言：signed24 端点诊断已补齐

后续实读新增 README、prepare_fullrange.py、fullrange_inputs.json 和 results_fullrange.jsonl，并用 Python 3.12 独立核对 8 条记录的状态/服务/访问/检查数；没有重跑。它是 **1 个新增诊断×full/cert×ready/BP×cold/warm＝8 个命令**，不是 8 个独立真实病例。直接 Y 域包含 −8388608、8388607、0、±1 及跨 t 交替，384 个 H8 组实际 e 全为 24。真实固定 A 下独立 int64 MVM 的 U 范围为 [−125825203769,112176373235]，64 正/32 负 gain、6 constant，850 个 U==tau，其中非 constant 负 gain 等号 240 个。

full 9216 planes、cert 4714 planes/348 提前组，与独立高位 MVM 包络模型及实际记录一致；新增 gate/Y 各 245760、full U 122880、上下界 3016960、指数 3072 次核对。合计 304 命令，gate/Y 各 9338880、full U 4669440、上下界 48334080、指数 116736；原 32real 聚合没有混入该诊断，前述周期与新颖性结论保持。signed24 端点实测缺口已闭合，但这不是从 binary FC1 产生的分布输入，也不是穷举所有 signed24 组合；实际 FC1 交接、原 H96 打包及阈值供数生成仍按前述边界处理。
