# FC1→PSN→H96 接口独立审阅

实读实际 `joined_fc1.sv`、TB、生成脚本、prepare/inputs、SUMMARY，以及最终 444 条主记录和 36 条生命周期回放。只用 Python 3.12 独立重聚合、核计费，没有重跑 Verilator/综合。**未发现当前固定模型与 host 冷/热配置合同下的功能错误；原 post-Y 的实际 FC1 交接与 H96 门打包缺项已在此叶补齐。** 这是 isolated Verilator 下的完整子算子链，不是源分类器、整网或同面积 PPA 通过。

## 唯一 Y 与真实生产交接

DUT 输入为 320 行真实 g′，TB 不提供 Y。FC1 沿共同 mode4 内容去重、24 词 payload/4 描述符和原 96lane 条件加法实际产生 Y；静态 D×W response 表虽由软件编译，实际表值请求、返回、消费者 routes 和累加均由 RTL 执行。TB 另以原 S/W 直接点积验证 Y，再以 A×Y 验证 U/gate；没有用表输出循环证明自己。

持久 Y 只有 `y[320][96]` 一份 92160B。FC 两级 fvalue 实际写回，8 bank 的 yvalid 控制未写部分读零，转发覆盖尚未提交的同 row 更新。退出 FC 同时要求 jobs/qcount/active/fv/pending 排空，才让 native 或 fused 接管；没有把末次写回留在 PSN 读之后。共同 `y_read_addr/y_read_data` 只有一处数组读取表达式，FC、native PREAD/POUT、fused FYREAD/FOUT 分时选择；FC 可与流水写回并行，不能称整个存储只有单一读写合并口。

新增 TB shadow 逐次核对实际读到的已提交 Y，以及每次有效 bank 的部分写回值/row/mask。累计 123564192 个实际读观察值、91169280 个实际部分写值核对；读观察包括 invalid-bank 的应有零值，物理有效 bank 计数另列，不把所有观察值当有效存储流量。Y_write_cycles=FC updates，bank×12 标量写数、共同读使能周期和最终 Y 均闭合。

## 参数与 warm 生命周期

三臂统一实际密排配置：A13+D12+tau360+flags9+class3=397 个 128bit 词；同一份 A200B、tau5760B、flags144B。已经消除了旧 post-Y 两个48bit一词的 tau480 与 flags10 等格式差异，fused 不再自行重取这些参数。

warm 仅在 config_valid 且 hblock 相同、host 承诺 A/D/W 对应 class/tau/flags 未变时合法。RTL 没有模型身份/version 检测，所以 **同 hblock 换模型仍必须 cold**。cold 失效化 LUT；native 不做无用构表；warm 首次从 native 转 full/cert 时，若 LUT 未就绪仍实际构建64表行+21个P/N周期。动态 routes、Y valid、队列与流水状态每命令重新建立，不把暖配置当成 Y 内容可复用。

已读 36 条 swap 回放：3case×两BP，各依次 native cold→full warm→cert warm→native warm→full cold→cert warm。实际配置词为 [397,0,0,0,397,0]，表写为 [0,64,0,0,64,0]，独立核对通过。这验证同模型跨模式生命周期；不代表在线参数修改的检测或一致性协议。

## 真实 120B 转排与数值

`gatepack[10][96]` 是实际 120B 状态。每次 FSTORE 为全部10个 t 各写当前 hgroup 的8位，12组共覆盖完整960位，然后 FOUT 才按 t 输出十行96门。hg 在每 P 由0递增至11，FOUT 完成前不进入下一 P；因此无需物理清零旧门，旧值不会在覆盖前被读出。每命令确有384次80bit组写和320次96bit接受；输出拒绝时 row/gate/Y/U 都保持。

fused 的完整 U 只在80lane组观察口核对，未新建一份 U 转排存储；cert 不承诺完整 U。生产接口只把 H96 gate 作为共同终点，out_y/out_u/mon 为验证观察，不要求下游消费这些宽观察总线。

full/cert 数学沿用已核的符号头、两半 DA、prefix、同时界判定与偶数 tail 递推。负 gain 的 <= tie 与 constant 优先保持。三臂同 S/W/A/tau/flags 与共同 mode4 前级；各病例 FC updates/jobs/coeff/config 与 CPU 独立期望一致，full/cert planes 和 early 与独立高位 MVM 模型一致。固定 A 的16bit子集表 admission沿用父叶，当前源/权重也满足24bit Y、48bit U 域。

## union 硬件与费用

共同第一层96个48bit加减位置由 FC、native累加、fused前80lane和构表/P-N分时使用；新增第二层80、界160、tail20，共 **356 个显式加减位置**。因此相对原native增加260个位置；不是额外再放340，也不是总共320。native 的96个16×24乘法器、4级 product holding、U5760B、yhold288B与原门比较路径全部保留在 union。fused 的160路界比较是新增路径，不能说成全模块仅有160路比较。

新增数据数组为 LUT1280B+ybuf2880B+v480B+P/N/tail240B+gate/locked20B+gatepack120B=5020B，另有60bit指数、控制和组合逻辑。LUT 一份但有160个16bit 32:1读 mux；A/tau 一份也不等于单口 SRAM 免费提供跨 t 的80项，实际寄存器选择网络须算资源。变长移位、第二级加法、比较/归约和共享输入 mux 的长路径都未测时序。

外部池为8192×128bit=128KiB、8bank各一笔在途；它不包含上述 Y、U、routes、LUT和本地 holding。真实 source/gate 各320×96bit，配置/系数请求每命令完整计数。native/full/cert 同一个未裁剪 union 模块，所以模式间周期对照成立；它不证明专用native与专用fused等面积、等Fmax或能耗。

## 最终记录与比较

主444命令为37case×三模式×两BP×cold/warm，另36生命周期命令，共480；smoke84不重复并入下列覆盖。gate14745600、native/full U9830400、最终/缓冲Y观察24576000、上下界50269440次核对。独立核总周期=全部21状态之和，PSN/build/FC分项、参数/系数词、Y读写、pack与先验计算量全部相符；未用 CPU 工作量替换周期。

下表是32real（8个源tile×4个H96函数）**完整命令 cycles**，已包含 go/config/FC/build/PSN/pack/output/DONE；不能再加一拍 go，也不能与 `source_to_gate` 或旧叶 service 混比。

| 日历 | native | full | cert | cert 对 native 节省 |
|---|---:|---:|---:|---:|
| ready cold | 184408 | 227219 | 176448 | 4.3165% |
| ready warm | 159000 | 199091 | 148320 | 6.7170% |
| BP cold | 212997 | 255588 | 204698 | 3.8963% |
| BP warm | 165340 | 205189 | 154330 | 6.6590% |

ready cold 有30个病例正、2负，BP cold有31正、1负；warm两日历均32正。full 仍慢于 native，cert 对同 union full 的 cold ready/BP 节省22.34%/19.91%。三臂同日历/冷热的 FC 状态、系数/config和实际 Y 写 bank逐病例相同，差异来自后级执行与其输出日历。

真实转排使 cert PSN 从父叶97424变为107664，恰多32命令×320行输出费用。与 native 比的整个命令频率盈亏门槛应为 `Fcert/Fnative>0.95683`（ready cold）、0.96104（BP cold）、0.93283/0.93341（warm）；不能继承叶级0.82323。它们是固定所测周期日历下的条件比值，不是实测Fmax，也没有native已达到某个ns周期的前提。

## 判读

本轮已经完成“真实 FC1 最后写者→唯一 Y→T10供数→H8/T10门→H96输出”的具体缺口。周期正结果可以保留，但整链冷收益只有约4%，共享组合路径若降低频率会很快吃掉它。这里的完整链止于此 FC1/PSN 子算子，未接 source classifier、FC2/shortcut 或整网质量评估。

独立新颖性仍暂评 **3/10**：DA、界证书、Claude同拍判界思路与普通存储/转排/驻留都属借入；新增是收费接口与执行粒度适配的闭环，不因接通后为正而自动成为新标题。当前适合留为下一阶段的可测底座，不能用单次 Verilator 通过替代物理时序与整网证据。

