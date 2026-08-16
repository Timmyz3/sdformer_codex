run_id:      verification_20260720_builder_projection_vplan
design_name: gatestack_builder_projection_single_context_top
tool:        文档级验证规划（Icarus Verilog、Verilator、SVA、覆盖工具待执行）
start_time:  2026-07-20T00:00:00Z
last_stage:  p0_bias_and_abort_directed_regression_complete

result:
  builder_projection_small: PASS_C0_C1
  real_s0_s3_c0_c1: PASS_466560_acc32_zero_mismatch
  accumulator_overflow_quarantine: PASS_zero_final_handshake
  synchronous_bias_req_rsp: PASS_full_acc_width_identity_reject_backpressure
  hatf_width_sweep: PASS_4_widths_4_stages
  coverage_closure: NOT_COMPLETE
  formal_signoff: NOT_COMPLETE

# 2026-07-20 BSF 验证闭环

- 专用BSF runner完整PASS：Icarus、Verilator动态SVA、真实HATF96 S0-S3、Yosys、Erie。
- 覆盖错误bias身份重试、32-bit正负大偏置、同步响应延迟、final backpressure和162-token复用credit。
- 四阶段共233280个acc32元素0 mismatch；bias请求2430降至15。
- 未覆盖完整Builder中resident装载后的mid-tile abort动态场景，coverage/formal sign-off仍未完成。

# 2026-07-20 等并行度结构基线验证

- 3xIndependent32三路不同tag/tile/channel/gate/weight/bias resident事务通过，384输出元素0 mismatch、0串扰。
- engine1/2独立weight和final反压均命中；Icarus、Verilator+projection SVA、Yosys、Erie通过。
- GateStack既有runner的Erie错误检测统一补充标准[ERROR]前缀，避免假PASS；bash语法检查通过。
- H67真实S0-S3、随机长反压、abort/reset和覆盖率仍未完成。

# 2026-07-20 DCTF Term语义验证

- Adapter 83周期发射9条合法command，覆盖8类错误、multi-beat/single、drain、collect/emit flush和backpressure，0 mismatch。
- Fabric Q2/3/4扩展元数据后synthetic周期402/391/387，随机flush语义下守恒与三bank逐项保序通过。
- Adapter/Fabric Icarus、Verilator动态SVA、Yosys、Erie均通过；EVENT_WAYS=2额外elaboration通过。
- bank compute completion、late response epoch和完整abort联动仍未覆盖。

# 2026-07-20 DCTF32 Bank Executor验证

- 单bank定向验证覆盖multi/single destination、奇偶Acc路由、weight/Acc反压、错误响应身份、错误command元数据和mid-term flush。
- 一条三destination term只产生一次weight request、三次Acc update和一次term_done；最后destination的Acc握手是compute complete边界。
- flush后用相同tag/channel/tile重发，先注入旧epoch响应再注入新epoch响应；旧响应被原子drop，`stale_rsp=1`且无Acc更新或term_done。
- 专用runner通过Icarus、Verilator动态SVA、Yosys和Erie 0 error/warning；三bank联动、完整Acc/bias/final及覆盖率签核仍未完成。

# 2026-07-20 DCTF96三Bank集成验证

- Q2/Q3/Q4 fabric随机回归分别402/391/387周期，260条accepted command，覆盖随机flush、full retire+accept和supertile sideband逐bank保序。
- DCTF96顶层覆盖相邻不同supertile重叠、三bank weight乱序/反压、六路Acc反压、奇偶路由、非法地址零副作用和三bank epoch ABA。
- 主代理独立复跑：Icarus/Verilator均PASS；每bank4次weight request、6次Acc update、3次term completion、1次stale drop，六parity通道各3次。
- head compute done只在head-last term三bank完成掩码全1时产生；完整Acc/bias/final、真实trace和功能覆盖率仍未签核。

# 2026-07-20 Accumulator Flush验证

- flush专用TB覆盖读改写中断、bias/final反压中断、组合屏蔽、同tag重启、valid隔离、counter保留和overflow恢复。
- 主代理独立复跑Icarus与Verilator动态SVA均PASS，`quarantined=1 overflow_cleared=1 recovery_finals=2 counters_preserved=1`。
- Yosys check与Erie 0 error/warning通过；既有single-head、multihead和G1基础回归通过。
- 未验证整tile final quarantine；合法数值域上界或tentative commit协议仍是系统P0。

# 2026-07-20 Projection Acc数值范围验证

- 新脚本逐channel重算S0-S3中间部分和与最终输出，并比较现有RTL金参考，四stage mismatch均为0。
- 实际最大绝对final为S3的55035；全INT8配置级最坏界为50233425，32-bit裕量42.75x。
- Python单元测试2项PASS；sample0实际分布不能替代更多trace覆盖，但配置级上界与样本无关。

# 2026-07-20 DCTF96完整Projection协议验证

- 定向TB覆盖双head、多destination、zero-term、三bank错峰weight/bias、六final独立反压、并发source_done、wrong-current、长flush、旧epoch和同tag恢复。
- Icarus与Verilator动态SVA均为`heads=4 terms=2 finals=18`且PASS；Yosys 0 process，Erie RTL/TB 0 error/warning。
- SVA实际发现并推动done error锁存修复，未通过放松断言掩盖问题。
- 真实S0-S3、长随机回归、functional/code coverage和formal仍未关闭。

# 2026-07-22 DCTF96真实S0-S3验证

- Icarus S0-S3全部PASS；S0 Verilator动态SVA PASS且与Icarus同为822周期。
- 真实四stage共233280个acc32逐元素零失配，所有输出head的162 token无重无漏。
- 逻辑term/物理weight/bias计数均与生成器manifest一致；stale、protocol error和overflow为0。
- generator单元测试3项PASS；仍缺多sample、随机存储延迟、final拥塞覆盖和formal。

# 2026-07-22 DCTF-2C验证

- 2C adapter定向TB覆盖双context不同物理sideband、collect/emit重叠、command反压、duplicate malformed term原子丢弃、sticky error清除和flush清空。
- 动态SVA检查validated-head可见性、context不覆盖、fill所有权、精确idle、flush清空、反压稳定、head-last边界、command sequence连续和error sticky。
- H67 S0-S3完整projection Icarus全部PASS；S0 Verilator同时绑定adapter与full-top SVA，周期与Icarus同为764。
- 四stage 5010逻辑term、15030物理weight请求、7290 bias请求、233280 acc32比较全部守恒且0 mismatch。
- 1C term datapath、1C完整projection和2C adapter三套runner独立复跑exit 0。
- 尚缺长随机多term/多flush、随机SRAM latency、final sink拥塞分布、functional/code coverage、formal和门级回归。
- 追加128个连续合法term确定性随机反压压力流：696个压力destination，加定向用例共704条command，命中139拍collect/emit重叠、235拍反压和8-bit sequence回绕；Icarus/Verilator计数一致。
- 长合法term流已覆盖；剩余缺口收敛为随机多flush与连续malformed、随机SRAM latency、final sink拥塞分布、functional/code coverage、formal和门级回归。

# 2026-07-22 非法Metadata Drain验证

- 两类非法metadata均以标准ready/valid握手被消费，随后last event通过隔离drain；adapter保持idle，issued/weight/Acc副作用为0。
- 新增5条动态SVA属性并在Verilator中命中，term datapath全流程通过。
- 完整projection定向回归PASS；2C真实四stage周期仍为764/718/5356/47072，233280 acc32零失配。
- 连续malformed、多flush随机压力与formal liveness仍未关闭。
- 首轮复审提出的非空洞解锁、2C混合并发、clear/error竞争、零destination与drain中flush均增加定向TB/SVA。
- 1C/2C分别通过Icarus与Verilator+SVA，共4组零失配；2C场景证明一个合法context在途时非法输入可被独立消费且不增加issued/fabric状态。
- 结果包`results/gatestack_dctf96_illegal_metadata_fix_20260722`保存输入SHA256、工具版本、日志索引和结构化mismatch字段；仍缺随机长malformed/多flush和formal。
- 第二次复审指出2C只检查隔离、未完成原合法context；已增加drain后恢复执行，三bank逐项核对合法term的身份、目的token、product、weight/Acc/term-done守恒。
- 最终2C Icarus/Verilator+SVA为126/121周期、issued=5、completed=4/4/4、mismatch=0；多出的1个term仅为合法恢复测试，非法term继续保持零计算副作用。
- 第三次独立复审PASS，非法Metadata活性修复P0/P1/P2 OPEN均为0。

# 2026-07-22 PPDI Executor叶模块验证

- 覆盖split pair、same-cycle pair、only-even、only-odd、跨command product复用、malformed、partial flush和stale epoch。
- 动态SVA逐端口检查反压稳定、parity、done-mask不重发、全部commit后cmd_ready、term-done精确边界和flush屏蔽。
- Icarus与Verilator动态SVA计数一致：71非reset周期、5 command、5 weight、Acc 4/4、4 done、1 stale、0 mismatch。
- Yosys与Erie通过；等待独立叶模块审阅，未进入集成或coverage/formal签核。
- 首轮独立审阅的stale/valid负向实验已纳入正式TB并通过；新增child clear、paired continuation、odd-first和done-mask因果SVA。
- `EPOCH_W=3`满8项pending-generation定向证明fail-closed与drain恢复；不再仅依赖有限回绕假设。
- 真实Banked Acc共同flush：旧partial=1/0，恢复token2 final逐lane只含新product，bias4、update/write6/6，双模拟器+双模块动态SVA通过。
- 结果包现固化所有日志及日志SHA256；等待独立复审。
- 第二次复审的identity P1已补回归：错误tag的pending response不会清generation，正确epoch+tag+channel+tile后才释放；Acc stalled valid全程保持。
- clear SVA增加“无同拍本地新错误”前提；结果manifest改相对路径并补四份build log，目录内`sha256sum -c log_sha256.txt`可独立验证。

# 2026-07-29 Local5 双模式验证

- Verilator 全回归通过：数值256向量、Direct/TARE row、MFEP/bridge、T450
  地址、window4、window16、三窗口 line buffer。
- TARE row 遍历全部32种五候选mask：80个有效edge与80个issue守恒，
  ZERO/SPARSE/DENSE为38/22/20。
- score-to-term sink 使用确定性随机反压；TARE与Direct均输出388条相同命令。
- 新增绑定SVA：anchor/command反压稳定、分类不超发、stencil完成时分类守恒；
  TARE/Direct两版均PASS。
- Icarus独立运行TARE row与score-to-term均PASS；TB握手改为negedge驱动，
  消除posedge阻塞赋值竞争。
- `DEST_W=9`实际访问destination 449；不是仅编译检查。
- Erie对新增SVA和顶层接口PASS；旧row/Shiftmax因项目外部规则禁止function和
  参数化for-loop仍需waiver或后续展开，不标Erie全RTL签核。
- 未关闭：随机SRAM latency、active reset长压力、coverage、formal、
  target-library LEC和门级回归。
# 2026-07-29 Local5 独立审阅整改回归

- clean Verilator 主回归 21 条 PASS，无 `%Error/FAIL/TIMEOUT`；新增 T450
  72000 term/naive 动态计数边界。
- Direct/TARE 动态 SVA 通过，新增 done-stall tag 稳定和完成退休前禁止下一 anchor 属性。
- 正式 Icarus runner 覆盖 TARE row 与 Direct/TARE score-to-term。
- Phi/Prosperity 双线模型 14/14 单测通过；Local5 缺 ordered tail、Motion T450
  仅外推时正式 tail contract 均阻断。

# 2026-07-30 Local5 T450 全链验证

- Local5 主功能回归累计22个PASS日志；T450全链在Verilator与Icarus均通过。
- 新增投影SVA：CLEAR期间禁止命令、逐行地址递增、末行转RUN、命令反压稳定；TARE、Direct、T450三组动态SVA通过。
- Verilator T450 lint和Yosys bank-local T450参数化检查通过。
- Erie对参数化for循环存在既有规则误报；独立lint bind/TB因缺被绑定模块依赖产生外部工具warning，不计PASS，也不计RTL功能FAIL。
- 缺口：随机SRAM latency、多窗口fullres ordered trace、functional/code coverage、formal、门级LEC。

# 2026-07-30 Local5 T450 最终协议回归

- T450参考改为独立Python整数模型；两窗口共检查3600项Acc，覆盖重复目的、多mask、正负权重和第二窗旧值清除。
- 新增CLEAR期间非法读、权重未装完早启动、DONE后非法目的500读三个负向场景，均按fail-closed合同通过。
- 最终parity、SVA、Icarus cross-sim、Verilator lint、Yosys和14个模型单测全部PASS。
- 27项完整编译依赖、脚本、TB、SVA和向量SHA256已写入`results/local5_t450_signoff_20260730/full_input_sha256_final.txt`。
- 最终接口复审未发现当前参数范围的新P0/P1；真实fullres ordered trace、同步SRAM和目标PPA仍未关闭。
