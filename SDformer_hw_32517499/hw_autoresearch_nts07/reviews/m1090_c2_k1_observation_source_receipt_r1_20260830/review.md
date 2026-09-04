# M1090：C2 K1 observation-only mapped-gate diagnostic source receipt

M1090 在冻结 M1058 K1 顶层外增加纯观察 wrapper，把既有的 `header_accept`、`raw_accept`、busy/protocol/fault，service group/request/response/context/result 计数，以及 adapter bundle/bank request/response、live-slot 计数导出到顶层。22 个 `obs_*` 端口均只从现有功能/调试信号扇出；wrapper 不增加时序状态，观察信号不进入 M1058 实例，也不反向影响任何功能输出。

短窗口 mapped TB 固定使用 case 0。header 接受后每拍打印一次 stage，逐一对全部 22 个观察端口执行 `$isunknown`；第一个未知值立即 fail closed，窗口严格限制为 128 拍。TB 和 runner 不启用 SAIF、VCD/FSDB、toggle dump 或随机寄存器初始化。未来 runner 的执行顺序固定为：先消费独占 attempt，再 fresh DC，再 mapped VCS compile，再执行唯一一个 128 拍 case-0 diagnostic；失败必须进入新隔离目录且不自动重试。

由于已有 M1089 编号属于 final-checkpoint rebind 审阅，原 C2 observation 草稿已完整迁移为 M1090：模块、token、wrapper、TB、filelist、合同和 namespace 均不再占用 M1089。现有 M1091 runner 只引用 M1090 source，并等待不同作者 M1092 的精确外层封印与 GO token；已有 checkpoint M1089 审阅目录未修改。

本作者只执行 Python 3.10 静态检查：合同/双层 sidecar、16 项 filelist、22 个纯扇出观察端口、runner AST/顺序/namespace、M1080 DO-NOT-RETRY、旧 M1089 C2 精确路径不存在、M1091 attempt/result 不存在、docs/359 身份均通过。没有执行 DC、VCS、PTPX、GPU 或远端任务，也没有消费 attempt。因此本收据不提供 mapped-X 定位、周期、时序、面积、功耗、能量或系统加速结论；下一步只授权不同作者 M1092 source hammer。
