# 真实FC1 → PSN → H96门接口

从冻结support_fc1独立fork，前级固定普通mode4内容去重/96lane条件加法及原24词/4描述符预取；输入为真实g′的320×96bit门，不由TB提供Y。三臂仅PSN选择native/full/cert，producer/外存配置/源/最终96gate接口共同。借入fused_datapath已经实测的一拍符号头＋逐plane组合两加＋界比较；不同时连接source classifier。

持久Y唯一320×96×24＝92160B，原FC1 bank-valid懒清零及转发保留；FC全部请求/两个写回级排空后交给PSN，同一个2304bit共同行读地址，无第二份90KiB、无交接复制。fused按每P10次真实读取形成2880B暂存，然后12个H8×T10组。最终逐t门依次输出前，以实际120B gatepack保存本P的12组；每组80bit有使能写，全部覆盖后才可读10行96bit，消费者拒绝保持，最后输出后才能复用。native原生H96输出不承担无用转排；拥有同全部资源预算。

A200B、tau5760B、flags144B只存一份；沿native密排配置：A13＋D12＋tau360＋flags9＋class3＝397个128bit请求/响应词，绝不由fused重复加载。冷命令全读并使LUT失效；warm仅在同模型重复且配置已有效时跳过，跨模型必须cold。fused在首次使用时真实构建64表行＋21个P/N拍；native无无用构表。跨模式warm若LUT未就绪则实际构建。固定A模型不在计算时免费供TB答案。

共同96个48bitALU：前级FC1、native累加、fused第一层80lane和冷表/P-N分时共用。fused额外第二层80＋上下界160＋tail20＝260个48bit加减位置，总356；native原96个16×24mult仍留共同预算。native/full/cert由同module选mode，未裁剪闲置硬件。单份1280B多mux表、160 signed比较、变长移位、指数选择等沿fused显账，无同频/同面积/PPA结论。native的U5760B仅供native归约；fused完整U从真实80lane组结果观察口检查，不为TB增加第二份U转排存储。cert不承诺U。

测试2real＋5诊断，然后32real＋原诊断，三臂×ready/BP×cold/warm，连续换源/换mode。TB从真实S/W独立重算每个Y、A@Y和门；实际每个Y读值、native/full U、最后96gate和所有握手计数检查。保留原32real，诊断分列。整命令计冷加载/FC/构表/PSN/转排/output/go/done；bytes/state/stall及相同边界clock break-even均列。无需训练/EDA/生产修改。
