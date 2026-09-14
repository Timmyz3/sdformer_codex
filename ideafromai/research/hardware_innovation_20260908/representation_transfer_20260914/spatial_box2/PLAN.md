# Box2：同moment函数的源侧共同因子强控制

先合同后RTL，只写本目录，旧locked臂只读。冻结moment `g=[a,a+b,b]`，把共同 `(1+x)` 移到Q1前：由实际相邻门产生0/1/2计数，直接计算 `E_j=Z_j+Z_{j+1}`；Q2只存原g0/g2，`y0=aE0+bE1,y1=aE1+bE2`。不改moment系数、函数、量化或质量，也不声称新公式。

静态全binary输入：E范围[-17124,14854]，含全部Q1前缀，signed16充分；Q2两tap任意归约前缀绝对值≤140756698，signed32充分。Q1使用唯一8×32ALU、bit16断carry形成dual16，count2通过实际signed W左移1的接线及每字段0/W/2W选择器进入ALU，不增加乘法器或第二加法器。

source仍1920B，每C实际读16个原生10bit字进20B窗口，再按ky取原8个source掩码。六个非padding位置的OR/AND组合网络构造count非零/count2，两个padding位置为0；仅保存两个正在更新的2bit count，不存完整count数组，也不接受TB count。CHECK拍构造pending，TIMESEL取count，均实付。Z采用原8bank×40row×32=1280B，实际承载2Y×3E列及第四padding列；单公共读地址/单读写授权，原地累加，最终实扫E支持。

Q2原3tap缩为[a,b]两tap：384个N8向量、4992B原13bit物理权重，cache每lane16词/总208B。p仍15360B；8个19×13乘法器对E16符号扩展后乘a/b，原native跨stripe psum读回和最终ordered480行保留，不增加恢复阶段。原FP32 identity→J20→wide64→I24 consumer资源/端口不变，全部配置、source邻接供数、count构造、支持扫描、缓存、psum、背压及启动收费。

从ordinary factor派生，先15 small功能/角落/BP/无reset两遍，再两64与18序列36tile。CPU同时由原g/Q1/source重算Z→E、由0/1/2直接Q1重算E，再比较两tapP、原moment展开W和新I24；跨序列只借不变上游source/identity，不借旧Q11输出。最终与同moment3/ordinary/general按完全相同输入及gold对账。不扫参、不训练、不GPU、不EDA、不Git。
