# D2 Q2小组distributed arithmetic：实现前合同

B：R8 Q2虽全缓存，仍每个非零z rank一次MAC。将R8分两组R4，为每N8构造 `L[g,m]=Σr∈g m_r Q2[:,r]`（每组16项），signed13 z按最小公共signed宽度展开，正位加、最高符号位减。全K第一因子仍由最新native窗+双P packed产生，不由TB喂latent。

完整A：小组LUT distributed arithmetic、有效位枚举、符号修正均是经典方法。强对照14为同资源最新cachedOS八MAC，候选15 DA。两侧共同8×32bit加减、8×19×13乘、8×32bit移位、2×16×N8×signed18 LUT（576B）与LUT/vector端口。LUT构建使用相同八ALU，每N8两组零项初始化+30个非零表项，不能离线免税。系数最大4×32768，signed18充分。

每位置额外208bit z向量读及13位展开/支持编码计拍；去掉超过最小signed宽度的冗余sign平面，其余按组4bit mask枚举并付LUT读/移位/add。表内整向量零同样过滤，控制保留Q2整rank向量零和位置支持。完整480输出/配置/源、权重和输出背压保持，无中间RNE。

失败数：构表每N8固定32拍，位平面通常比最多8个rank多；应实测停止这个布局而非DA家族。14旧fixture112runs，不开位宽/组宽扫描。
