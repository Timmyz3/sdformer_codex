# 固定C2：系数响应内W8解码

2026-09-13。延续`../literature_owned/CANDIDATES.md`的C2。这里只跑ordinary/interior ready及既有固定stress两个点，每点expanded16、旧RF84、response decode三臂。不得追加权宽、层、缓存或延迟扫描。

**B：** 将每H8 W8展开从“CR响应→RF84 signed16→RF读回16B权重暂存”改为“同一CR响应→有费选择/符号扩展→同一个16B权重暂存”。U32 MAC、行scale的两次16×24乘法、原U/V RNE/sat及bias全部保留。

**A：** MiLo等的有费打包/解码和普通流式权重缓存；不是新量化算法或MiLo GPU复现。**待证X：**这个具体放置减少共享RF/issue往返后，在同资源完整局部执行下是否超过同函数expanded16；若普通也可用则作为共同供数底座，不强称lifting专属。

**参数：** `stage_20260912/weight_compensation/ordinary_lowbit_gpu_parameters.npz`的`W8_U/W8_code/W8_scale_q16/W8_V`，逐字段核对同路径`algorithm/weight_controls/aee/ordinary/W8/deployed_constants.npz`。这是已验证旧R32 W8函数，不是C1 R24 activation量化、当前新320步学生或native_w8。

**边界：** 复用`breadth_20260912/hardware/run_packed.py`：实际I24→preview/sn2→完整K864/Conv2/merge→projection门＋连续U32/V96 PED。同点共同producer实际执行一次，然后复制其全部Machine、状态、时间、pending与仲裁给三臂；不导入gate值、不加旧表。native/globalBN/join不在此边界。

**固定延迟假设：** 响应字已到后，每个H8展开占一条有费decode issue，结果从issue起经过2个槽才可用，期间第二个槽作为显式等待；不与MAC/其他issue重叠。没有“解码零代价”假设。实现逻辑是32B响应中的4选1×64bit选择、8个signed8→signed16扩展、已有16B暂存写使能及valid控制；没有综合/STA来验证此2槽假设，不据此报PPA。

**相同容量：** CR256仍单个32B响应；64B共同staging中`[0:24)`给source gather，`[24:40)`给既有H8权重，剩24B不另用。权重必须保留到这一H8所有P2/T10 MAC读完，下一次decode才能覆盖；source gather只能写前24B。96×8×48 RF、SR64/SW64/CR256、128KiB state/coeff及8192B源ROM不变，scale RF80..83/split85..86/source88..91/header94沿用；新路径不写RF84。

**最强控制：** expanded16与旧RF84按同真实W8函数、相同权重常量cache/普通零旁路/输出处理。两旧臂必须逐count复现保存结果，避免通过改计费方式制造胜出。新响应路径另核源gather与权重保留不相互覆盖。

**停止条件：** 若新路径仍慢于expanded16，停止这个2槽响应内放置的加速主张；若连乐观免除全部新增decode槽也无余量，不追加decoder实现或延迟优化。负结果不否定W8量化、MiLo或所有响应接口。可保留它作为正确放置，但不启动EDA/训练或新AEE。

Root负责代码审阅；CPU正确性不替代响应选择逻辑的RTL功能/时序证明。本目录为唯一写入目录，所有Python导入禁写bytecode；无GPU/EDA/main.tex/hash/git。
