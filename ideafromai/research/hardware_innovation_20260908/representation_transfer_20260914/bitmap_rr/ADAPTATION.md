忠实mode8已完成520条small、32条跨row short以及两64各16条冷暖/BP。两组冷mode2／4／21／8为787603／786421／772611／766786与831996／830886／810788／811381。后一集合mode8比count21冷慢593拍、暖慢1491拍，冷BP慢5610拍；该原布局保留。

实测的可改接口：mode8每K16按P/T枚举，已经付416bit读却只使用其中一个13bit P字段。同T其他P随后再次读写同一行，两64各41445／50403次条带读写。独立的Q2 selected-bank z读不在此数字中。原单context/208bit布局对此没有同样的四P全字权限。

唯一有界适配mode7在每K16内按T后P枚举，利用原52bit z_hold留住同T其他P字段。BM_SCAN从holding取当前P字段；各P仍用同八pop和同ALU完成，BM_STORE在同T还有P时付一拍holding提交，最后一P才占原z grant写全52bit字。下一T重新付完整z读。无新数据数组、ALU、pop或W/z端口，不省略holding提交拍。所有位置最后仍完整写z，Q2及raw原序不改；原mode8原枚举顺序保留。

增加固定T/P next-index mux、same-row比较与控制；每context的原416bit holding现在跨P存活。这是既有宽口权限下的具体行复用，未测mux时序和面积，不称新算法。预期只减少重复读状态和z争用，净窗口变化必须重新实测，不能从机会数推导RR周期。最终对2／4／21／8／7完整重跑small、short、两64以及同模型无reset跨输入集合切换；除此不加机制或扫参。
