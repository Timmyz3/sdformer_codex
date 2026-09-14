# Q11：同函数 output-stationary AAC → 原 FP32 identity → I24

已用真实 [Q11 因子](../../spatial_winograd_inputs/factors.npz) 的 `expanded_int32` 实现完整普通OS分母，复用 [os_core.sv](../os_core.sv)，没有新增量化或微机制。W系数和aQ40/bQ20从该Q11函数重新导出，135个真实tile的raw/J/wide/I24独立重算后与其gold全同；原始spike的任意部分和绝对值≤39865935，signed32充分。Q13记录独立保留，不继承其AEE。

完整接口经 [os_stream.sv](../os_stream.sv) 接到原 [i24_consumer.sv](../../spatial_r16_rtl/i24_consumer.sv) 和 [wide_phase_alu.sv](../../spatial_r16_rtl/wide_phase_alu.sv)，后两者只读编译。DUT输入只有native C96×4×4低10bit源和原始IEEE FP32 identity；实际执行 `J=sat32(RNE(identity·2^20))`、signed64 `wide=p·a+((J+b)<<20)`、`I24=sat24(RNE(wide/2^26))`。没有预先赠送J或其他连续中间值。raw、J、wide与I24的3840个值、原ordered480行与最后包全部实际比较；所有结果背压保持。

| 64tile第一次连续遍历 | start→最后I24 | 冷service（含实际配置与启动） |
|---|---:|---:|
| held ready | 2493120 | **2601944** |
| disjoint ready | 2897388 | **3006212** |
| held BP | 3083251 | 3192075 |
| disjoint BP | 3635427 | 3744251 |

新增18序列、每序列首帧固定edge128/interior9664，共36个live tile，顺序和文件/原点见 [sequence_manifest.json](sequence_manifest.json)。实际source经两级因子与独立expanded-W计算、原FP32 identity经J/wide/I24均与根代理capture零差；ready/BP及两遍无reset实际回放再全过144命令552960 I24（raw/J/wide各同数）。36tile首次ready `c_cycles=1116324`、冷service **1182084**；BP为1356493、**1422253**。独立raw基本服务1029024，consumer实接增加87300=36×2425周期。全部Q11结果累计 **716命令2749440 I24**，不是仅在原单帧两个区域内比较。

15 small（8真实、zero/one/random/tail/rank正负/图外poison）加两套64，各执行ready/BP、无reset两遍不同tile，共 **572命令、2196480个I24，raw/J/wide也各同数，全部零差**。[CONSUMER_SUMMARY.json](CONSUMER_SUMMARY.json) 保存紧凑完整汇总，`consumer_{small,held,disjoint}_{0,1}.jsonl` 每记录一行保存原始结果。[verify_consumer.py](../verify_consumer.py) 逐项复核OS的source/W/bitmap/psum计数与所有状态，以及consumer固定服务和周期恒等式。ready时每tile消费者末输出比raw基线增加2425周期；这来自实际连线运行，不是把末段周期事后相加代替RTL。

配置W10368拍、consumer a/b24拍，仅第一tile付；每tile另source1536拍、origin1拍、start1拍。冷64为 `Σ c_cycles + 10392 + 64×1538`。256bit配置总线静态传332544B，source/origin每tile49184B。消费者每tile真实读取480个256bit FP32 identity行、480个raw行、24个系数行，执行480次8-lane32×32乘法、960次8-lane64bit加法、480次RNE和转换，再写480行输出。consumer保持原单context、8×32×32乘法器和8×64宽链，未借给producer；其全部端口与factor/融合臂相同。

Producer仍为8×32ALU、1920B源、15360B p、物理1280B bitmap、单256bit W读与原同背压；bitmap由actual source构造，source每P重读、全部构表/供数收费。W容量331776B，与factor的较小因子存储不同；同执行端口不证明总面积相同。OS的同拍异步W读取与加法未做时序验证，不声称等Fmax。详细源/W/bitmap/psum bytes及普通OS控制边界见 [Q13 OS报告](../README.md)，源事件相同不代表Q11/Q13函数或质量相同。

复现：`/opt/anaconda3/bin/python3.12 -B ../run_consumer.py --function q11` 完成导出、build、small；通过后运行相同命令加 `--streams`。跨序列再执行 `prepare_sequence.py`、`../run_consumer.py --function q11 --streams --sets sequence`，最后 `../verify_consumer.py --function q11 --sets small held disjoint sequence`。原始单帧source/origin/identity采用指向现存capture fixture的链接，跨序列只导出实际需要的36个输入，保留原文件/原点信息；没有任何TB赠送中间值。完整网络AEE由根代理质量报告另述，此处只报告同函数硬件与独立数值验证。
