# 四帧源门认证缓存：已执行，当前布局不晋级

**24条完整局部CPU链全部完成；自然72次暖查询没有命中。** 相对同函数强普通控制，exact memo总服务多0.163%，三级认证多0.432%–0.454%。这次停止的是两个固定像素、24个H8、三级各向同性半径的布局，不是全部帧间复用。程序、完整结果与逐帧表为 [execute.py](execute.py)、[summary.json](summary.json)、[frames.csv](frames.csv)。

| 四帧合计，CPU服务槽 | raw完整CSE | exact-I24 memo | 三级认证 |
|---|---:|---:|---:|
| ready | 10,838,630 | 10,856,294 | 10,887,842 |
| 固定背压 | 11,961,370 | 11,980,858 | 12,013,018 |
| 相对raw增加，ready | — | 17,664 / 0.163% | 49,212 / 0.454% |
| 相对raw增加，背压 | — | 19,488 / 0.163% | 51,648 / 0.432% |

每臂都使用真实matched dense stage320的四帧I24，执行完整源、K864 preview/sn2、Conv2/merge、projection gate、原U/V及PED/门出口。源门、preview、updated、U、PED和实际外送均0差；sn2、updated、projection gate和PED也与这四次实际GPU捕获逐值0差。没有训练或新valid825，不能将这四帧AEE作为新质量总体。

范围是固定interior halo的完整局部链；不是完整层、native/global BN或整网。ready背压分别保持一台Machine跨四帧，冷启动计入。每帧系数替换、当前I24 DMA和全部连续义务保留，故门相同不会错误地跳过当前连续值。

**没有命中的原因不只是参数太保守。** [独立离线解释](../capture_owned/opportunity.json)显示，72次同位置H8转移连真实80门位完全相同都为0；未量化的最大安全半径也0命中。实际I24最大变化中位90,786.5，证书安全半径中位374.5。单lane的T10门序列有239/576相同（41.49%），但同类L∞证书逐lane也0命中。不能靠增大三级阈值继续声称无损；新的方向须改变表示、可取消粒度或参考，而不是扫本布局的半径。

**真正命中分支另作了定向验证。** [12条source-only检查](directed_hits.json)使用同一实际源裁剪：重复输入exact/certified均24/24命中；人工每个I24加1时，certified命中21/24、exact为0，两种背压下实际门均0差。人工扰动结果不计入上述自然命中率或全链性能。

## 资源与费用

共同96×8 signed48 RF、128KiB状态、128KiB系数池、单发射、SR64/SW64/CR256及原两拍整数写回。历史`state[114688,121024)`占6,336B，含每项240B真实I24、16B门词、8B元数据；没有额外SRAM。证书320B常量放在`coef[126976,127296)`，避开124,544B preview原系数图，真实冷装/CR收费。普通后缀也使用最新P1/H48留U与H8源供数，避免回退到弱分母。

源原program前十条恰好全读RF0–9，查询复用这十个实际输入，失败不重复加载。历史读、差分、绝对值、T10和H8最大值归约、门余量、三级比较及存取均付费。查询元数据存RF94；三级选择用两槽、最多两RF读，并复用既有3B scalar collector，未免费增加第三读口。门回送复用原16B门collector/64B暂存。当前串行查询/刷新，无同entry并发读取，参考最后提交有效位；命中不刷新旧参考。

ready四帧三级认证额外SR 18,048B、SW 25,536B、CR 30,720B、CW 320B；真实源dot从Machine寄存器取得，四帧共7,680值与实际捕获S48一致。离线margin数组完全不参与调度。

新增ABS/MAX/AND/跨lane归约及比较按两拍CPU ALU收费，尚未综合其逻辑；同端口/容量不等于已证同面积。这份表不能称RTL加速比或ASIC PPA。

## 调试与复查

首个认证接线曾把常量放在96000，覆盖preview U；完整后继数值核对立即失败，日志为`setup_coefficient_overlap.log`，没有有效性能结论。改用实际空闲系数区后通过。独立审阅随后补出RF元数据持久位置和三级选择读口费用，exact/certified两种背压均已按最终代码重跑。最终表不混用修正前数字。

执行命令为 `python3.12 execute.py --mode {raw,exact,certified} [--stress] --frames ../capture_owned/000_*.npz ../capture_owned/001_*.npz ../capture_owned/002_*.npz ../capture_owned/003_*.npz`；使用带NumPy的`/opt/anaconda3/bin/python3.12`。完成后`check_hit_path.py`验证定向命中，`summarize.py`仅汇总六个最终文件。起始设计见[PLAN.md](PLAN.md)，代码独立审阅见[CODE_REVIEW.md](CODE_REVIEW.md)。
