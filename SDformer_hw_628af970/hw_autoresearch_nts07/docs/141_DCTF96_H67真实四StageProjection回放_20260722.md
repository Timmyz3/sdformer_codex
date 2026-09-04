# DCTF96 H67真实四Stage Projection回放

## 1. 目的

小规模协议TB只能证明控制正确，不能证明DCTF适合H67真实workload。本轮把H67 sample0、window0的S0-S3真实K/gate、INT8 projection weight、acc32 bias和expected output转换成直接term/event向量，在完整DCTF96 bank-local projection顶层上回放。

比较边界严格冻结为projection execution slice：

~~~text
真实K/gate -> term/event序列
                    |
                    v
 DCTF96 term fabric + 3x weight/product + 3x Acc + bias/final
                    |
                    v
            expected acc32逐元素比较
~~~

该边界不含typed-slot和decoder，因此后续Central96与3xIndependent32也必须使用同一term/event输入，不能把decoder复制代价混入第一张性能表。

## 2. 向量与数据质量

生成器`generate_gatestack_dctf_real_trace_vectors.py`执行以下检查：

1. 校验源NPZ的SHA256；
2. 固定S0/S1/S2/S3为3/6/12/24 heads；
3. 按`gate_code -> lane -> token list`生成term；
4. 重新计算`activation @ INT8 weight + acc32 bias`金参考；
5. 检查所有结果不越过signed int32；
6. 为term、token、weight、bias和expected文件记录SHA256；
7. 单元测试3项全部通过。

S1真实K trace全零，因此term和weight访问为零。验证仍必须执行全部bias和final，不能把“全零输入”错误处理成“无输出tile”。

## 3. RTL实测

三路weight与bias端口采用独立固定一拍同步存储模型，六路final均ready。每个逻辑supertile对应三个物理32-lane bank。

| Stage | Heads | 逻辑supertile | 周期 | 逻辑term | 物理weight请求 | bias请求 | acc32逐元素检查 |
|---|---:|---:|---:|---:|---:|---:|---:|
| S0 | 3 | 1 | 822 | 62 | 186 | 486 | 15552 |
| S1 | 6 | 2 | 718 | 0 | 0 | 972 | 31104 |
| S2 | 12 | 4 | 5652 | 652 | 1956 | 1944 | 62208 |
| S3 | 24 | 8 | 55072 | 4296 | 12888 | 3888 | 124416 |
| 合计 | 45 | 15 | 62264 | 5010 | 15030 | 7290 | 233280 |

结果说明：

- Icarus S0-S3全部PASS；
- S0 Verilator动态SVA PASS，周期与Icarus同为822；
- 233280个acc32元素零失配；
- 每个输出head的162个token无重复、无缺失；
- stale weight/bias、protocol error和accumulator overflow均为0；
- 逻辑term总数5010、物理weight访问15030，与文档131公平合同一致。

复现入口：

~~~bash
bash sim_hitflow/run_gatestack_dctf96_real_trace_checks.sh
~~~

详细中文结果位于`results/gatestack_dctf96_real_trace_20260720/实测汇总.md`，机器可读结果为同目录`实测结果.json`。

## 4. 对架构的直接指导

### 4.1 稀疏收益必须按stage分账

S1是纯bias/final主导；S3则由4296个term及其多destination执行主导。单一平均density无法指导统一流水，论文至少要报告stage级term、destination、weight访问和尾部bias占比。

### 4.2 DCTF的真实优化对象

DCTF没有减少物理weight bit读取：5010个逻辑term仍对应15030次32-lane bank访问。其潜在价值是：

- decoder/term只生成一次；
- 三bank本地计算，避免中央768-bit返回join和96-lane product总线；
- bank可独立反压；
- Acc/bias/final保持六个物理窄端口。

因此必须用时序、互连活动和面积证明价值，不能把逻辑请求减少三倍写成weight能量减少三倍。

### 4.3 Bias尾部仍是确定开销

15个逻辑supertile共产生7290次32-lane bias请求。S1零term仍需718周期，说明逐token bias/final是低计算密度stage的主要下界。BSF虽能减事务，但现有flop驻留面积代价过高；后续只允许bank-local SRAM/RF或bias-as-initial-state方案参与能量评估。

## 5. 证据边界

- 只覆盖sample0、window0，不代表100-frame分布；
- INT8权重与acc32 bias沿用候选部署合同，不能替代valid825精度冻结；
- 固定一拍存储模型不是SRAM编译器时序；
- final全ready，尚未给出真实sink拥塞的p95/p99；
- 当前只有DCTF结果，Central96与3xIndependent32同边界结果完成前不能计算speedup或EDP；
- 没有DC、STA、SAIF、SRAM宏、LEC或布局布线。

## 6. 下一步

1. Central96使用同一term/event向量、相同一拍三weight bank和六final物理工作量；
2. 3xIndependent32使用三个独立term client，明确三读typed-slot的面积/能量成本；
3. 比较wall time、每stage瓶颈、物理访问与逻辑面积；
4. 再接真实SRAM延迟和bank skew扫参；
5. 只有DCTF相对Independent达到完整projection EDP至少15%，或面积与能量均至少10%改善，才晋级DATE主贡献。
