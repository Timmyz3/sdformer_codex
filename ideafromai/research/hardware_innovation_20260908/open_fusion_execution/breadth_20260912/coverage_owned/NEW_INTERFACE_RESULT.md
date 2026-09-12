# NRV∩W 共享索引的实际接口重放

**本轮补试了旧私人索引实现未覆盖的同址合并和共享缓存，但当前索引格式仍未胜最强普通路径。** 补齐逐次 NRV 消费读口后，普通路径不从合并获得完成时间收益；因此不能把减少 W 字节直接写成加速。这个结果停止当前固定布局作为新性能候选，保留不同共享索引/原生生产接口的研究空间。

先写的 [PLAN.md](PLAN.md) 把 B 定为“私人索引为每个 ID 重读同一行元数据”，把最强 A 定为“dense 同样享有共享/缓存/队列权限”。[shared_index_probe.py](shared_index_probe.py) 实际执行 6 个控制：dense/private、dense/merge、dense/cache、index/private、index/merge、index/cache。首次 4 臂后追加无 cache 的 merge 控制来拆开合并和阻塞填充；未扫 cache 大小。自核发现预重读 NRV 后使用 Python 列表会隐藏消费期间的读口，已改为逐条物理 NRV 响应并全部重跑。**旧预读模型的约 7.86% dense 改善不成立，已撤回；下面只有最终有费版本。**

真实输入来自 `psn/rtl/gp_slice/intersection_cases.bin`：C384、16 位置、2 输出 W 行、全 T10，2 tile×4 ID。32 个实际学生表示用例来自 **16 个不同的输入/权重选取**，各有 class/time 两种表示；12 个定向表示用例覆盖零/尾部/交集等条件。实际选取包括原整数化学生的 v000/v006 共 12 个 block 捕获，训练过的 row2of4 两个捕获与 C16 两个捕获。它们不是本阶段最新 R24/lifting 学生，本次没有跑新 AEE。与 ready/stress、6 臂组合后共 528 次执行，并非 528 个独立样本。

最终共同资源与收费如下。源配置是本次 CPU 边界新指定的有限预算，**不是旧 RTL 四个 1KiB 源 bank 的同资源延续**。

| 项目 | 六臂共同权限/实际收费 |
|---|---|
| 源存储 | 4×4KiB，每 bank 2R32/1W64；576B packed code + 最多1536B NRV，解码窗口最多128bit |
| 源扫描 | 两个32bit口拼接一个64bit响应，实际请求/返回及12bit码解码；源值由原3bit码重建，不输入 gold NRV |
| NRV 生成/消费 | col9+code12 装入32bit记录，实际64bit写入；W阶段每次iterator请求一条32bit NRV，1拍响应。每bank两个32bit读口分别服务两tile，每tile/ID至多一条请求/拍；不预取整表到免费寄存器 |
| W 存储 | 共同2KiB/tile；dense行384B，sorted-index行 `u16 nnz + nnz×(u16 index,i8 W)` 最多1154B |
| W 物理口 | 每tile两个8bit口，1拍返回、II=1；每ID一个逻辑待决请求，四ID轮询；返回可带4bit owner mask |
| 合并 | 从最多四个待决地址中形成最多两个不同地址组，只对相同字节地址广播；dense/index同权限。地址比较逻辑没有做物理综合 |
| 缓存 | 固定16行×4B；64B data、18B tag、2B valid、8B等价LRU年龄，共92B/tile；冷启动、全相联、每拍一次查找。4B miss付四次W8读取，两拍填充，另付commit槽；填充期间不查找 |
| 控制状态 | 两边预留共同128B/tile，覆盖四ID游标/当前码和索引、待决请求、两口返回和fill/owner状态；最大1154+92+128=1374B仍在2KiB内。Python容器是这些有界状态的功能模型，非真实宏或布局 |
| 压力 | 固定32拍中后8拍阻塞源/NRV/W读取；ready主表，stress只作功能与环境敏感性。源NRV写入为共同固定前缀，不模拟写压 |

端点是 **源扫描、NRV写入前缀 + NRV/W供数前端完成槽**；两 tile 因明确分配的读口并行，取较慢者。S 累加、PSN、T10门均计算作功能核对，但未计其服务时间；没有把 CPU 墙钟、旧 RTL 总周期或后继 BN 费用拼进此表。表中的槽数是该离散服务模型的结果，不能称硬件实测周期、整局部链加速或 PPA。

主表取 ready、class 表示，按 12/2/2 个实际捕获等权平均。time 表示与压力臂在完整 JSON 内逐例检查。

| 模式 | 原整数学生：槽 / W8读 | row2of4：槽 / W8读 | C16：槽 / W8读 |
|---|---:|---:|---:|
| dense private | **1343.75 / 2427.33** | **1471.00 / 2966** | **1471.00 / 2966** |
| dense merge | 1343.92 / 2310.67 | 1471.50 / 2775 | 1471.50 / 2775 |
| dense cache | 2157.92 / 989.33 | 2291.00 / 872 | 2291.00 / 872 |
| index private | 2842.17 / 8425 | 2192.00 / 4568 | 2223.00 / 4505.50 |
| index merge | 2745.75 / 7587.83 | 2118.00 / 3574.50 | 2102.50 / 3520 |
| index cache | 6167.67 / 4689.67 | 3270.50 / 1748 | 3126.50 / 1486 |

最优 index/最优 dense 完成槽比为 **2.0433× / 1.4398× / 1.4293×**。index merge 相比自身 private 缩短 3.39% / 3.38% / 5.42%，仍未越过普通分母。dense merge 的槽数反而微增约 0.012% / 0.034% / 0.034%；在该服务模型中普通路径受到逐次 NRV 返回节奏约束，W 合并减少流量不足以缩短端点。cache 明显减少 W8 读取，但每拍一次查找和阻塞填充增加服务时间；这只约束当前冷、16×4B、blocking 配置，不证明所有 cache 都差。

[shared_index_results.json](shared_index_results.json) 保存 528 次逐例结果；[shared_index_summary.json](shared_index_summary.json) 与 [CSV](shared_index_summary.csv) 保存汇总。检查 **118272 个 S 值、168960 个门 bit，差异均为0**；同时验证 present 和源码往返。128 个实际 private 臂的 W8 读取数与旧 scalar RTL TSV 精确一致，源扫描 **64bit等效**读取数也一致。本次源实际用2R32，因此该计数核对不声称旧/新源宏或总周期相同。

仍未收费或闭合的部分是：外部源/W冷 DMA与配置、源码生产者、S/PSN/后继消费者、真实 SRAM/控制逻辑面积与频率。程序、系数、阈值只参与功能 gold 比较，不在此前端执行预算中。原行压缩在加载前构造，压缩生成成本未计；dense 同样从驻留行开始。因缺这些边界，本次只能支持“这个新共享接口在当前资源模型/旧真实捕获上仍未胜普通前端”，不能支持完整原作、当前学生质量或芯片性能结论。

复跑：

```bash
cd /home/zhumd/work/sdformer_codex/ideafromai/research/hardware_innovation_20260908/open_fusion_execution/breadth_20260912/coverage_owned
/opt/anaconda3/bin/python shared_index_probe.py
/opt/anaconda3/bin/python summarize_probe.py
/opt/anaconda3/bin/python build_coverage.py
```

下一项最小实验是保持原生生产/消费者和源口预算真正闭合的流式共享头或跨ID游标复用；当前 fixed cache 的负结果不能直接关掉它，也不能把旧私人索引的8bit元数据重读税说成所有原始稀疏算法的固有下界。
