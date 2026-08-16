# Head-major PSUM Spill公平下界与架构决策

## 一、要回答的问题

GateStack采用output-tile-stationary：一个output tile的AccTile跨所有head驻留，然后切换到下一个output tile。它需要重放head payload，但避免把32-bit dense partial sum在head之间写回大容量SRAM。

公平对照是head-major：每个head只decode一次，再依次更新所有output tile。其代价是每处理完一个head就spill尚未完成的partial sum。

## 二、可综合最小调度器

新增`gatestack_head_major_spill_scheduler.sv`，实现理论上最有利于head-major的事务序列：

```text
for head:
    decode head once
    for output tile:
        for token bank batch:
            head 0:        write psum，不read
            middle heads:  read psum -> write psum
            last head:     read psum -> bias/final，不write
```

定向TB使用3 head、2 tile、6 token、2 bank：

- decode 3次；
- spill read 12次；
- spill write 12次；
- final 6个batch；
- spill value traffic 768 bytes；
- Icarus与Verilator/SVA均在55周期通过；
- Erie lint为0 error/0 warning；
- Yosys结构代理为161 generic cells。

调度器只生成事务和计数，不执行projection算术，因此它是traffic/cycle下界，不是完整bit-exact head-major核。

## 三、H67四stage真实Trace下界

参数为`TOKENS=162、OUT_TILE=32、BANKS=2、ACC=32 bit`。首head免read、末head免write，得到：

```text
PSUM_capacity = output_tiles * tokens * OUT_TILE * 4 bytes
minimal_spill = 2 * (heads - 1) * PSUM_capacity
```

| Stage | H/Tiles | 全PSUM容量 | 最小spill | GateStack payload | decode-once payload | payload节省 | spill/节省 | 最小事务周期 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 3 | 60.8 KiB | 243.0 KiB | 0.86 KiB | 0.48 KiB | 0.38 KiB | 648.0x | 1215 |
| S1 | 6 | 121.5 KiB | 1215.0 KiB | 0.09 KiB | 0.09 KiB | 0 | N/A | 5346 |
| S2 | 12 | 243.0 KiB | 5346.0 KiB | 5.06 KiB | 1.20 KiB | 3.87 KiB | 1382.4x | 22356 |
| S3 | 24 | 486.0 KiB | 22356.0 KiB | 67.68 KiB | 4.97 KiB | 62.71 KiB | 356.5x | 91368 |

这是乐观下界，尚未计入head decode后descriptor/event buffer的跨tile重放、权重、bias、算术和控制停顿。

## 四、架构决策

当前证据支持保留output-tile-stationary：

1. GateStack重放的是低位宽稀疏payload；
2. head-major spill的是32-bit dense partial sum；
3. S3只为节省约62.7 KiB payload，最少增加22.36 MiB psum流量；
4. head-major还要求486 KiB全tensor psum容量，而当前AccTile只需一个output tile；
5. 161-cell控制器不是瓶颈，SRAM容量和读写能量才是关键。

这不是“output-stationary普遍优于head-major”的通用结论，而是H67当前维度、稀疏表示和32-bit accumulator合同下的workload-specific判断。

## 五、论文用法

可以作为架构消融图：

- 横轴：stage/head数；
- 左纵轴：payload replay bytes与minimal psum spill bytes，对数尺度；
- 右纵轴：片上PSUM容量；
- 标注：spill/payload-saved比例。

贡献表述应为：

> GateStack以compact final-gate payload重放换取dense partial-sum驻留，真实H67窗口下head-major的最小spill比其节省的payload高356.5x至1382.4x，从而驱动output-tile-stationary AccTile数据流。

必须注明`[真实trace统计+RTL事务调度+理论下界]`，不得写成目标SRAM能量或完整RTL性能。

## 六、剩余工作

- 用目标SRAM宏读写能量把bytes转成pJ；
- 为head-major加入descriptor/event replay buffer后重新计算；
- 若审稿要求完整数值基线，再复用projection backend实现bit-exact版本；
- 当前更高优先级仍是Adaptive format metadata/residency、目标PPA和valid825量化。

## 七、入口

- RTL：`rtl_hitflow/gatestack_head_major_spill_scheduler.sv`；
- TB：`tb_hitflow/tb_gatestack_head_major_spill_scheduler.sv`；
- 回归：`sim_hitflow/run_gatestack_head_major_spill_checks.sh`；
- 报告：`results/gatestack_head_major_spill_20260718/report.{md,json}`。
