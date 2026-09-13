# 原始来源与精读表

截至2026-09-13，12份原论文全文已取得并精读所列核心章节，另1份ASNA-Flow主文未取得。未运行外部代码。正式venue、arXiv日期和读取版本分开记录。

|编号|原始来源、venue、日期|精读定位|代码核验|用途与限制|
|---|---|---|---|---|
|1|[End-to-End Learning of Representations for Asynchronous Event-Based Data](https://openaccess.thecvf.com/content_ICCV_2019/papers/Gehrig_End-to-End_Learning_of_Representations_for_Asynchronous_Event-Based_Data_ICCV_2019_paper.pdf)<br>ICCV 2019<br>2019-10|§3.1–3.3; PDF pp.3–4 / printed pp.5635–5636|[官方仓库已核；学习表示代码；不是完整光流硬件](https://github.com/uzh-rpg/rpg_event_representation_learning)|测量/核/采样定义；极性与时间投影的信息损失|
|2|[Event-based Asynchronous Sparse Convolutional Networks](https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123530409.pdf)<br>ECCV 2020<br>2020|§3.1–3.2; PDF pp.4–8|[官方仓库已核；同步/异步SSC实现](https://github.com/uzh-rpg/rpg_asynet)|SSC不等于普通卷积；rulebook/活动变化/非线性状态|
|3|[Lightweight Event-based Optical Flow Estimation via Iterative Deblurring](https://pure.tudelft.nl/ws/portalfiles/portal/220695337/Lightweight_Event-based_Optical_Flow_Estimation_via_Iterative_Deblurring.pdf)<br>ICRA 2024<br>2024-05-13 to 2024-05-17|§III-A–F / Algorithms1–2; PDF pp.4–6 / printed pp.14709–14711|[官方仓库已核；训练/评估和ID/TID配置](https://github.com/tudelft/idnet)|当前批次ID vs 跨时间TID；合法先验与warm-start费用|
|4|[Efficient Meshflow and Optical Flow Estimation from Event Cameras](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_Efficient_Meshflow_and_Optical_Flow_Estimation_from_Event_Cameras_CVPR_2024_paper.pdf)<br>CVPR 2024<br>2024-06|§3.1–3.2.3; PDF pp.3–5 / printed pp.19200–19202|[官方仓库已核；模型/训练/评估/权重链接](https://github.com/boomluo02/EEMFlow)|稀疏平滑meshflow不同于dense flow；CDC是稠密细化|
|5|[EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_EDCFlow_Exploring_Temporally_Dense_Difference_Maps_for_Event-based_Optical_Flow_CVPR_2025_paper.pdf)<br>CVPR 2025<br>2025-06; arXiv 2025-06-04|§3.1–3.4; PDF pp.3–4 / printed pp.1986–1987|[官方仓库已核；模型/训练/评估/权重文件目录](https://github.com/KK-xi/EDCFlow)|高分辨率差分与低分辨率匹配互补；warp依赖上一迭代flow|
|6|[EMatch: A Unified Framework for Event-based Optical Flow and Stereo Matching](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_EMatch_A_Unified_Framework_for_Event-based_Optical_Flow_and_Stereo_ICCV_2025_paper.pdf)<br>ICCV 2025<br>2025-10|§3.1–3.2; PDF pp.3–5 / printed pp.5847–5849|[官方仓库和主文链接已核；未运行](https://github.com/BIT-Vision/EMatch)|无事件位置仍需上下文；TRN/SCA产生稠密对应表征|
|7|[From Contrast to Consistency: Rethinking Event-based Continuous-Time Optical Flow Estimation](https://openaccess.thecvf.com/content/CVPR2026/papers/Hu_From_Contrast_to_Consistency_Rethinking_Event-based_Continuous-Time_Optical_Flow_Estimation_CVPR_2026_paper.pdf)<br>CVPR 2026<br>2026-06; arXiv 2026-05-25|§3.1–3.3; PDF pp.3–5 / printed pp.15127–15129|主文与arXiv均未找到官方代码链接；不等于不存在|VWE保持时间结构；局部结构/轨迹一致性；双向完整窗口依赖|
|8|[SpiDR: A Reconfigurable Digital Compute-in-Memory Spiking Neural Network Accelerator for Event-based Perception](https://arxiv.org/pdf/2411.02854)<br>arXiv; 正式venue未核实<br>2024-11-05|§II; Fig4; §III/TableII; PDF pp.2–4,7|未找到官方RTL/代码；不得把论文图当开源实现|65nm实测；DSEC/T10；AER门槛例子和位图/S2A成本|
|9|[hARMS: A Hardware Acceleration Architecture for Real-Time Event-Based Optical Flow](https://space.pitt.edu/sites/default/files/2024-10/hARMS_-A-Hardware-Acceleration-Architecture-for-Real-Time-Event-Based-Optical-Flow.pdf)<br>IEEE Access 10:58181–58198<br>2022-05-13|§II-B; §III–IV; Algorithm1; PDF pp.3–6 / printed pp.58183–58186|本轮未定位官方代码；原文硬件边界可读|孔径法向流；recent flow buffer；局部流计算在PS而非PL加速范围|
|10|[MotionDeltaCNN: Sparse CNN Inference of Frame Differences in Moving Camera Videos with Spherical Buffers and Padded Convolutions](https://openaccess.thecvf.com/content/ICCV2023/papers/Parger_MotionDeltaCNN_Sparse_CNN_Inference_of_Frame_Differences_in_Moving_Camera_ICCV_2023_paper.pdf)<br>ICCV 2023<br>2023-10; arXiv v1 2022-10-18|§3.1–3.7; PDF pp.3–5|[作者页已核；未定位本篇完整公开代码；不能以DeltaCNN仓库充当本篇代码](https://dabeschte.github.io/paper/2023/10/02/motiondeltacnn.html)|运动对齐与环形缓存；bias初始化；边界状态和刷新|
|11|[Flow-Based Visual Stream Compression for Event Cameras](https://arxiv.org/pdf/2403.08086)<br>精读arXiv v1；后续IEEE IoT Journal 11(24):40229–40243<br>arXiv 2024-03-12; issue 2024-12-15; DOI10.1109/JIOT.2024.3450428|§IV-A–D; PDF pp.4–6|主文/摘要未找到官方代码；最终出版主文未取得|过去flow预测事件；sending/predicting；周期刷新与匹配开销|
|12|[ASNA-Flow: An Efficient Asynchronous Neuromorphic Accelerator for Real-Time Event-Based Optical Flow](https://ieeexplore.ieee.org/document/11142472/)<br>IEEE TVLSI 33(12):3409–3422<br>2025-12|全文未取得；publisher正文不可读|未找到官方代码；不据摘要推测实现|最近邻待核；不计入全文精读|
|13|[Dynamic Frame Interpolation in Wavelet Domain](https://arxiv.org/pdf/2309.03508)<br>IEEE TIP 2023; DOI10.1109/TIP.2023.3315151<br>arXiv v1 2023-09-07; v2 2023-09-21|§III-C–D; Algorithm1; PDF p.6|[官方仓库已核；本分支未执行；父研究另核源码](https://github.com/ltkong218/WaveletVFI)|粗运动/动态阈值/高频mask/dilate3/通道并集皆为已有A|

## 本地证据

|文件|已读取内容与限制|
|---|---|
|[current_bottlenecks.md](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/current_bottlenecks.md)|当前层份额、r0源稀疏性；含转置卷积泛化零项文案，应以profile_current.py实际公式纠正|
|[profile.json](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/profile.json)|单帧实际调用/源density；非硬件周期|
|[profile_current.py](../../open_fusion_execution/major_operator_fusions_20260913/root_owned/profile_current.py)|普通conv按output算；transpose按input scatter×9算，未额外计stride插零|
|[parent_network.py](../../open_fusion_execution/breadth_20260912/algorithm/parent_network.py)|patch BN校准；PROJECT的动态onepass覆写；不能据此推定decoder2 BN模式|
|SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_modules.py|MS residual、MS transpose decoder原始语义；部署远端快照本地不完整|
|SDformer/third_party/SDformerFlow/models/STSwinNet_SNN/Spiking_submodules.py|完整T×T PSN；不能声称未来bin可省|
|SDformer/third_party/SDformerFlow/DSEC_dataloader/event_representations.py|双极性插值voxel；原始精细时间不可从voxel逆恢复|
|SDformer/third_party/SDformerFlow/DSEC_dataloader/DSEC_dataset_lite.py|预处理npy和原始事件list为不同接口|
|ideafromai/gptpro/ChatGPTpro-#第二轮跨领域调研：.md|选择精读事件光流、SpiDR、DeltaCNN、Phi相关条目；仅作线索|
|ideafromai/gptpro/ChatGPTpro-#硬件idea深挖0911.md|选择精读光流输入、差分和模式复用条目；不继承θ身份错误|
|ideafromai/research/grok46_20260905/04_paper_survey.md|事件光流硬件/稀疏前端相关条目；所有技术主张回原论文|

可机读日期和来源见 [sources.csv](sources.csv)、[sources.json](sources.json)。
