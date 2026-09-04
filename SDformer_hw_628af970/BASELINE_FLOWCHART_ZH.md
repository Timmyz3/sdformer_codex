# Baseline 流程框图

这份文档只做一件事：

**把 baseline 从原始数据到 loss 的整条数据流，画成能反复对照的流程框图。**

---

## 1. 最简总图

```mermaid
flowchart TD
    A["原始 DSEC 数据<br/>events.h5 / rectify_map.h5 / flow png / timestamps"] --> B["预处理脚本<br/>prepare_dsec_single_sequence.py<br/>prepare_dsec_full.py"]
    B --> C["saved_flow_data<br/>gt_tensors / mask_tensors / event_tensors / sequence_lists"]
    C --> D["DSECDatasetLite<br/>按样本名读取 chunk / mask / label"]
    D --> E["DataLoader<br/>把多个样本打成 batch"]
    E --> F["train_flow_parallel_supervised_SNN.py<br/>训练循环"]
    F --> G["输入整理<br/>正负极性拆分 / 归一化 / 必要时阈值化"]
    G --> H["MS_SpikingformerFlowNet_en4<br/>模型总入口"]
    H --> I["Spikingformer_MultiResUNet<br/>主机身"]
    I --> J["spiking_former_encoder<br/>encoder"]
    J --> K["MS_Spiking_SwinTransformer3D_v2<br/>Swin 主干"]
    K --> L["residual blocks"]
    L --> M["decoder"]
    M --> N["prediction heads<br/>多尺度预测"]
    N --> O["pred_list['flow']"]
    O --> P["flow_loss_supervised<br/>结合 label 和 mask 计算 loss"]
    P --> Q["backward + optimizer.step()"]
```

---

## 2. 只看“数据”怎么流

```mermaid
flowchart TD
    A["原始 flow png"] --> B["decode_flow_png()"]
    B --> C["gt_tensors/样本名.npy<br/>label"]
    B --> D["mask_tensors/样本名.npy<br/>mask"]

    E["原始 events.h5"] --> F["按时间窗截取事件"]
    G["rectify_map.h5"] --> H["rectification"]
    F --> H
    H --> I["voxel 化"]
    I --> J["event_tensors/.../样本名.npy<br/>chunk"]

    K["train_split_seq.csv / valid_split_seq.csv"] --> L["样本名"]
    L --> C
    L --> D
    L --> J
```

---

## 3. 只看“训练时一个样本怎么走”

这里始终盯一个样本：

```text
zurich_city_09_a_0001.npy
```

```mermaid
flowchart TD
    A["train_split_seq.csv 中的一行<br/>zurich_city_09_a_0001.npy"] --> B["DSECDatasetLite.__getitem__(idx)"]
    B --> C["读取 mask_tensors/zurich_city_09_a_0001.npy"]
    B --> D["读取 gt_tensors/zurich_city_09_a_0001.npy"]
    B --> E["拆出 sequence 名<br/>zurich_city_09_a"]
    E --> F["读取 event_tensors/10bins/left/zurich_city_09_a/zurich_city_09_a_0001.npy"]
    C --> G["返回 mask"]
    D --> H["返回 label"]
    F --> I["返回 chunk"]
    G --> J["DataLoader 打 batch"]
    H --> J
    I --> J
```

---

## 4. 只看“chunk 进入模型后怎么走”

```mermaid
flowchart TD
    A["chunk"] --> B["训练器输入整理<br/>正负极性拆分 / 归一化 / 阈值化"]
    B --> C["MS_SpikingformerFlowNet_en4<br/>最外层入口"]
    C --> D["Spikingformer_MultiResUNet<br/>主机身"]
    D --> E["spiking_former_encoder"]
    E --> F["MS_Spiking_SwinTransformer3D_v2"]
    F --> G["encoder 输出 blocks"]
    G --> H["取 blocks[-1]"]
    H --> I["residual blocks"]
    I --> J["decoder"]
    J --> K["每层 decoder 后做 prediction"]
    K --> L["得到多尺度 predictions"]
    L --> M["最外层整理成 pred_list"]
    M --> N["pred_list['flow']"]
```

---

## 5. 只看“loss 怎么算”

```mermaid
flowchart TD
    A["pred_list['flow']"] --> B["flow_loss_supervised"]
    C["label"] --> B
    D["mask"] --> B
    B --> E["只在有效 mask 区域上计算误差"]
    E --> F["得到 curr_loss"]
    F --> G["backward"]
    G --> H["optimizer.step()"]
```

---

## 6. 一句话总结

```text
原始数据先被预处理成 saved_flow_data；
训练时 DSECDatasetLite 从 saved_flow_data 读出 chunk、mask、label；
训练器继续整理 chunk 后送入模型；
模型经过外层入口、主机身、encoder、Swin 主干、residual、decoder、prediction 得到 pred_list['flow']；
最后 loss 用 label 和 mask 对预测进行监督。
```
