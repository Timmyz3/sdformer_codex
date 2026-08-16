# H67 四 Stage Projection 累加器位宽与 Overflow 合同

## 1. 目的

DCTF 完整 projection 会让六路 final 独立流出。如果正常工作负载可能在最后几个 token 才发生 Acc overflow，那么早期已经被下游接受的 final 无法撤回，需要整 tile quarantine 或 tentative commit协议。

本轮先回答更基础的问题：在当前 H67 dyadic INT8 projection合同下，32-bit Acc是否从数学上可能溢出。

## 2. 数据与重算

输入来自现有真实RTL向量：

- 四stage共45个head、7290条真实gate/K raw record；
- S0-S3的INT8 projection weight；
- 32-bit bias accumulator；
- token-major `expected_output_acc32`。

脚本按RTL等价公式重算：

~~~text
activation[token, head*32+lane] =
    K[token, head, lane] ? gate[token, head] : 0

output[token, out] =
    sum(activation * signed_int8_weight) + signed_bias
~~~

四个stage与金参考全部0 mismatch。

## 3. 结果

| Stage | DIM | 实际gate最大 | 激活密度 | 实际最终最大绝对值 | 实际中间部分和最大绝对值 | 真实激活绝对和界 | gate511+当前权重界 | 全INT8配置界 | 配置界所需有符号位 | int32最小裕量 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S0 | 96 | 81 | 1.151% | 22865 | 22854 | 29174 | 1644410 | 6279351 | 24 | 341.99x |
| S1 | 192 | 0 | 0.000% | 1375 | 0 | 1375 | 4285793 | 12559711 | 25 | 170.98x |
| S2 | 384 | 81 | 0.555% | 32892 | 32877 | 44096 | 6282262 | 25116781 | 26 | 85.50x |
| S3 | 768 | 81 | 2.172% | 55035 | 55015 | 85572 | 8895497 | 50233425 | 27 | 42.75x |

S1在当前sample/window上没有激活事件，输出仅为bias；这是真实向量特征，不应外推为S1恒空。

## 4. 四种边界

1. **实际最终值**：当前真实trace按RTL公式重算并加bias；
2. **实际中间部分和**：按input-channel顺序逐项累加，记录任意中间拍最大绝对值；
3. **真实激活绝对和界**：对当前每个激活乘积取绝对值后求和再加bias，不依赖累加顺序；
4. **gate511+当前权重界**：所有K均激活、gate取9-bit最大511，使用当前weight/bias；
5. **全INT8配置界**：进一步把所有weight幅值放大到128。

最后一项是当前DIM、gate位宽和weight位宽下的配置级保守界，不依赖sample0的稀疏率。最坏S3为50,233,425，小于int32正上限2,147,483,647约42.75倍。

## 5. 硬件决策

当前主配置冻结：

~~~text
DIM <= 768
gate unsigned <= 511
weight signed INT8
ACC_W = 32
~~~

在该合同下，正常数值路径不需要整 tile final quarantine。DCTF可以继续使用六路流式 final，避免为极晚错误缓存 `162*96*32 bit` 输出。

overflow仍保留为：

- 非法DIM/位宽配置；
- weight/bias存储损坏；
- 协议或软件部署合同破坏；
- 未来网络扩大后未重新签核。

完整顶层在检测到overflow时必须报告错误且不得宣告成功tile completion，但论文不能把这一点写成“已撤回所有早期final”。真正的安全依据是配置级静态上界，而不是运行时回滚。

## 6. 复现

~~~bash
python3 scripts/test_analyze_projection_accumulator_range.py
python3 scripts/analyze_projection_accumulator_range.py \
  --vector-root tb_hitflow/vectors \
  --raw-records tb_hitflow/vectors/gatestack_all45_builder_20260720/raw_records.memh \
  --output-dir results/projection_accumulator_range_20260720
~~~

结构化结果与中文自动报告位于：

~~~text
results/projection_accumulator_range_20260720/
~~~

限制：真实分布只覆盖sample0/window0；但全INT8配置界不依赖该样本。若DIM、gate、weight或bias合同变化，必须重新运行并重新签核。
