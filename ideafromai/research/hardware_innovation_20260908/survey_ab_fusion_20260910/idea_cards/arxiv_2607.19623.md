# arXiv:2607.19623 · UEP Bit-Position DNN Memory

- uid/来源: `ARX-026`｜arxiv_2607.19623+本地excerpt（`p0_excerpt_batches_gap/gap_07.json`）
- 题名: From Bit-Position Sensitivity to Unequal Error Protection for DNN Inference Memory
- 精读深度: 方法级（方法摘录窗口+补arxiv HTML §5 UEP）+依据：跨模态Xsafe地板；三档FP保护；UEP三层码(SECDED/SEC/bypass)；cacheline类型标签；双分区SRAM；原摘录偏灵敏度结果已补§5

## 可继承 A
推理存储不等错误保护：按位敏感定Xsafe地板/模态档 + 三层UEP码(符号指数SECDED/高阶尾数SEC/LSB旁路) + cacheline类型标签 + 双电压分区SRAM——相对均匀SECDED的保护/能耗对照（借入≠X）。

## 强对照 B
均匀SECDED护全比特；无位敏感标定；单电压全保护宏；忽略多单元翻转边界。

## 可差分 X线索
UEP/Xsafe≠lifting X；可靠性旁路，勿搬ECC面积%/读能当净服务%。

## 与 F1–F7 / Stage B 关系
F7弱相关（容错存储组织）。可靠性旁路。不抢 Stage B。f_candidates含F7。

## 不可搬用边界
FIT/ECC面积≠valid825；补HTML§5；位平面交错≠lifting供数合同。

## 可复用 idea 点
- Xsafe地板(FP16=6/BF16=4/FP32=15)作保护边界合同
- 三层UEP+奇偶分码作分级纠错模板
- cacheline数据类型标签作运行时档旁证
- 双分区低电压旁路LSB作读能边
- 负结果只停该UEP挂接

## 杀门建议
SER约束或与Stage B同分母冲突 → 保持旁路。
