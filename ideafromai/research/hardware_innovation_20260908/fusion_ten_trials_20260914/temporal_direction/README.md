# 精确T10共同方向RTL

[结果与判定](REPORT.md) · [资源](RESOURCE_CONTRACT.md) · [SUMMARY](SUMMARY.json) · [原始收据](results.json) · [独立审阅](INDEPENDENT_REVIEW.md)

最终mode0/1/2共228命令，各rawp/J20/I24检查875520值；新增缩放在真实八块未命中，比强控制多1058拍。范围固定收口。

```bash
/opt/anaconda3/bin/python3.12 prepare.py
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 verify.py
/opt/anaconda3/bin/python3.12 summarize.py
```

Verilator4.028，严格-Wall、--cc --exe后make。SV/C++是唯一实现源，脚本不生成或替换RTL；prepare只构造原生source/静态参数和独立gold。依赖旧冻结r8_consumer_fusion_20260914/data中的真实输入/消费者系数，路径由脚本解析。fixture binary/build可重建且ignore。first_control_receipt.json是被公平控制补强替换的旧收据，不计最终运行数；benefits.csv只含最终228条。
