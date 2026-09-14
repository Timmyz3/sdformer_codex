# d1_forward

结论见[REPORT.md](REPORT.md)，共同资源见[resource_contract.json](resource_contract.json)，每条实际收据见results*.json和benefits.csv。固定输入函数及执行范围以报告为准。

在本目录依次运行：

```bash
/opt/anaconda3/bin/python3.12 prepare.py
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 run_stream.py --count 64 --first 128
```

在父目录运行 `python3 audit.py` 重做收据/地址一致性检查。依赖旧冻结 data 的真实NPY/NPZ，路径由脚本解析；Verilator4.028，严格-Wall、--cc --exe、make。SV/C++是完整可编译源，不依赖一次性编辑脚本。旧树只读。

独立实现审核：[root REVIEW_DATAFLOW](../../REVIEW_DATAFLOW.md)。
