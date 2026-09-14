# D3有限双context共享执行

[REPORT.md](REPORT.md)给最终三臂结论；[RESOURCE_CONTRACT.md](RESOURCE_CONTRACT.md)说明唯一共享producer算术/权重和双方双tile状态。最强普通RR对照胜过阶段错位，保留负差分。独立审阅见[root REVIEW_DATAFLOW](../../REVIEW_DATAFLOW.md)。

```bash
/opt/anaconda3/bin/python3.12 prepare.py
/opt/anaconda3/bin/python3.12 run.py
/opt/anaconda3/bin/python3.12 run_stream.py --count 3 --first 159
/opt/anaconda3/bin/python3.12 run_stream.py --count 64 --first 128
/opt/anaconda3/bin/python3.12 run_stream.py --full
```

Verilator4.028，严格-Wall、--cc --exe后make。run_stream默认三模式；`--modes 2 --append`可在已有两个完整收据后仅补mode2，不重复其它模式。SV与C++为唯一执行源，不依赖bootstrap生成脚本。父目录`audit.py`/`summarize.py`分别检查和汇总最终收据；需要旧冻结data中真实NPY/NPZ，按脚本路径读取。
