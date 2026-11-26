# step271482 真实文件与CPU状态核验

2026-10-16 已从专用A100服务器只读核实 `native_crop9_fresh_20261012/checkpoint_step_000271482.pt`，完整660,107,890 bytes，SHA `1e72cca3946f40e24a1575f3348f9a596d744460808ad910a616ba713b5e4977`。其latest、sidecar和segment记录一致，CPU可反序列化；711个模型张量、55,001,588个元素全部finite、全部在CPU。没有启动此权重的GPU推理或恢复原训练。【[远端原文及完整文件哈希](remote_receipt.json)、[CPU检查](cpu_checkpoint_audit.json)】

该状态来自canonical6000训练，完成271482个optimizer更新，即45个完整epoch加1482个样本，因wall_budget停止。配置目标仍为80epoch，不能称完成原训练配方；也不意味着满足正式DSEC复现门。【[segment](segment_result.json)、[run contract](run_contract.json)】

嵌入的contract与run_contract.json里的contract对象完全一致，按生产者的canonical JSON算法复算SHA为`961f1700e76f3427c74d4e9af9799a684433ff00449b8386be6c1f9cdd75e934`，与PT/sidecar/metadata相符。整个JSON文件的SHA和对象规范化SHA是不同口径，不混用。【[CPU合同检查](cpu_contract_audit.json)】

另一个代理独立检查了返回原文、游标算术与SHA绑定，结论是记录内部一致；它没有第二次读取PT，也不冒称完整optimizer/scheduler/RNG恢复等价或质量验收。【[独立记录核验](independent_metadata_review.json)】

R11和R12有界普通恢复仍使用已冻结的135930+CMA176，避免把更换权重计为新评分器或恢复收益。若后续采用271482，必须重新核native/FF/CC/固定保护质量、对应BN统计和数值图；新的前缀特征与损伤标签须重新采集。此目录不证明新状态AEE、正式测试口径、评分优越性或硬件收益。
