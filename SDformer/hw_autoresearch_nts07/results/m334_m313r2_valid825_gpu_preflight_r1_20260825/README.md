# M334：M313r2 GPU 启动前旧 launcher 独立评审

旧 launcher（SHA256 `9954c62f...`）存在一个 P1：同字节 contract 克隆可通过完整 SHA pin，和合同中的 `contract_clone_allowed=false` 不一致。因此旧 bytes 不准启动 GPU。

没有 P0。克隆必须保持完整 SHA，不能替换语义或字节；其风险是唯一命令和路径 provenance 不严，不是伪造结果。其余 argv、baseline 替换、输入 SHA 漂移和预建 result root 攻击均 fail-closed。

本评审没有导入 wrapper/evaluator，没有启动 GPU，没有修改 contract、baseline 或 docs/359。该发现已由后续 M334r2 对修复 launcher 独立复核。
