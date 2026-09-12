"""Generate bounded-source supplement and one-page execution record."""
import csv
import json
from pathlib import Path

H=Path(__file__).resolve().parent
rows=[]
def add(id,name,venue,year,url,code,local,old,read,missing,decision):
    rows.append(dict(id=id,name=name,venue=venue,year=year,primary_url=url,author_code=code,catalogue_status=local,original_survey_status=old,reading_depth=read,complete_original_implementation_missing=missing,decision=decision))

add('L01','ZipServ: Fast and Memory-Efficient LLM Inference with Hardware-Aware Lossless Compression','ASPLOS',2026,'https://www.cse.ust.hk/~weiwa/papers/zipserv-asplos26.pdf','https://github.com/HPMLL/ZipServ_ASPLOS26 ; Apache-2.0 LICENSE read','missing','exact-name search: no matches','Primary PDF §3–4, Algorithm2, §5 boundary; README/LICENSE. Not entire experimental reproduction.','Original BF16 TCA-TBE tile hierarchy, 128-bit padding, offsets, fused CUDA ZipGEMM and serving not ported. Local FP32 mode retains 24 sign/mantissa bits and uses simpler sequential packing.','Add strong exact-compression prior; no transferred GPU speedups.')
add('L02','Atalanta: A Bit is Worth a “Thousand” Tensor Values','ASPLOS',2024,'https://www.ece.mcmaster.ca/~ameer/hadilab/publications/ASPLOS24-Atlanta.pdf','No original-author code URL verified in this bounded search','missing','exact-name search: no matches','Author PDF §3–4.2, compression boundaries; local full PDF available, partial method reading only.','Arithmetic-coded range symbol plus raw offset, probability tables, parallel codecs and original RTL not implemented locally.','Add direct prior against treating range-plus-residual coding as new.')
add('L03','Shannonic: Efficient Entropy-Optimal Compression for ML Workloads','MLSys',2026,'https://proceedings.mlsys.org/paper_files/paper/2026/file/96f39c8de84678cb2a908cd52bfd7819-Paper-Conference.pdf','Official supplement exists; no original-author code repository verified','missing','exact-name search: no matches','Official PDF §4–5.1 and reported ASIC implementation boundary; partial methods, not full-paper replication.','Optimal range partition, tANS tables/state transitions, full software/RTL not ported; evaluated primary inputs mostly INT8, not arbitrary I24/FP32.','Add stronger entropy-code control, especially after explicitly trained quantization.')
add('L04','MiLo: Efficient Quantized MoE Inference with Mixture of Low-Rank Compensators','MLSys',2025,'https://proceedings.mlsys.org/paper_files/paper/2025/file/9032e5c9ec394ce768a2fa9bdc56af6c-Paper-Conference.pdf','https://github.com/Supercomputing-System-AI-Lab/MiLo ; MIT LICENSE read','missing','title/acronym+colon search: no matches','Official PDF §3.2 optimization, §3.3 kernel, Appendix F artifact; README/LICENSE.','Full HQQ alternating quantizer/compensator optimization, adaptive rank, INT3 compensator/kernel and MoE experiments not ported.','Use weight-quantization plus low-rank compensation as strong A, not a new contribution by itself.')
add('L05','HG-PIPE: Vision Transformer Acceleration with Hybrid-Grained Pipeline','ICCAD',2024,'https://arxiv.org/pdf/2407.17879','https://github.com/PKU-SEC-Lab/ICCAD24-HG-PIPE ; MIT LICENSE read','missing','exact-name search: no matches','Author PDF §4.1–4.3 pipeline and precision boundary; README generation/test route.','HLS kernels, auto-generation and full FPGA board flow not run; ViT attention/quantization is not Motion-XOR/strict dynamic BN.','Add common pipeline/precision control; no new C1 or source-address subproject.')
add('L06','SpikeStream: Accelerating Spiking Neural Network Inference on RISC-V Clusters with Sparse Computation Extensions','DATE',2025,'https://arxiv.org/pdf/2504.06134','Snitch public ecosystem discussed; exact paper-complete author artifact not verified','missing','exact-name search: no matches; distinct from old SpikeStream simulator','Author PDF §III/Listing1, §IV numerical/RTL evaluation boundary.','Sparse-position extraction, SpVA kernels, FP16/FP8 toolchain, Snitch RTL/cycle simulation not ported.','Add software/hardware sparse baseline; not applicable to deleting nonzero continuous obligations.')
add('L07','A 24.46TOPS/W and 3.04TOPS/mm2 BF16×1-bit CIM-based BERT Accelerator in 28nm CMOS','CICC',2026,'https://ieeexplore.ieee.org/document/11509512/','No original implementation verified','missing','exact performance-string search: no matches','Publisher-indexed title/abstract metadata; original document open attempted, browser challenge. No full paper read.','Digest/circuit figures and code missing; BF16/binary silicon numbers cannot price signed24/f14 local digital implementation.','Real lead for continuous×binary direction; withhold detailed circuit claims pending primary digest.')
add('L08','OFQ-LLM: Outlier-Flexing Quantization for Efficient Low-Bit Large Language Model Acceleration','TCAS-I',2025,'https://ieeexplore.ieee.org/document/10924797','No original implementation verified','missing','not established beyond catalogue search','Author publication list confirms identity; publisher open challenged. No full paper read.','Original method and code not accessed; third-party abstracts not used for mechanism claims.','Supplement identity only; do not select as fused mechanism until original text is read.')
add('L09','A Broad-Spectrum and High-Throughput Compression Engine for Neural Network Processors','TCAS-II',2024,'https://ieeexplore.ieee.org/document/10433078','No original implementation verified','existing W0749','already in previous venue supplement','Author list and publisher entry reopened; challenged original. Prior reading level not upgraded.','Complete method/codec RTL absent in this round.','Existing strong-neighbor identity retained; mixed codec selection is not automatically novel.')
add('L10','Advancing Spatiotemporal Representations in Spiking Neural Networks via Parametric Invertible Transformation','ICLR',2026,'https://proceedings.iclr.cc/paper_files/paper/2026/file/0ac46bb0a72a7afe311d9b48b5088df8-Paper-Conference.pdf','https://github.com/YinsongYan/ICLR26 ; MIT LICENSE.txt read','missing','exact-title phrase search: no matches','Official PDF §3.2–3.4 and Appendix A reparameterization; README/LICENSE.','Full training, corrected surrogate, 3-sigma initialization and folded inference not ported. Actual A_t is diagonal time/channel scaling, not a free dense temporal matrix.','Add strongest learnable-scaling/gradient control; folds must be re-derived for dynamic BN and shared consumers.')
add('L11','VISTREAM: Improving Computation Efficiency of Visual Streaming Perception via Law-of-Charge-Conservation Inspired Spiking Neural Network','CVPR',2025,'https://openaccess.thecvf.com/content/CVPR2025/papers/You_VISTREAM_Improving_Computation_Efficiency_of_Visual_Streaming_Perception_via_Law-of-Charge-Conservation_CVPR_2025_paper.pdf','https://github.com/Intelligent-Computing-Research-Group/ViStream ; LICENSE says MIT; README badge says MuLan PSL2.0, discrepancy retained','missing','exact-name search: no matches','Primary §3–4.4 and equilibrium/early-exit limitations; README/LICENSE.','ST-BIF stateful implementation, checkpoint and full streaming tasks not run; energy code is estimation, not local silicon.','Reopen only cross-inference/stateful new-function experiment, not lossless substitution into noncausal PSN.')
add('L12','ReverB-SNN: Reversing Bit of the Weight and Activation for Spiking Neural Networks','ICML',2025,'https://arxiv.org/pdf/2506.07720','No author code URL located in PDF or bounded search','missing','mentioned twice only in 2609.04949 extracted reference/excerpt, no dedicated record found','Original PDF §3 and Algorithm1 including learnable binary amplitude/reparameterization.','Full real-valued-neuron training and all network results not replicated.','Reopen continuous-activation plus sign-weight hardware hypothesis; distinguishes weight-only change from activation Dg+residual trial.')
add('L13','EDCFlow: Exploring Temporally Dense Difference Maps for Event-based Optical Flow Estimation','CVPR',2025,'https://openaccess.thecvf.com/content/CVPR2025/papers/Liu_EDCFlow_Exploring_Temporally_Dense_Difference_Maps_for_Event-based_Optical_Flow_CVPR_2025_paper.pdf','Author link is in official CVF page; code not fetched/run this round','existing W0195; local proposals W0280','39 text matches; includes earlier EDCΔFuse/IF-TID proposals','Official PDF architecture/method and DSEC protocol passages; supplement assumption checked.','Full pretrained refinement, cost-volume branch and same-DSEC split comparison not run.','Existing task prior; only re-open as training teacher/control, never call EDC fusion a newly discovered idea.')
add('L14','BSViT: A Bit-Serial Vision Transformer Accelerator Exploiting Dynamic Patch and Weight Bit-Group Quantization','TCAS-I',2024,'https://ieeexplore.ieee.org/abstract/document/10601322','No implementation verified','missing','not established beyond catalogue search','Author publication list confirms title; publisher blocked. No full paper read.','Original paper/bit-group quantization hardware not read or implemented. Not 2026 Burst Spiking ViT with same acronym.','Keep disambiguated metadata only; no mechanism ranking yet.')

with (H/'literature_supplement.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(H/'literature_supplement.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2)+'\n')

head='''# 本轮文献补漏与阅读边界（2026-09-12）

独立轮先于目录/其他ideas阅读：[independent_round.md](independent_round.md)。这是定向补查，不是逐篇proceedings审计，也不据“未搜到”声称首次。14个条目中12个未入当时works表，2个已有；12个漏项内含3个仅身份/摘要级线索（CICC/TCAS），不计入原文方法阅读。10份主论文PDF已本地保存并按相关方法部分阅读，没有把“下载了PDF”写成全文精读。未执行任何原作者完整系统。

已比对catalog/works.csv、VENUES.md、venue_coverage.csv、local_execution_priors.json、gptpro两文及research/audit_comparison_20260909/grok_comparison.md。指定research/Grok目录实际不存在，改用真实Grok对照文件；两份原表未修改。除本轮stage与生成catalog外，对ideafromai下md/json/csv/txt做了有界题名文字检索，结果见 [original_survey_text_crosscheck.json](original_survey_text_crosscheck.json)。它不能排除中文别名或未提取PDF中的引用。ReverB只在旧论文参考文献提过，EDCFlow确有大量已有提案。

| ID/工作 | venue/年 | catalog与旧survey | 本轮原文/代码层级 | 取舍 |
|---|---|---|---|---|
'''
lines=[]
for r in rows:
    lines.append('| '+r['id']+' ['+r['name']+']('+r['primary_url']+') | '+r['venue']+' '+str(r['year'])+' | '+r['catalogue_status']+'；'+r['original_survey_status']+' | '+r['reading_depth']+' '+r['author_code']+' | '+r['decision']+' |')
tail='''

逐项原作缺项见CSV/JSON的 `complete_original_implementation_missing`，不能把局部算子/README迁移记成完整A。GitHub API本轮遇rate limit，后用作者raw README/LICENSE成功读取；未冒称API树或commit已核。VISTREAM许可证文件与README徽标不一致，未集成其代码，保留差异。

本轮搜索覆盖ASPLOS/MLSys/DATE/ICCAD/CICC/TCAS与ICLR/CVPR，偏2024–2026，并补读ICML2025。TCAS两条新题名只核作者原始目录，IEEE原文被challenge/robots阻断；因此不是“TCAS已查完”。CICC不是新找到一篇就代表补齐所有节目。查到的GQA-LUT实际是DAC2024（作者CV/会议节目一致），不误填成TCAS-II；同名BSViT已消歧。EBPC/BDI/Finch/BNFF/LoopTree等旧强先验继续保留，本轮不重标全文阅读。

明确旧前提变化：用户现允许重训/新表示，原+0.005相对ordinary限制取消；因此原“无法无损折叠”只否定原函数无损实现，不能否定训练后新函数。反之，允许重训也不允许继承原模型精度。每个候选最终仍须同DSEC口径超过本地NB0 valid825 AEE=1.445353；diverse10=1.454603仅作小样本口径参照。
'''
(H/'LITERATURE_SUPPLEMENT.md').write_text(head+'\n'.join(lines)+tail)

fp=json.loads((H/'exact_codec_results.json').read_text());iq=json.loads((H/'i24_codec_results.json').read_text())
ss=[]
for r in fp['rows']:
    if 'full_output' in r['label']:
        c=r['codec_bytes'];ss.append('| '+r['label'].split('/')[0]+' FP32 BN输出 | '+format(r['raw_bytes'],',')+' | fill '+format(c['default_fill_or_raw'],',')+' | fill+exp '+format(c['default_plus_exponent_or_raw'],',')+' | 普通全模式 '+format(c['ordinary_all_modes'],',')+' |')
for r in iq['rows']:
    if r['layout']=='spatial_T_C':
        c=r['codec_bytes'];ss.append('| '+r['label'].split('/')[0]+' I24 PED | '+format(r['raw_packed24_bytes'],',')+' | signed-width '+format(c['block_signed_width'],',')+' | signed-delta '+format(c['base_signed_delta'],',')+' | 普通全模式 '+format(c['ordinary_all_modes'],',')+' |')
one='''# 真实一页结果：精确连续数据codec机会（2026-09-12）

**结论：I24普通块位宽已吃掉几乎全部本轮codec增量；FP32默认值＋指数编码可省字节，但只是更强普通底座，未证明标题级X。** 未用GPU/EDA、未改生产树或模型、未改BN算术。

数据：真实 `000_zurich_city_09_a_0001` 一帧，ordinary/lifting_raw两臂。FP32是capture的全域projection BN输入/输出 `10×96×120×160`，以及corner/interior preview切片；I24是权威capture_full_producers的 `full_continuous_q24`，signed24/f14，真实生产者布局spatial,T,C，并另列native布局及小source窗口。I24未重新量化：按24位打包作为原分母，不能拿NumPy int32文件当32位流量。

| 张量 | 原始B | 强简单对照B | 另一mode/组合B | 全普通mode选择B |
|---|---:|---:|---:|---:|
'''
one+='\n'.join(ss)
one+='''

FP32 fill+exp相对fill再省12.50%/12.56%，但全普通mode选择还省约0.89%/0.90%；这不是一个超越普通组合的X。I24全模式相对块signed-width仅省0.1045%/0.1080%。I24在native布局另有数值，但转换并非免费，不能跨布局挑最好数当已部署增量。全FP32源输入两臂也已测，见JSON。

实现：64值一块；模式/位宽字节、基值、默认值、指数基、掩码、escape原字及byte padding计入。FP32保留1+23符号/尾数位和完整指数，NaN payload、±0也可还原。FP32共43,008个mode-block、I24共20,480个mode-block实际序列化/解码为0 bit mismatch；另23个边界组合通过。全域费用是从每个真实块的整数长度计算，不把抽样压缩率外推。编码只支持顺序流，没有免费随机访问索引；如每块加32位索引，两个全域张量每臂各另加1,152,000B。FP32当前帧通道直方图/default选择是乐观编码端扫描，未给其免费实时预测身份。

边界：这是CPU功能/字节机会测试，非端口周期/面积/功耗，也未重跑AEE。只要原字按位恢复并保持消费算术，表示本身不改函数；尚未接入真实消费者。未完整移植ZipServ/Atalanta/Shannonic/EBPC，故也不能声称打赢这些原作。未知编码/解码/双消费者缓冲费用可能吃掉字节收益。更直接的本地反证是：已实现ordinary BN→PED融合不物化BN输出，针对该输出的当前新增可避免流量为0；离线capture大小不是融合后仍在搬运的字节，不能为codec恢复已删物化。

复现：`/opt/anaconda3/bin/python exact_codec_probe.py` 与 `... i24_codec_probe.py`。数据源路径、所有12个FP32和8个I24结果在 [exact_codec_results.json](exact_codec_results.json) / [i24_codec_results.json](i24_codec_results.json)。脚本与两份run.log、[edge_roundtrip.json](edge_roundtrip.json)同目录。主试验实际wall约10.72s与5.01s，仅用于复现，不当硬件性能。首跑系统Python3.6/缺numpy的系统3.12后已改用现有Anaconda3.12，不做环境安装。

下一决定：停止把I24自适应codec当主创新；把FP32普通fill+exp保留为所有候选共有的强对照。只在普通融合后仍必须spill的其他源上证实事务和decoder预算，才考虑孤立codec RTL。已有算法精度优于NB0不能替代这些新表示的端到端验证。
'''
(H/'ONE_PAGE_RESULT.md').write_text(one)

print('wrote',len(rows),'supplement rows and one-page result')
