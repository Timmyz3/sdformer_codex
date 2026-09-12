"""Explicit interface mapping of every catalog view; overlapping views != ideas."""
from pathlib import Path
import csv
import json
from collections import Counter

HERE=Path(__file__).resolve().parent
OPEN=HERE.parents[1]
ROOT=OPEN.parent

# Explicit assignments after reading the view list and its mechanism fields.
# These are navigation families, not a claimed count of independent mechanisms.
GROUPS={
'F01':[0,6,10,12,16,25,83,85,90,102,103,128,173,174,183,190,210,213,231,233,238,241,251],
'F02':[1,5,49,73,88,89,92,127,131,136,202,257,275,276,287],
'F03':[2,3,71,212,252,253,289,304],
'F04':[4,23,24,26,35,37,38,39,48,52,56,74,75,95,133,208,223,224,234,235,254,256,278,298,301],
'F05':[7,50,54,58,59,69,70,137,141,199,219,272,300],
'F06':[8,45,60,66,80,93,97,98,132,140,147,189,192,237,297,299,302,303],
'F07':[],
'F08':[36,47,53,87,118,229,244],
'F09':[64,65,77,96,126],
'F10':[258,259,271,290,294],
'F11':[13,14,46,86,119,165,170,178,194,225],
'F12':[166,260,296],
'F13':[19,57,68,101,104,107,111,112,120,121,124,149,153,154,155,158,160,169,177,182,186,206,218],
'F14':[67,76,91,109,130,261,262,263,269,270,284,285,295],
'F15':[9,61,72,78,81,94,134,139,148,152,159,164,176,180,185,191,221,222,239,242,264,265,266,267,268,280,281,282,286,293],
'F16':[79,195,197,273,277,283,291],
'F17':[11,15,27,28,30,31,62,82,100,105,106,113,114,116,117,129,135,138,142,143,144,145,146,151,156,175,216,220,228,232,240,243,249,279],
'F18':[29,32,33,34,40,41,42,43,198,226,250],
'F19':[17,18,20,21,22,51,55,63,84,99,108,110,115,122,123,125,150,157,161,162,163,167,168,171,172,181,187,200,203],
'F20':[184,214],
'F21':[179,188,193,196,201,204,205,207,209,211,215,217,227,230,236,245,246,247,248],
'F22':[255,288,292],
'F23':[],
'F24':[274],
'F25':[44],
}

# status, concrete tested interface, real output evidence, remaining distinct interface, batch
PROFILES={
'F01':('稀疏NRV×W供数/共享索引','已实做局部；完整原作未迁',
 '2tile×4ID完整C384→T10的原双指针RTL，以及本轮同址合并/冷cache CPU前端',
 'psn/rtl/gp_slice/intersection_result.json;open_fusion_execution/breadth_20260912/coverage_owned/shared_index_results.json',
 '8×8完整层、索引与源/NR4并行生成、真实冷DMA、后继FC2/BN2；本文CPU不覆盖这些', 'B1'),
'F02':('时间类别/FTP/在途上下文','已实做固定F_live2；其余布局开放',
 'packed-class/time与有限NR4；本轮同P4×R7×F2=56字S/总双NR4的352次CPU数值排程，F1同预取/合并/编译权限，当前F2布局未胜最强F1',
 'psn/rtl/gp_slice/intersection_result.json;open_fusion_execution/breadth_20260912/coverage_owned/flive2/results.json;open_fusion_execution/breadth_20260912/coverage_owned/flive2/README.md',
 '完整阵列、不同NRV推进/包所有权的多上下文布局及source16+consumer80交织；F4未试且本轮不扫，不扩大候选容量','B2'),
'F03':('重复/包含关系/有限公共部分和','只停旧布局',
 'M20/K16/H8完整K864的TCAM/列流与当前融合已测，元数据税抵消算术收益',
 'open_fusion_execution/EXECUTION_QUEUE.md',
 '固定lifting子图的受限匹配、跨K有限父值生命周期；完整Prosperity/Phi/ExSpike不是旧融合结果','B6'),
'F04':('物理源字删读/结构化剪枝','已实做局部；新训练布局未试',
 'global/phase/row-phase源门剪枝、请求前地址生成、同mask/all_keep控制',
 'open_fusion_execution/stage_20260912/hardware/mask_fusion_global_group2.json;open_fusion_execution/stage_20260912/hardware/mask_fusion_all_keep.json',
 '同预算窄稠密、2:4、VENOM/HiNM完整训练；真实物理字并集目标而非分支稀疏率','B3'),
'F05':('时间因子结构/常量编译/配对训练','已实做匹配训练、新825、局部执行及GPU halo',
 '保留旧RTL/35RNE/DFS；新三结构同父320步恢复/十帧/新常量公共RTL及matched局部六例CPU已执行；另同规则两项signed-PoT固定投影dense/lifting已做CPU门差和公共RTL，指数/cutoff/下游不变，量化新函数两臂十帧过NB0、四局部链及实际GPU整数端点匹配',
 'open_fusion_execution/stage_20260912/hardware/rtl_source/schedule_controls.json;open_fusion_execution/breadth_20260912/algorithm/matched_training/run.json;open_fusion_execution/breadth_20260912/source_execution/;open_fusion_execution/breadth_20260912/hardware/matched_local_chain/;open_fusion_execution/breadth_20260912/algorithm/valid825/run.json;open_fusion_execution/breadth_20260912/source_constant_probe/summary.json',
 '主三臂825/GPU halo、两项新函数diverse10及四halo已完成；真实I24全层/新参数完整下游、普通小RF同权交织、不同配对/PIT完整恢复仍开放','B3'),
'F06':('精确证书/整组接受与回退','只停旧证书/旧早停布局',
 '原9/11已做严格equal-a/水平P2分级余量；本轮独立复放同305760lane及38220SIMD8证书命中0，不计新接口，未停不同参考轴',
 'PRO_GROK_REVIEW_20260911.md;open_fusion_execution/EXECUTION_QUEUE.md;open_fusion_execution/breadth_20260912/rounding_margin/results.json',
 '新参考轴/可训裕度/连续消费者联合接受，比较应不先算原乘法；量化稳定不等于门正确','B5'),
'F07':('低秩/PED门条件连续表示','已实做普通量化强对照及两种有费解码；整链开放',
 '两旧R24+onepass各825；普通P1留Z公共布局已测；本轮两学生×fixed/center/affine/diagonal/full五表示diverse10，train-only参数；fullD缓存条件加、CR5+5及紧凑RF5+5实际执行，表放置未胜条件加',
 'open_fusion_execution/new_interface_selection/resident_latent/results.json;open_fusion_execution/breadth_20260912/representation/aee/run.json;open_fusion_execution/breadth_20260912/hardware/prediction_decoder/ready.json;open_fusion_execution/breadth_20260912/hardware/prediction_decoder/compact_ready.json',
 '在完整消费者中消化残差码/step/非零默认值，支付当前门生成与共享端口费用；普通affine/diagonal保留，表示十帧不继承父825','B4'),
'F08':('低位权重/符号主体补偿','已实做压缩code/尺度局部执行；只停当前放置',
 'sign+rank旧CPU与W4/W8小集；本轮四窗加压力25条同Machine局部链，真实packed code+row-scale、宽尺度分解和原RNE收费，功能正确但当前布局服务增加',
 'open_fusion_execution/stage_20260912/weight_compensation/results.json;open_fusion_execution/breadth_20260912/hardware/packed_summary.json;open_fusion_execution/breadth_20260912/hardware/PACKED_RESULTS.md',
 '系数响应内解码/低位乘法datapath另计成本，MiLo/ReverB或joint sign+rank同预算恢复；旧R32父执行不能混成R24+onepass或新matched学生整链','B4'),
'F09':('比特面/有符号位稀疏','只停旧串行布局',
 '真实I24位字默认/校正在公共强MAC下慢4–5倍',
 'open_fusion_execution/EXECUTION_QUEUE.md',
 '相同位宽/面积的bit-PE、低校正密度训练；原BitVert/LoAS完整内核未迁','B7'),
'F10':('动态BN/默认值/晚到参数','已实做完整外部gate/PED起点后段；I24前段整层未闭',
 '公共onepass/两旧R24组合825；本轮完整192000×96的native→实算全域BN→PED join同Engine，外部raw/PED读写收费；固定两块目录跨H32复用和新W8全权驻留已测，后者付解码后再少3.45%服务且同父十帧过NB0，是更强普通分母',
 'open_fusion_execution/stage_20260912/algorithm/stage_summary.json;open_fusion_execution/breadth_20260912/hardware/native_bn_join/results_ready.json;open_fusion_execution/breadth_20260912/hardware/native_bn_join/directory_reuse/results_ready.json;open_fusion_execution/breadth_20260912/coverage_owned/native_chain_review.md',
 '真实I24源/preview/双消费者整层生成gate/PED、code0格式及新训练参数完整链；CPU native/GPU有差异不能继承825；running-BN/无norm替代须重训','B1'),
'F11':('有限生命周期/保留重算/上下文交织','已实做局部；交织未试',
 '8088归因/驻RF供数/P1留Z已测；原dense两链CSD524字未准入；相同策略在两项量化dense上229字/13工作RF+门RF已过RTL但慢于完整CSE；16+80仅设计，尚未执行',
 'PRO_GROK_REVIEW_20260911.md;open_fusion_execution/stage_20260912/hardware/summary.json;open_fusion_execution/new_interface_selection/resident_latent/results.json;open_fusion_execution/breadth_20260912/coverage_owned/INTERLEAVE_NEXT_INTERFACE.md;open_fusion_execution/breadth_20260912/source_constant_probe/dense_low_state/results.json',
 '共同RF/issue/SR-SW下真正交织：原80acc+4供数+13源临时+1门需98RF，固定2供数方案待测；已准入低RF普通控制须同权且量化质量另评，40/61RF不是dense下界','B2'),
'F12':('真正帧间差分/状态尾部','已实做机会探针；完整新接口未试',
 'motion/result.json已实测带连续时间戳4帧/3pair及上一帧flow；另有T10块内不同探针',
 'motion/result.json;motion/capture/frames.json;open_fusion_execution/motion_residual/results.json',
 '当前学生连续帧完整数值/动态BN尾部/历史读写端口；ST-BIF平衡与固定预算对照','B5'),
'F13':('因果运动/事件唤醒与细节教师','未试目标唤醒；已有相邻帧机会',
 '只已有上一帧完成flow引导的支持统计；不是本视图的事件/TDE wake实现',
 'motion/result.json;PRO_GROK_REVIEW_20260911.md',
 '原事件或过去完成预测构建wake，计预测/错过恢复；禁止当前/未来flow oracle','B5'),
'F14':('Motion-XOR/二值注意力精确跳过','已实做部分；身份/完整范围待补',
 '旧ep35 QK census、MX3P叶/脏行/K零机会，不能迁为当前R24网络周期',
 'open_fusion_execution/catalog/idea_views.csv;PRO_GROK_REVIEW_20260911.md',
 '匹配检查点的行内memo、三pop独立门控、K零只停输出乘积且保留分母','B7'),
'F15':('注意力/残差算子替换或训练','未试此网络替换',
 '原打分叶分析不等于SDSA/QKFormer/STSA/SLI等新网络恢复',
 'open_fusion_execution/catalog/idea_views.csv',
 '只替换一个固定block、同训练预算与原head比较；IAND不能无损删PED','B8'),
'F16':('tokenizer/稀疏stem/decoder岛','未完整试；旧切片仅部分',
 'decoder仅旧ep34 shards，未闭完整Table-A；Spiking Patches未恢复',
 'open_fusion_execution/catalog/idea_views.csv',
 '原GT任务下stem/decoder完整层支持与费用，然后一个固定替换恢复','B8'),
'F17':('平台/混合引擎/CIM启发','未迁原平台；非直接独立idea',
 '个别数字叶可借；不存在本项目原宏/原工艺完整实现',
 'exploration_tcasii_20260911/round4_wide/REPORT.md',
 '仅迁与器件无关的地址/调度/复用并另计数字成本；宏/PPA不能直接搬','B9'),
'F18':('编译/仿真/库/综述入口','工具或地图；非独立机制',
 'Verilator已运行；其余库的链接不是已安装或全量复现证明',
 'open_fusion_execution/stage_20260912/hardware/rtl_source/results.json;open_fusion_execution/catalog/repositories.csv',
 '只在具体算子需要时核API/格式并迁最小kernel；无需为数目逐库建空实验','B9'),
'F19':('任务backbone/数据/参考模型','任务对照；多数模型替换未试',
 'SDformerFlow本地NB0已实际配对；其他模型名不代表当前学生已有对应实现',
 'open_fusion_execution/accuracy_baseline/valid825_summary.json',
 '明确新学生/数据需求后一次等预算对比，不能用不同任务FPS/AEE作硬件胜出','B8'),
'F20':('时间/尺度/梯度训练先验','已实做固定常量投影；原完整训练配方未试',
 '当前三结构已有同预算QAT；另固定两项signed-PoT对新dense/lifting相同整数最近投影已CPU/公共RTL实做，无训练/指数扫描，不能称DeepShift/PIT/S-TLLR完整复现',
 'open_fusion_execution/stage_20260912/literature/FUSION_CANDIDATES.md;open_fusion_execution/breadth_20260912/algorithm/matched_training/run.json;open_fusion_execution/breadth_20260912/source_constant_probe/README.md',
 '两项新函数AEE单独评估；PIT time-channel scale+surrogate、普通per-channel QAT同预算与连续分支同步，原训练配方仍开放','B3'),
'F21':('器件/传感应用/无直接任务接口','当前对象不适配；不等于缺资料',
 '应用/器件指标不提供当前T10双消费者可执行义务',
 'open_fusion_execution/catalog/idea_views.csv',
 '先有明确可迁数字算子才重开；不为凑试验数替换传感器/任务','B9'),
'F22':('字节布局/广播/搬运','旧广播已试；新transpose未试',
 'TSBG旧ep34 B8广播已实测；新注意力字节permute仅提案',
 'open_fusion_execution/catalog/idea_views.csv',
 '相同内容/总容量/物理字宽下打包与转置读写，布局转换不能免费','B7'),
'F23':('精确codec/非零默认值','只停已测布局作标题',
 'FP32/I24共63488 mode-block往返；I24全模式只比signed-width好约0.1%',
 'open_fusion_execution/stage_20260912/literature/exact_codec_results.json;open_fusion_execution/stage_20260912/literature/i24_codec_results.json',
 '只在融合后仍必需spill的位置重开，计真实编码/索引/随机访问和端口','B7'),
'F24':('AT-LIF身份/幅度折权','已澄清身份；非新机制',
 '推理输出{0,theta}、固定theta可折入W；连续PSN/PED另算',
 'ATLIF_INFERENCE_CONTRACT.md',
 '若改神经元或检查点需显式新函数；撤销不可吸收逐事件int8旧前提','B0'),
'F25':('错绑/身份冲突视图','身份冲突',
 'MAIN-R057标FireFly而正文SCNN，此视图不能算FireFly全文证据',
 'open_fusion_execution/catalog/idea_views.csv',
 'SCNN正文可回F01；FireFly独立核题名/原文后再迁，不继承错绑PPA','B0'),
}


def main():
    with (OPEN/'catalog/idea_views.csv').open(encoding='utf-8-sig') as f: views=list(csv.DictReader(f))
    with (OPEN/'catalog/works.csv').open(encoding='utf-8-sig') as f: works={r['catalog_id']:r for r in csv.DictReader(f)}
    assigned={}
    for family,indices in GROUPS.items():
        for i in indices:
            assert i not in assigned,(i,assigned.get(i),family)
            assigned[i]=family
    assert set(assigned)==set(range(len(views))),('unmapped',set(range(len(views)))-set(assigned))
    # Only concrete unresolved primary materials reported in the read originals.
    external={91:'Configurable CSA正文位宽/两模式时序未取得；可先做通用CSA控制',
              93:'COMPASS完整推测/恢复原文未取得；不阻塞普通数字早判探针',
              107:'ASNA-Flow仅摘要，无原PE/状态细节；不能冒称完整迁移',
              108:'ERAFT FPGA原PDF/工件未取得；其帧RAFT身份不能当事件SNN',
              124:'同ASNA-Flow摘要缺正文，别名视图不新增idea',
              132:'同COMPASS来源缺正文，别名视图不新增idea',
              135:'SPARTA全文/工件未取得，token skip思想可先用普通数字控制'}
    rows=[]
    for i,v in enumerate(views):
        f=assigned[i];name,state,done,evidence,remaining,batch=PROFILES[f]
        w=works.get(v['catalog_id'],{})
        if i in external: state='缺指定原作外部正文；通用接口仍可试'
        if i in (63,122): state='已实做本地NB0任务对照'
        if i in (254,261,262,263,269,270): state='已有旧身份实测；当前学生未迁'
        if i == 302: state='原9/11已实试；本轮独立复放非新接口'
        row=dict(view_index=i,idea_ref=v['idea_ref'],name=v['name'],view_type=v['view_type'],catalog_id=v['catalog_id'],
            interface_family=f,family_name=name,coverage_state=state,
            original_A=v['A'],original_B=v['B'],original_X=v['X'],
            concrete_family_execution=done,family_evidence=evidence,
            view_specific_remaining=v['next_or_stop']+' | '+w.get('untried_interfaces',''),
            remaining_executable_interface=remaining,next_batch=batch,
            external_gap=external.get(i,''),
        execution_scope='Current family status/evidence describes bounded representative execution, NOT full implementation of this paper/variant. view_specific_remaining and stale_source_status preserve historical source wording; use remaining_executable_interface for the current family boundary.',
            stale_source_status=v['status'],source=v['source'],original_boundary=v['boundary'])
        rows.append(row)
    target=HERE/'coverage_all_views.csv'
    with target.open('w',encoding='utf-8-sig',newline='') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]));out.writeheader();out.writerows(rows)
    profiles=[]
    for family,(name,status,done,evidence,remaining,batch) in PROFILES.items():
        profiles.append(dict(family=family,name=name,status=status,view_count=len(GROUPS[family]),
            executed=done,evidence=evidence,remaining=remaining,batch=batch))
    summary=dict(catalog_view_rows=len(rows),view_types=dict(Counter(v['view_type'] for v in views)),
        distinct_nonempty_catalog_ids=len({v['catalog_id'] for v in views if v['catalog_id']}),
        coverage_states=dict(Counter(r['coverage_state'] for r in rows)),families=profiles,
        independent_idea_count='NOT_DEFINED: views overlap and families are navigation groupings, not independent novelty counts',
        scope='All305 current idea views mapped once. Supplemental Grok/Pro/new-literature interfaces are explicit in COVERAGE.md; this is not all775 works experimentally reproduced.',
        external_gap_note='These7 views name specific locally unavailable primary materials, not proof they are globally unobtainable. No external gap blocks the generic first prototype.')
    (HERE/'coverage_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='families'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
