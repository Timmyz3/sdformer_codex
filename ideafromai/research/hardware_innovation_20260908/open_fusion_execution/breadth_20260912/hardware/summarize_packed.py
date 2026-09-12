from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
OLDSTAGE=HERE.parents[1]/'stage_20260912'


def main():
    rows=[];phase=[];quality={};values=0
    for axis in ('ordinary','lifting_raw'):
        quality[axis]={}
        for mode in ('W8','W4'):
            p=OLDSTAGE/'algorithm/weight_controls/aee'/axis/mode/(mode+'_summary.json')
            quality[axis][mode]=json.loads(p.read_text())
        for label in ('corner','interior'):
            for stress in ([False,True] if (axis,label)==('ordinary','interior') else [False]):
                path=HERE/(axis+'_'+label+('_stress' if stress else '')+'.json')
                d=json.loads(path.read_text());assert d['complete']
                ref=d['rows'][0]
                for r in d['rows']:
                    values+=sum(c['values'] for c in r['checks'].values())
                    assert all(c['differences']==0 for c in r['checks'].values())
                    if not r['mode'].endswith('_packed'):continue
                    mode=r['mode'].split('_')[0]
                    expanded=next(x for x in d['rows'] if x['mode']==mode+'_expanded16')
                    rows.append(dict(axis=axis,window=label,stress=stress,mode=mode,
                        original32_service=ref['service_slots'],expanded_service=expanded['service_slots'],packed_service=r['service_slots'],
                        packed_change_percent=r['service_change_vs_same_function_expanded_percent'],
                        consumer_change_percent=r['consumer_change_vs_same_function_expanded_percent'],
                        coefficient_cold_fill_bytes_saved=expanded['coefficient_pool_blob_bytes']-r['coefficient_pool_blob_bytes'],
                        CR256_bytes_saved=expanded['physical_port_bytes']['CR256']-r['physical_port_bytes']['CR256'],
                        SW64_bytes_change=r['physical_port_bytes']['SW64']-expanded['physical_port_bytes']['SW64'],
                        SR64_bytes_change=r['physical_port_bytes']['SR64']-expanded['physical_port_bytes']['SR64']))
                    if axis=='ordinary' and label=='corner' and not stress:
                        phase.append(dict(mode=mode,expanded_U=expanded['stages']['resident_U_ped'],packed_U=r['stages']['packed_U_ped'],
                            coefficient_fill_change=r['stages']['integer_coefficient_replace']-expanded['stages']['integer_coefficient_replace'],
                            decode_groups=r['counts']['packed_code_sign_extend_RF'],weight_RF_to_staging=r['counts']['packed_weight_RF_to_common_staging'],
                            row_scale_16x24_products=r['counts']['row_scale_existing_16x24_product'],row_scale_shift_adds=r['counts']['row_scale_shifted48_merge'],
                            extra_operand_wait=r['counts']['operand_wait']-expanded['counts']['operand_wait']))
    report=dict(complete=True,rows=rows,ordinary_corner_phase=phase,checked_values=values,value_differences=0,
        AEE_links='stage_20260912/algorithm/weight_controls/aee: actual original R32/CUDA parents; neither R24+onepass825 nor new AEE inherited.',
        verdict='Actual packed code/scale interface works and saves coefficient traffic, but this decode-and-scale layout increases full local service. Keep low-bit quality/parameters and stop claiming this layout accelerates.',
        evidence='CPU payload slot prototype; no production or EDA.')
    (HERE/'packed_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    table='\n'.join(f"| {r['axis']} {r['window']} {'stress' if r['stress'] else 'ready'} {r['mode']} | {r['original32_service']:,} | {r['packed_service']:,} | +{r['packed_change_percent']:.3f}% |" for r in rows)
    (HERE/'PACKED_RESULTS.md').write_text(f'''# W4/W8真实压缩权重执行

**完成固定四窗与一组压力，总25条完整局部链；压缩接口功能正确，当前布局没有服务净收益。** 同函数的expanded16与packed都对实际GPU部署字段逐项一致；不是拿原权重精度配新压缩拍数。全部updated I24、projection gate与PED检查共{values:,}值0差。

| 同负载 | 原R32／同函数expanded16 | 实际packed | 同函数整局部服务变化 |
|---|---:|---:|---:|
{table}

每窗W8实际少填2,976B、少读23,808B系数；W4少填4,512B、少读36,096B系数。SR64与SW64字节不变。代码/行尺度/32B头确实写入模拟coefficient memory，原expanded U16没有出现在packed存储体。U分别为3072B或1536B code＋64B尺度＋32B头，V还是6144B原q16，原bias保留。

代价出在哪里：ordinary/corner普通U为113,936槽，W8 packed131,200、W4 packed130,432。每窗3,072个H8解包，3,072次RF→既有staging搬移，1,280次16×24尺度乘积与640次移位合并，较普通多9,360个operand等待槽。减少的冷填W8为558槽、W4为846槽，抵不过这些真实步骤。完整局部服务分别增加16,706/15,650槽；固定压力例增加17,888/16,160槽。不同量化后的V输出同样逐值核对，未减少有义务的PED写出。

双方同96×8×48 RF、128KiB state/coef、SR64/SW64/CR256、8192B源ROM、32B/5slot DMA。代码解包到RF84，再付费读到共同64B staging的16B区域；source gather最多24B，预算不增加。MAC读取acc＋source两RF口，权重来自暂存，未开第三口。行尺度作用于宽点积，因此用signed24 hi/lo分解、两次共同16×24乘法和shift/add，之后才做原U RNE/sat；没有把宽乘法当一拍免费操作。V RNE及bias原位置不变。

每窗前级真实执行一次并复制其完整Machine状态/时间/待写回/仲裁，分别跑五个后级；原R32逐count复现上一阶段。没有旧表相加、没有新门替换旧SRAM输入。边界仍从真实I24到完整K864、BN2/rawI24、gate＋PED U32/V96；**native投影、全域BN与最终join不在此表**。

本轮的A是普通低位权重底座，不是X。只停止当前RF解包＋尺度回写布局的加速主张，保留已过十帧质量的低位参数。尚未试的接口是共用系数响应路径内解码／更低位乘法 datapath，但不能在本表中假设其无成本。这里无法证明ReverB/MiLo完整系统失败，也不能仅凭压缩比立贡献。

实现：[packed_weights.py](packed_weights.py)、[run_packed.py](run_packed.py)；同负载机器表：[packed_summary.json](packed_summary.json)。旧stage和生产目录保持只读，没有训练/EDA或新AEE。
''')
    print('PACKED_SUMMARY',len(rows),values)


if __name__=='__main__':main()
