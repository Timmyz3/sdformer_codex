from pathlib import Path
import json

HERE = Path(__file__).resolve().parent

def main():
    data = json.loads((HERE/'results.json').read_text())
    rows = []
    total_checks = dict(values=0, differences=0, directory_calls=0)
    for row in data['rows']:
        a = row['arms']
        if len(a) != 4:
            continue
        original, scan, pred, iterator = (a[k] for k in ['original_scan','coalesced_scan','static_predicate','address_iterator'])
        for report in a.values():
            for check in report['checks'].values():
                total_checks['values'] += check['values']
                total_checks['differences'] += check['differences']
            total_checks['directory_calls'] += report['NRV_directory_calls']
        r = {k:row[k] for k in ['axis','window','mask']}
        r.update(service={k:v['service_slots'] for k,v in a.items()},
            directory_service={k:v['directory_service'] for k,v in a.items()},
            directory_read_bytes={k:v['directory_read_bytes'] for k,v in a.items()},
            physical_port_bytes={k:v['physical_port_bytes'] for k,v in a.items()},
            common_coalescing_vs_original_fraction=1-scan['service_slots']/original['service_slots'],
            static_predicate_vs_coalesced_fraction=1-pred['service_slots']/scan['service_slots'],
            iterator_vs_static_predicate_fraction=1-iterator['service_slots']/pred['service_slots'],
            iterator_vs_coalesced_fraction=1-iterator['service_slots']/scan['service_slots'],
            directory_iterator_vs_predicate_fraction=1-iterator['directory_service']/pred['directory_service'],
            same_mask_all_NRV_and_integer_checks_exact=all(v['NRV_byte_exact_to_same_mask_original_scan'] and all(c['differences']==0 for c in v['checks'].values()) for v in a.values()),
            changed_network_delta_vs_original_capture=iterator['changed_network_delta_vs_original_capture'])
        # Split removed controller visits: static-deleted k versus common
        # image-boundary-empty k. Neither is a proprietary primitive.
        full_visits=scan['directory_counts']['K864_scan_select_and_predicate']
        valid_visits=scan['directory_counts']['INRV_CACHED_issues']
        actual_visits=iterator['directory_counts']['INRV_CACHED_issues']
        predicate_visits=pred['directory_counts']['K864_scan_select_and_predicate']
        bitmap_cost=sum(iterator['directory_counts'].get(k,0) for k in ['H8_retained_offset_bitmap_merge','uniform_boundary_offset_bitmap_merge','uniform_H8_bounds_bitmap_select'])
        r['controller_accounting']=dict(coalesced_scan_visits=full_visits, predicate_visits=predicate_visits, iterator_visits=actual_visits,
            common_boundary_empty=full_visits-valid_visits, statically_masked_valid=valid_visits-actual_visits,
            iterator_visits_removed_vs_predicate=predicate_visits-actual_visits, iterator_only_bitmap_build_slots=bitmap_cost,
            net_iterator_slots_removed_vs_predicate=pred['service_slots']-iterator['service_slots'])
        rows.append(r)
    comparisons=[]
    for axis in ['ordinary','lifting_raw']:
        for window in ['corner','interior']:
            group=[r for r in rows if r['axis']==axis and r['window']==window]
            if len(group)!=3:continue
            global_row=next(r for r in group if r['mask']=='global_group2')
            for r in group:
                comparisons.append(dict(axis=axis,window=window,mask=r['mask'],
                    best_static_service=min(r['service']['static_predicate'],r['service']['address_iterator']),
                    best_global_static_service=min(global_row['service']['static_predicate'],global_row['service']['address_iterator']),
                    best_static_delta_vs_best_global=min(r['service']['static_predicate'],r['service']['address_iterator'])-min(global_row['service']['static_predicate'],global_row['service']['address_iterator']),
                    iterator_delta_slots_vs_global=r['service']['address_iterator']-global_row['service']['address_iterator'],
                    iterator_reduction_fraction_vs_global=1-r['service']['address_iterator']/global_row['service']['address_iterator']))
    result=dict(evidence=data['evidence'],scope=data['scope'],complete_fixed_cases=len(rows),
        no_new_algorithm_no_training_no_AEE=True,no_native_projection_globalBN_or_fullframe=True,
        common_word_coalescing_static_skip_and_priority_encoder_are_not_X=True,
        resource_limits='96x8 RF with signed48/FP32 lanes, state128KiB+coef128KiB, 1R64+1W64 state and 1R256 coef; no added ports. 9 RF cache vectors occupy432physicalRFbytes for144logicalpayloadbytes; geometry9RF vectors, mask1RF vector. 64bit bounded scalar iterator-controller reserve is granted to all coalesced arms; original diagnostic uses old controller.',
        check_totals=total_checks,rows=rows,same_iterator_phase_vs_global=comparisons)
    stress_path=HERE/'results_ordinary_interior_phase_joint_stress.json'
    if stress_path.exists():
        stress=json.loads(stress_path.read_text())
        sr=stress['rows'][0]
        if len(sr['arms'])==4:
            ready=next(r for r in data['rows'] if (r['axis'],r['window'],r['mask'])==(sr['axis'],sr['window'],sr['mask']))
            for name,report in sr['arms'].items():
                assert report['checks']==ready['arms'][name]['checks']
                assert report['changed_network_delta_vs_original_capture']==ready['arms'][name]['changed_network_delta_vs_original_capture']
            result['fixed_stress']=dict(axis=sr['axis'],window=sr['window'],mask=sr['mask'],
                service={k:v['service_slots'] for k,v in sr['arms'].items()},
                all_ready_stress_payload_checks_exact=True,
                trace='Period32 SR blocked24..31, SW blocked28..31; original one RF WB arbitration.')
    (HERE/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 完整K864源地址接口：固定实测结果','',
        '这一轮闭合了已有掩码的物理地址跳过接口；普通读合并、静态掩码和地址迭代均归公共底座，没有建立独占创新X。采用原始固定参数和真实已发出的sn2门图，连续raw I24与PED义务全部保留。','',
        '范围是四个固定窗口（两学生×corner/interior），每窗16个锚点，完整K864→U16→F/BN2+raw I24→projection gate及PED U32/V96；另保留窗口内非锚点raw I24门消费者。不是全帧、RTL、PPA或新AEE。','',
        '|学生/窗|掩码|原扫描|共同读合并|普通静态predicate|地址iterator|iterator比predicate少|','|---|---|---:|---:|---:|---:|---:|']
    for r in rows:
        s=r['service'];lines.append(f"|{r['axis']}/{r['window']}|{r['mask']}|{s['original_scan']:,}|{s['coalesced_scan']:,}|{s['static_predicate']:,}|{s['address_iterator']:,}|{100*r['iterator_vs_static_predicate_fraction']:.3f}%|")
    lines += ['',
        '地址/掩码费用：四phase相同mask自动得到uniform常量编译：每H8仅选一次实际metadata bit，零组直接跳过；这条相同规则适用于任意候选。普通predicate不支付无用位图构建。静态metadata为48bit，冷填32B并逐目录经过SR64读取、拆为四个12bit相位行；源phase取全局sy/sx。每H4以9个现有RF向量收集P2×9的18个SR64字，144B逻辑门载荷占432B现有RF物理位。几何、边界、掩码查表、H8 offset位图构建、每次选择/地址控制、解包写回与NRV写出均收费。H8跨两个H4；不新增读端口。门图外部DMA没有减少。',
        '',
        '强对照权限：三种mask共同得到相同coalescer、metadata格式及resident MAC；同mask比较的NRV顺序/字节/live完全相同，后继Acc48/RNE、更新I24、门和真实PED逐值符合独立整数gold。iterator比predicate少的访问须再扣它独有的位图构建；corner还含普通图像边界空项，summary已分开计数。global/interior中iterator反而多168槽。这不是增加一个新稀疏数学原语。',
        '',
        f"验证总数：{len(rows)}个固定负载×4臂，{total_checks['directory_calls']}次完整K目录；检查{total_checks['values']:,}个后继输出值，差异{total_checks['differences']}。不同掩码改变网络输出的误差仍单独保留，不能将同mask实现的0差读成剪枝无损。",
        '',
        '当前结论：给uniform-global完整静态权限后，phase/水平P2在本次完整局部消费者上均慢于最强global约0.3–0.7%。iterator相对同mask predicate只再减少约0.04–0.12%（global interior反而略慢）。本控制布局保留为可用公共编译/供数底座，不以它恢复phase-H8标题。相对同iterator的phase与global差异见summary，不以原始重复读取版夸大新意。当前mask只是先前10帧评价的同一参数；本轮没有新增训练、量化或valid825。没有改生产或旧执行程序。',
        '',
        '文件：execute.py为隔离wrapper，PLAN.md为执行前B/A/X与强对照，results.json含全部收费分项与目录轨迹，summary.json含相同负载与不同mask两个分母。']
    if 'fixed_stress' in result:
        lines += ['', '固定压力复核：ordinary/interior/phase_joint四臂在同一period32读写背压下完成，所有NRV与后继输出检查同ready一致；服务值见summary.fixed_stress。']
    (HERE/'README.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
