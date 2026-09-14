"""Same-population quality comparison; network elapsed time is not hardware speed."""
from pathlib import Path
import csv
import json
import statistics

H = Path(__file__).resolve().parent
B = H.parents[1]
R = B / 'r8_consumer_fusion_20260914/data'


def main():
    nb0 = json.loads((R / 'nb0_valid825.json').read_text())
    axes = {'NB0': nb0['rows'],
            'flat_R8': json.loads((R / 'deployed_valid/integer_consumer_frames.json').read_text())}
    runtime = {}
    variants = [('spatial_q13', H), ('spatial_q11', H / 'q11'),
                ('moment', H / 'moment'), ('native_tap', H / 'native_tap'),
                ('unconstrained', H / 'unconstrained')]
    live_checks = {}
    for name, directory in variants:
        report = json.loads((directory / 'deployed_valid.json').read_text())
        assert report['complete'] and report['result']['complete']
        assert report['python'] == nb0['python'] and report['torch'] == nb0['torch'] and report['gpu'] == nb0['gpu']
        axes[name] = json.loads((directory / 'deployed_valid/spatial_integer_frames.json').read_text())
        diverse = json.loads((directory / 'deployed_diverse/spatial_integer_frames.json').read_text())
        valid_by_name = {row['file']: row for row in axes[name]}
        assert all(row == valid_by_name[row['file']] for row in diverse)
        runtime[name] = {k: report[k] for k in ('gpu', 'torch', 'python', 'TF32_matmul', 'TF32_cudnn')}
        if name in ('moment', 'native_tap', 'unconstrained'):
            import numpy as np
            with np.load(directory / 'sequence_tiles.npz') as actual, np.load(
                    H.parent / 'spatial_winograd_pruning' / name / 'gold_sequences.npz') as gold, np.load(
                    H / 'q11/sequence_tiles.npz') as mother:
                for field in actual.files:
                    assert np.array_equal(actual[field], gold[field]), (name, field)
                for field in ('source_words', 'output_origin_yx', 'identity_fp32_bits'):
                    assert np.array_equal(actual[field], mother[field]), (name, 'prefix', field)
                live_checks[name] = dict(tiles=len(actual['source_words']), raw_values=int(actual['p_int'].size),
                                        differences=0, independent_gold=True, matched_Q11_prefix=True)
    names = [row['file'] for row in axes['NB0']]
    assert len(names) == 825 and len(set(names)) == 825
    populations = [(row['file'], row['valid_pixels']) for row in axes['NB0']]
    for rows in axes.values():
        assert [(row['file'], row['valid_pixels']) for row in rows] == populations
    means = {name: dict(frame_mean=statistics.mean(row['AEE'] for row in rows),
                        pixel_mean=sum(row['aee_sum'] for row in rows)/sum(row['valid_pixels'] for row in rows))
             for name, rows in axes.items()}
    paired = {}
    for name, _ in variants:
        paired[name] = {}
        for ref in ('NB0', 'flat_R8', 'spatial_q11'):
            delta = [r['AEE']-b['AEE'] for r, b in zip(axes[name], axes[ref])]
            paired[name][ref] = dict(mean_difference=statistics.mean(delta),
                                     better_frames=sum(v < 0 for v in delta), worse_frames=sum(v > 0 for v in delta),
                                     largest_increase=max(delta), largest_decrease=min(delta))
    sequences = list(dict.fromkeys(name.rsplit('_', 1)[0] for name in names))
    by_sequence = []
    for sequence in sequences:
        indices = [i for i, name in enumerate(names) if name.rsplit('_', 1)[0] == sequence]
        by_sequence.append(dict(sequence=sequence, frames=len(indices),
                                **{name: statistics.mean(rows[i]['AEE'] for i in indices) for name, rows in axes.items()}))
    with (H / 'by_sequence.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(by_sequence[0]))
        writer.writeheader(); writer.writerows(by_sequence)
    result = dict(complete=True, frames=825, valid_pixels=sum(v for _, v in populations),
                  sequences=len(sequences), means=means, paired=paired, environments=runtime,
                  live_sequence_checks=live_checks,
                  interpretation='Same-task NB0 gate; student has existing coarse head, NB0 full flow[-1]. Not single-factor causal attribution.',
                  training=False, hardware_speedup='not inferred from AEE or GPU wall time')
    (H / 'comparison.json').write_text(json.dumps(result, indent=2)+'\n')
    text = ['# 空间R16及三种剪枝接口的完整825质量', '',
            '| 模型/接口 | 逐帧平均AEE | 像素加权AEE |', '|---|---:|---:|']
    for name, values in means.items():
        text.append(f"| {name} | {values['frame_mean']:.12f} | {values['pixel_mean']:.12f} |")
    text += ['', f'人口完全相同：825帧、{result["valid_pixels"]:,}有效像素、{len(sequences)}序列。独立diverse10与完整825子集逐帧一致。', '',
             '同一A80080GB/env312（Torch2.2.2+cu121）评估，无训练。NB0是本地原SDformerFlow PSN/SDSA ep29完整flow[-1]；候选保留已部署matched-dense学生与粗头。此比较满足用户的同任务baseline质量门，不能把整个差距归于R16或Winograd。', '',
             'Q13与Q11分别独立评估，没有将旧浮点R16或Q13精度借给Q11。Q11仅一次由Winograd系数和的13bit硬件上限推导的选择；量化有损，固定Q11后的普通卷积与Winograd重排则整数精确相等。', '',
             '三种剪枝均不训练、不使用验证集活动来选择系数：moment在原Q11整数域投影到一条变换项为零的约束；native_tap按相同BN/输出尺度权重删除一个物理抽头组；unconstrained直接删同一变换项而不投影原卷积。最后一种是独立两相位算子，保存p2并将输出尺度减半，重新RNE构造a_q40，不能称为普通共享1×3卷积。', '',
             'moment和unconstrained已经使用同一个三项RTL，所有记录周期完全一致；其差异必须由完整网络质量判断。native_tap用其更快的普通执行臂作性能对照，不能强迫它走已测更慢的通用Winograd。十帧为完整825的子集，只用于排错，最终取舍以本表为准。', '',
             '三种剪枝均在实际网络重捕18序列首帧的固定边缘/内部36tile：source和原identity与Q11母体相同，Z/raw/J/wide/I24与各自独立gold逐值一致。它连接网络函数与组件RTL输入，不表示36tile代表完整网络耗时。', '',
             f'平均改善不代表每帧改善：Q11相对NB0有{paired["spatial_q11"]["NB0"]["better_frames"]}帧更好、{paired["spatial_q11"]["NB0"]["worse_frames"]}帧变差，单帧最大AEE增加{paired["spatial_q11"]["NB0"]["largest_increase"]:.6f}。Q11逐帧平均略好于Q13，但像素加权平均略差，两项都保留。', '',
             '源为AT-LIF {0,θ}，θ已吸入系数。网络实际注入新I24/16384，下一位级读取者每帧检查完全相等；首diverse帧135tile的原source/identity/Z/p/J/wide/I24与本地独立gold相等。整网仍有浮点算子，不是整网RTL/bittrue证明。', '',
             '首次cuDNN FP64空间卷积出现约2.73e-12非整数微扰并被停止。当前只在整数oracle内部禁用cuDNN算法变换，直接FP64整数点积；未对错误中间值就地round，未改其它网络的TF32设置。', '',
             '[逐序列误差](by_sequence.csv)、[逐帧配对汇总](comparison.json)、[Q13原始结果](deployed_valid.json)、[Q11原始结果](q11/deployed_valid.json)、[moment](moment/deployed_valid.json)、[native_tap](native_tap/deployed_valid.json)、[unconstrained](unconstrained/deployed_valid.json)。', '',
             'AEE与网络运行时间不产生任何硬件倍率。真实周期、端口与消费者以相应RTL目录为准；当前无物理PPA。']
    (H / 'QUALITY_REPORT.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
