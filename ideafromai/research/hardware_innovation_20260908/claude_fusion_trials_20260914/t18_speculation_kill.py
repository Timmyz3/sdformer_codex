#!/usr/bin/env python3
"""T18：COMPASS 式推测+恢复（speculation + recovery）的定量杀。

对象：在 depth-d 平面处推测判决、乐观推进下游、误推测时冲刷恢复。
数据：results/t17_result.json 的 80 条 trace（40 序列×2 阶段）planes 直方图。

支配论证（三支）：
- 无验证：= T7 静态深度，错误判决率 ≤ misfire(d)=P(planes>d)，破坏精确合同；
- 有验证：验证"未锁定"就是证书的区间比较逻辑 → 推测方案内嵌证书；而证书是
  前缀供数的最优停时（T12 解析下界，1+max(msb−j*,1) 恰好达到）→
  推测供数期望 = 证书 + E[(d−planes)^+]（浪费），冲刷期望 = misfire(d)·d；
- 延迟轴：下游早 d 平面拿到判决不转换为吞吐——瓶颈是位串行接口供数
  （T13b 净服务/T15 位传输 6.3×），且 Y 唯一消费者无下游争用（T13a）。

本脚本算：misfire(d) 曲线、供数浪费 E[(d−planes)^+]、冲刷期望 misfire(d)·d、
"推测总开销/证书"比——全部配置 ≥ 证书。自有代码。
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
JIN = ROOT / 'results' / 't17_result.json'
OUT = ROOT / 'results' / 'T18_REPORT.md'


def main():
    res = json.loads(JIN.read_text())
    # planes 直方图池化（80 条 trace，G=20000/条）
    maxp = 23
    hist = np.zeros(maxp + 1, np.int64)
    for r in res:
        for k, n in r['planes_hist'].items():
            hist[int(k)] += n
    total = hist.sum()
    planes_pmf = hist / total
    surv = 1 - np.cumsum(planes_pmf)                    # misfire(d)=P(planes>d)
    mean_planes = float((np.arange(maxp + 1) * planes_pmf).sum())
    mean_cyc_cert = 1 + float(np.maximum(np.arange(maxp + 1), 1) @ planes_pmf)
    ratio_cert = mean_cyc_cert / 24

    cert_planes = float(np.maximum(np.arange(maxp + 1), 1) @ planes_pmf)
    rows = []
    for d in range(1, 9):
        mis = float(surv[d])                            # P(planes > d)
        # 接口供数 = max(planes, d)（未锁定组续供到 lock；锁定组已供 d）
        spec_planes = float(np.maximum(np.arange(maxp + 1), d) @ planes_pmf)
        waste = spec_planes - mean_planes               # E[(d−planes)^+]
        flush = mis * d                                 # 下游冲刷（消费者拍）
        rows.append({
            'd': d, 'misfire_pct': mis * 100,
            'supply_waste_planes': waste, 'flush_exp_cycles': flush,
            'spec_ratio': (1 + spec_planes) / 24,
            'over_cert_pct': ((spec_planes + flush) / cert_planes - 1) * 100,
        })

    bench = ('证书基准：E[planes]=%.2f，E[cyc]=%.2f 拍/组 = **%.1f%%（24 拍基线）**；\n'
             '推测（有验证支）接口供数 = E[max(planes,d)]，冲刷 = misfire(d)·d。' % (
                 mean_planes, mean_cyc_cert, ratio_cert * 100))
    lines = ['# T18：推测+恢复（COMPASS 式）定量杀（2026-09-15）', '',
             '> 脚本：`t18_speculation_kill.py`；数据：`results/t17_result.json`（80 trace，',
             '1.6M 组 planes 分布池化）。对象 = T12 指出的未覆盖同族变体 COMPASS',
             '（adaptive spike speculation，无全文——用其机制结构做支配论证）。', '',
             '## 1. 支配论证（结构，先于数字）', '',
             '设推测深度 d 平面（d=0 即不推测）。三条路：', '',
             '1. **无验证**：判决取部分和的当前侧 → 错误率 ≤ misfire(d)=P(planes>d)。',
             '   精确合同破坏（C1 的主张是逐判决精确终止），退化为 T7 静态深度',
             '   （T7 实测逐组表 pad0 误 fire 37–38%）。杀。',
             '2. **有验证**："检测未锁定" = Vmin/Vmax 区间比较 = 证书逻辑本身 →',
             '   推测方案内嵌证书硬件。而证书是前缀序精确终止的最优停时',
             '   （T12 解析下界：供数 ≥ 1+max(msb−j*,1)，证书恰好达到）→ 推测在此',
             '   之上只能加：接口供数 +E[(d−planes)⁺]（多余平面）+ 冲刷 misfire(d)·d',
             '   （下游乐观工作作废重算）。**严格 ≥ 证书，等号只在 d=0**。杀。',
             '3. **延迟轴**：推测的价值=下游早 d 平面拿到判决。但本设计瓶颈是位串行',
             '   接口供数（T15：位传输 6.3× 少；T13b 净服务口径），Y 唯一消费者无',
             '   下游争用（T13a）→ 接口拍数不变时延迟收益不转换为吞吐。不成立。', '',
             '## 2. 数字（80 trace / 1.6M 组池化）', '',
             bench, '',
             '| 推测深度 d | misfire(d) | 接口浪费(平面/组) | 冲刷期望(拍/组) | 有验证推测拍比 | 超出证书 |',
             '|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append('| %d | %.1f%% | %.2f | %.2f | %.1f%% | %+.1f%% |' % (
            r['d'], r['misfire_pct'], r['supply_waste_planes'],
            r['flush_exp_cycles'], r['spec_ratio'] * 100, r['over_cert_pct']))
    lines += [
        '', '证书行（d=0，即不推测）不在此表——它就是基准 17.1%。所有 d≥1 配置：'
        '接口供数严格增加、冲刷期望>0、错误风险>0（无验证支）——',
        '**没有任何配置优于证书**。', '',
        '## 3. 结论', '',
        '推测+恢复轴关闭。至此"终止信息"轴的三个变体全部定量杀死：',
        '静态预算（T7：最优 30% vs 证书 17.4%，锁深是样本性质）、',
        '值/指数预测（T12：解析下界，e′≥e 误差必须折进界）、',
        '推测恢复（T18：被证书双向支配，本表）。C1 的运行时精确区间证书',
        '在设计空间"终止信息"轴上无剩余竞争者。', '']
    OUT.write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
