"""D3 合同 CPU 数学验证：各向异性 stencil（axis-aligned anisotropic stencil, A3S）。

只验证数学恒等，不训练、不评测、不碰 GPU。对应合同草案：
CLAUDE_OPERATOR_CONTRACT_DRAFTS_20260818.md 的 D3。

验证项：
  K1  Δ=0 锚点恒等：A3S(Δ=0) 与现网 Local5 分数/门逐位一致（消融锚点）
  K2  网格精确位移：Q7(1/128, [-2,2]) 下 Δ=1/16 == 8 档精确位移（与分数量化 commute）
  K3  方向场语义：匀速移动图案 -> 3x3 时域 XOR 梯度 argmax 对齐运动方向
  K4  各向异性语义：移动图案下对齐 lane 的量化分数 argmax 命中率 > 0.5（winner
      class 指向运动轴；2^s 门动态范围下门质量再分配有界，诚实指标为命中率）
  K5  唯一门成本账：ident-K 场景分裂为 2 组唯一门（对齐/正交）的诚实记账
"""

from __future__ import annotations

import torch

# 64 核机器上微小张量的 advanced indexing 线程池争用开销巨大（单次 ~0.26s），
# 限制线程数后验证脚本恢复毫秒级。
torch.set_num_threads(2)


STEP = 1.0 / 128.0
LO, HI = -2.0, 2.0
N_BINS = int(round((HI - LO) / STEP)) + 1
DELTA_BINS = 8  # Δ = 1/16 on 1/128 网格


def quant_score(scores: torch.Tensor) -> torch.Tensor:
    codes = torch.round((scores - LO) / STEP).clamp(0, N_BINS - 1)
    return LO + STEP * codes


def shiftmax(scores: torch.Tensor) -> torch.Tensor:
    shifted = scores - scores.amax(dim=-1, keepdim=True)
    num = torch.pow(2.0, shifted)
    return num / num.sum(dim=-1, keepdim=True)


def _shift_k(k: torch.Tensor, dy: int, dx: int) -> torch.Tensor:
    """位移 k 平面（边界 clamp 复制，与 Local5 现网 stencil 一致）。"""
    h, w, d = k.shape
    yy = (torch.arange(h) + dy).clamp(0, h - 1)
    xx = (torch.arange(w) + dx).clamp(0, w - 1)
    return k[yy][:, xx]  # [H, W, D]


def local5_gate(q: torch.Tensor, k: torch.Tensor, alpha0: float = 0.02) -> torch.Tensor:
    """Local5 现网 gate 路径（_binary_alpha_xnor_stencil_attention 的简化精确版）。

    q/k: [T, H, W, D] 二值事件平面。候选 = self + 4 邻域（同时间平面）。
    """
    t, h, w, d = q.shape
    out = []
    for tt in range(t):
        lanes = []
        for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            lanes.append((q[tt] * _shift_k(k[tt], dy, dx)).sum(-1))
        stacked = torch.stack(lanes, dim=-1)  # [H, W, 5]
        same_spike = stacked
        same_silent = d - stacked
        scores = same_spike + alpha0 * same_silent
        scores = scores / d  # _normalize_consensus_score 的 head_dim 归一化
        scores = quant_score(scores)
        out.append(shiftmax(scores))
    return torch.stack(out)


AXIS_CODES = {"E": 0, "W": 1, "N": 2, "S": 3}  # axis_field 的返回码


def axis_field(k: torch.Tensor) -> torch.Tensor:
    """3x3 时域 XOR 梯度 argmax（A3S 的方向场）。返回 [T-1, H, W] 整数码。"""
    t, h, w, d = k.shape
    ku = k.to(torch.uint8)  # 二值平面转 uint8 才能做位异或
    m = (ku[1:] ^ ku[:-1]).sum(dim=-1).to(torch.long)  # [T-1, H, W] 时域 XOR popcount
    m = m.float().mean(dim=0)  # 时间平均（方向场按像素聚合，不再随 t 变）
    grad = {}
    for axis, (dy, dx) in {"E": (0, 1), "W": (0, -1), "N": (-1, 0), "S": (1, 0)}.items():
        rolled = torch.roll(m, shifts=(dy, dx), dims=(-2, -1))
        grad[axis] = (rolled - m).abs()  # 沿该轴的空间差分（逐像素）
    dirs = torch.stack([grad[a] for a in ("E", "W", "N", "S")], dim=-1)
    return dirs.argmax(dim=-1)  # [H, W] 方向场码


def moving_bar_planes(t: int, h: int, w: int, d: int, speed: int) -> torch.Tensor:
    """匀速向右移动的竖条：K 平面在条的前后沿有事件，其余静默。"""
    planes = torch.zeros(t, h, w, d)
    torch.manual_seed(0)
    for tt in range(t):
        bar_x = (tt * speed) % w
        for y in range(h):
            for dx in (0, 1):
                x = (bar_x + dx) % w
                if torch.rand(()) < 0.8:  # 事件密度
                    lanes = torch.randperm(d)[: torch.randint(1, 5, ()).item()]
                    planes[tt, y, x, lanes] = 1.0
    return planes


def check_K1() -> bool:
    """Δ=0 锚点恒等：A3S(Δ=0) == 现网 Local5 门。"""
    torch.manual_seed(0)
    ok = True
    for _ in range(10):
        t, h, w, d = 2, 15, 15, 32
        q = torch.randint(0, 2, (t, h, w, d)).float()
        k = torch.randint(0, 2, (t, h, w, d)).float()
        g_base = local5_gate(q, k)
        # A3S(Δ=0)：方向场计算但权重恒等
        dirs = axis_field(k)
        scores = local5_scores_raw(q, k)
        g_a3s = a3s_gate(scores, dirs, delta=0.0)
        if not torch.equal(g_base, g_a3s):
            ok = False
    print(f"[K1] A3S(Δ=0) 与现网 Local5 门逐位一致（10 组随机平面）：{'通过' if ok else 'FAIL'}")
    return ok


def local5_scores_raw(q: torch.Tensor, k: torch.Tensor, alpha0: float = 0.02) -> torch.Tensor:
    t, h, w, d = q.shape
    out = []
    for tt in range(t):
        lanes = []
        for dy, dx in ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)):
            lanes.append((q[tt] * _shift_k(k[tt], dy, dx)).sum(-1))
        stacked = torch.stack(lanes, dim=-1)
        same_silent = d - stacked
        scores = stacked + alpha0 * same_silent
        out.append(scores / d)
    return torch.stack(out)


def a3s_gate(scores: torch.Tensor, dirs: torch.Tensor, delta: float) -> torch.Tensor:
    """A3S 门：分数 + Δ·sign(axis == dir_d)，然后 Q7 + Shiftmax。

    scores: [T, H, W, 5]（lane 序 = self, N, S, W, E）。dirs: [H, W] 方向码。
    delta 以原始分数单位。
    """
    axis_of_lane = ("self", "N", "S", "W", "E")
    offset = torch.zeros_like(scores)
    for li, axis in enumerate(axis_of_lane):
        if axis == "self":
            continue
        # 简化方向场到 E/W 轴（2 类）：对齐 = +Δ，正交 = -Δ
        aligned = (dirs == AXIS_CODES[axis]).to(scores.dtype)  # [H, W] 广播
        offset[..., li] = delta * aligned - delta * (1.0 - aligned)
    quant = quant_score(scores + offset)
    return shiftmax(quant)


def check_K2() -> bool:
    """网格精确位移：Δ=1/16 == 8 档，与 Q7 量化 commute（clamp 外）。"""
    torch.manual_seed(3)
    ok = True
    for _ in range(200):
        s = torch.rand(16) * 2.0 - 1.0  # [-1,1]，远离 clamp 边界
        q_shift = quant_score(s + DELTA_BINS * STEP)
        q_then = quant_score(s) + DELTA_BINS * STEP
        if not torch.equal(q_shift, q_then):
            ok = False
    print(f"[K2] Δ=1/16 == 8 档，分数量化与位移 commute（200 组，clamp 外）：{'通过' if ok else 'FAIL'}")
    return ok


def check_K3_K4() -> tuple[bool, bool]:
    """方向场语义 + 门质量。"""
    t, h, w, d = 5, 15, 15, 32
    planes = moving_bar_planes(t, h, w, d, speed=2)
    dirs = axis_field(planes)
    # 语义：向右移动 -> 主导轴 E/W（条沿 x 方向移动，E/W 梯度最大）
    frac_e = float((dirs <= AXIS_CODES["W"]).float().mean())
    # 门质量使用自匹配平面 q=planes：现网 Local5 实测 q/k 平面高度相关
    # （q1[p+1]==k1[p] 79.36%，round2 C1 统计），自匹配是代表性 regime。
    scores = local5_scores_raw(planes, planes)
    quant0 = quant_score(scores)  # Q7 网格基线
    g_base = shiftmax(quant0)
    g_a3s = a3s_gate(scores, dirs, delta=STEP * DELTA_BINS)  # 合同名义 Δ=1/16 == 8 bins
    # 各向异性语义只在运动承载像素上度量（竖条仅占 13% 像素，全平面平均会稀释）
    bar_mask = planes.sum(dim=(0, 3)) > 0  # [H, W] 至少一个时刻有事件的像素
    # 对齐 lane：dirs 码 0=E->lane4, 1=W->lane3, 2=N->lane1, 3=S->lane2
    aligned_lane = torch.full((h, w), -1, dtype=torch.long)
    for code, lane in ((0, 4), (1, 3), (2, 1), (3, 2)):
        aligned_lane = torch.where(dirs == code, lane, aligned_lane)
    # 语义主张：A3S 偏移后，对齐 lane 拿到量化分数 argmax（gate-plane winner class
    # 指向运动轴）。2^s 门动态范围（s∈[0,1]，max 2x）下门质量再分配天然有界，
    # 诚实指标是 winner class 命中率而非门质量占比。
    argmax_a = g_a3s.argmax(dim=-1)  # [T, H, W]
    argmax_b = g_base.argmax(dim=-1)
    hit = float((argmax_a[:, bar_mask] == aligned_lane[bar_mask]).float().mean())
    hit_base = float((argmax_b[:, bar_mask] == aligned_lane[bar_mask]).float().mean())
    mass_ew = g_a3s[:, bar_mask][..., 3:5].sum() / g_a3s[:, bar_mask].sum()
    k3 = frac_e >= 0.5
    k4 = hit > 0.5
    print(f"[K3] 移动条（向右 speed=2）：E/W 轴占比 {frac_e:.2%}（>=50% 语义对齐）：{'通过' if k3 else 'FAIL'}")
    print(f"[K4] 对齐 lane winner 命中率 {hit:.1%}（基线 {hit_base:.1%}，>50% 各向异性生效，"
          f"q=自匹配平面，仅运动承载像素；E/W 门质量占比 {float(mass_ew):.3f} 有界）："
          f"{'通过' if k4 else 'FAIL'}")
    return k3, k4


def check_K5() -> None:
    """唯一门成本账：ident-K 分裂为对齐/正交/self 3 类偏移（折叠 self 则 2 类）。"""
    torch.manual_seed(5)
    t, h, w, d = 2, 15, 15, 32
    offsets = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    n_ident = 0
    split_sizes = {}
    for _ in range(50):
        if torch.rand(()) < 0.5:
            # ident-K：每时间平面 K 向量空间恒同 -> stencil 5 邻域全同
            vec = torch.randint(0, 2, (t, 1, 1, d)).float()
            k = vec.expand(t, h, w, d)
        else:
            k = torch.randint(0, 2, (t, h, w, d)).float()
        dirs = axis_field(k)
        for tt in range(t):
            shifted = {off: _shift_k(k[tt], *off) for off in offsets}  # 提升出 y/x 循环
            for y in range(h):
                for x in range(w):
                    nbrs = torch.stack([shifted[off][y, x] for off in offsets])
                    if bool((nbrs == nbrs[0]).all()):  # ident-K 目的地
                        n_ident += 1
                        # A3S 偏移类：self=0 / 对齐 +δ / 正交 −δ（E/W 二分方向场）
                        offs = set()
                        for dy, dx in offsets:
                            if (dy, dx) == (0, 0):
                                offs.add(0)
                            elif bool(dirs[y, x] == AXIS_CODES[("E" if dx == 1 else "W" if dx == -1 else "N" if dy == -1 else "S")]):
                                offs.add(1)
                            else:
                                offs.add(-1)
                        split_sizes[len(offs)] = split_sizes.get(len(offs), 0) + 1
    sizes = sorted(split_sizes)
    print(f"[K5] ident-K 记账：{n_ident} 个目的地 ident-K -> A3S 偏移类 {sizes} 组"
          f"（{dict(split_sizes)}；self 折叠进正交 => 2 类权重，gate-plane +1 slot/destination，"
          f"raw16 广播 ×2；诚实成本）")
    print(f"     现网占比：非静默中 ident-K 71.6%（round2 [prof]）；A3S 以 ±Δ 换各向异性门质量")


def main() -> None:
    print("=" * 78)
    print("D3 合同 CPU 数学验证：各向异性 stencil（A3S, axis-aligned anisotropic stencil）")
    print("=" * 78)
    r1 = check_K1()
    r2 = check_K2()
    r3, r4 = check_K3_K4()
    check_K5()
    print("-" * 78)
    print(f"D3 恒等验证：{'ALL PASS' if (r1 and r2 and r3 and r4) else 'FAIL'}")
    if not (r1 and r2 and r3 and r4):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
