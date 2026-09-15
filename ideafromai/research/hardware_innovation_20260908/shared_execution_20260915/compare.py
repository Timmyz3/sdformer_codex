"""Join completed local DUT receipts; no cross-function equality claim."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    control = [json.loads(line) for line in
               (ROOT / "r8_reference/comparison.jsonl").read_text().splitlines()]
    rows = []
    lines = [
        "共同资源预算下的完整组件服务（Verilator；无Fmax/PPA），2026-09-15。",
        "",
        "周期均从接受go到最后I24退休，包含go拍、静态配置、源/原点装载；取冷配置首遍。"
        "三组分别为首帧held64、disjoint64、18序列各两瓦片共36。后者是同一上游source/id，"
        "各自按不同学生函数产生独立gold。不是两个函数逐位等价的无损加速比。",
        "",
        "| 工作集 | 背压 | R8 native | R8借用 | R8 count | R8 bitmap | 空间共享 | 空间共享+借用 | 空间相对最佳R8变慢 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in ("held", "disjoint", "sequences"):
        for bp in (0, 1):
            c = {r["mode"]: r for r in control if r["stage"] == stage and
                 r["stall"] == bp and r["command"] == 0}
            assert set(c) == {2, 4, 21, 7}
            candidates = {}
            for borrow in (0, 1):
                path = ROOT / f"spatial_rr/results_{stage}_b{borrow}_s{bp}.jsonl"
                rr = [json.loads(line) for line in path.read_text().splitlines()]
                r = next(r for r in rr if r["repeat"] == 0)
                candidates[borrow] = r
            nums = [c[m]["service_cycles"] for m in (2, 4, 21, 7)]
            nums += [candidates[b]["total_cycles"] + candidates[b]["go_cycles"]
                     for b in (0, 1)]
            best_control = min(nums[:4])
            best_candidate = min(nums[4:])
            slowdown = best_candidate / best_control - 1
            lines.append(f"| {stage} | {bp} | " + " | ".join(f"{n:,}" for n in nums) +
                         f" | {slowdown:.2%} |")
            rows.append(dict(stage=stage, backpressure=bp, control_service=dict(
                zip(("native", "borrow", "count", "bitmap"), nums[:4])),
                candidate_service=dict(zip(("shared", "shared_borrow"), nums[4:])),
                slowdown_vs_best_tested_control=slowdown,
                control_mac_issues=c[7]["core_mac_issues"],
                candidate_q2_issues=candidates[1]["core_q2_issues"],
                candidate_consumer_waits=candidates[1]["consumer_join_wait_cycles"],
                candidate_arbitration_stalls=candidates[1]["core_arbitration_stalls"]))
    lines += ["", "两树都只有一套共享源/W/Z/psum服务、8×32位ALU、8×19×13位乘法，"
              "一个完整FP32 identity→J20→wide64→I24消费者；双context的Z物理声明共4160B，"
              "cache许可384B/context。进位切分、选择及静态配置不同，不是等面积/等Fmax网表。"
              "详细共同许可与实占见各树README及独立资源审阅。", "",
              "当前空间端点停止作为性能主线；保留其已有质量结果。"
              "额外的连续Q2发射、转换及状态流量尚未被共享排程抵消。"
              "负结果针对该phase3/rank/控制器组合，不是否定所有空间或张量分解。", ""]
    (ROOT / "COMPARISON.md").write_text("\n".join(lines))
    (ROOT / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
