#!/usr/bin/env python3
"""T51: 把 T50/T51 的 overlay 修复推到 sd5ai 并做冒烟验证。

修的三处（本地已改，见 results/T51_FIXES_AND_ATTENTION.md）：
  F1  bsa_attention.py  `_binary_event_ste` 被定义了两次，后者静默劫持前者
      → 4800 那个改名 `_bounded_binary_event_ste`（+ 它自己的 3 个调用点）
  F2  bsa_attention.py  motion XOR 的 `k_event` 用 `_binary_event_ste`(gt(0))，
      负事件被压成 0 → 换 `_ternary_sign_ste`（二值输入下前向逐位相同）
  F3  installer.py      跳过 Shiftmax overlay 结构上不可达的 24 个目标
      （12 `.sn2_q` 从不被调用 + 12 `.attn_sn` 输入上限 < 阈值）

用法：
  python t50_apply_overlay_fixes.py --dry-run     # 只打印要传什么
  python t50_apply_overlay_fixes.py --apply       # 备份 + 上传
  python t50_apply_overlay_fixes.py --smoke       # 3-step 冒烟，查 num_modules=81
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]                     # .../sdformer_codex
LOCAL_OVERLAY = REPO / "SDformer/neuron_experiments/H9_bipolar_self_attention/overlay/models/STSwinNet_SNN"
LOCAL_ENTRY = REPO / "SDformer/neuron_experiments/H9_bipolar_self_attention/entrypoints"
REMOTE_TREE = "/root/private_data/work/sdformer_codex/SDformer/neuron_experiments/H9_bipolar_self_attention"
REMOTE_OVERLAY = f"{REMOTE_TREE}/overlay/models/STSwinNet_SNN"
SSH = [sys.executable, str(HERE / "tools/sd5ai_ssh.py")]

FILES = [
    ("bsa_attention.py", f"{LOCAL_OVERLAY}/bsa_attention.py", f"{REMOTE_OVERLAY}/bsa_attention.py"),
    ("installer.py", f"{LOCAL_OVERLAY}/atlif_ternary_psn/installer.py",
     f"{REMOTE_OVERLAY}/atlif_ternary_psn/installer.py"),
    ("train.py", f"{LOCAL_ENTRY}/train.py", f"{REMOTE_TREE}/entrypoints/train.py"),
    # 评测路径的守卫与 train.py 是**两个不同文件**，漏了它会让所有 pre-fix
    # checkpoint（含 ep34 锚点）的 eval 直接 RuntimeError。
    ("h9_load_audit.py", f"{LOCAL_OVERLAY}/h9_load_audit.py",
     f"{REMOTE_OVERLAY}/h9_load_audit.py"),
]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, (proc.stdout or "") + (proc.stderr or "")


def remote(cmd: str) -> tuple[int, str]:
    return run(SSH + [cmd])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if not (args.dry_run or args.apply or args.smoke):
        ap.error("pick one of --dry-run / --apply / --smoke")

    for name, local, remote_path in FILES:
        if not Path(local).exists():
            print(f"MISSING local file: {local}")
            return 1
        print(f"{name}:\n  {local}\n  -> {remote_path}")

    if args.dry_run:
        return 0

    if args.apply:
        stamp = "20260917_t50"
        rc, out = remote(
            " && ".join(
                f"cp {remote_path} {remote_path}.bak.{stamp}" for _, _, remote_path in FILES
            )
        )
        print("backup rc=%d %s" % (rc, out.strip()[:200]))
        if rc != 0:
            return 1
        for name, local, remote_path in FILES:
            rc, out = run(SSH + ["--put", local, remote_path])
            print("put %s rc=%d %s" % (name, rc, out.strip()[:120]))
            if rc != 0:
                return 1
        rc, out = remote(
            "cd %s && md5sum %s" % (
                REMOTE_TREE,
                " ".join(remote_path.replace(REMOTE_TREE + "/", "")
                         for _, _, remote_path in FILES),
            )
        )
        print("remote md5:\n" + out)
        print("local  md5:")
        run(["md5sum"] + [local for _, local, _ in FILES])
        return 0

    if args.smoke:
        # 3-step 冒烟：只看 num_modules 与 installer 的 skip 提示，不落盘。
        # --max-steps 是 config 键（runtime.max_train_steps），由 retrain 包装脚本写进 yml。
        rc, out = remote(
            "cd %s && /opt/conda/envs/sdformerflow/bin/python -u /root/t49_ternary_retrain.py "
            "--name t51_fix_smoke --epochs 1 --max-steps 3 >/dev/null 2>&1; "
            "L=%s/results/t51_fix_smoke/train.log; "
            "tr '\\r' '\\n' < $L | grep -E 'skipped|num_modules|asymmetric_scale_modules|"
            "official_atlif_modules|Traceback|Error' | head -20"
            % (REMOTE_TREE, REMOTE_TREE)
        )
        print(out)
        return rc

    return 0


if __name__ == "__main__":
    sys.exit(main())
