#!/usr/bin/env python3
"""D3 (h88/local5_a3s) short 验证 launcher（GPU 空闲后手动/队列启动）。

延续 run_h82_class_major_ttx_ft15 / run_motion_t5_quotient_short 的纪律：
fcntl 锁 + status.log + SHA 冻结 + train.py subprocess。本脚本**不自动排队**：
GPU 忙时记录并退出 0，由 DATE 审计 agent / GPU 队列协调者在 GPU 空闲后调用。

启动条件（写死在 status.log 与冻结合同中）：
  * bsa_attention.py 追加区 SHA 冻结（0 删除行回归已验，见
    D3_LOCAL5_A3S_IMPLEMENTATION_20260818.md §6）
  * CPU 单测全绿：tests/test_local5_a3s_scores.py + _forward.py（K1-K5）
  * 锚点 checkpoint_epoch44.pth（Local5 valid825 AEE 1.2819@ep44）存在且 SHA 冻结
  * GPU 空闲（nvidia-smi memory.used < 4096MiB 且无已知训练/评测进程）

训练入口（train.py 既有接口，--finetune 1 续训）：
  --config  dsec_fullres_w15_H88_local5_a3s_ft5_short_20260818.yml
  --prev_runid  Local5 锚点 ep44
  --save_path  ROOT/checkpoint_epoch{}.pth

通过标准（验证实验 1，合同草案 D3）：
  loss 曲线形态与 Local5 基线同量级（不塌、不减半退化）；step-1k 后单调下降。
  结构保障：Δ 起调档 = 现网 Local5 逐位恒等（K1），a3s_delta_warmup_steps
  内从 0 线性升至 Δ=1/16（8 档）。
"""

from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


REPO = Path(__file__).resolve().parents[3]
EXP = Path(__file__).resolve().parents[1]
CONFIG = EXP / "configs/generated/dsec_fullres_w15_H88_local5_a3s_ft5_short_20260818.yml"
INIT = (
    EXP
    / "results/dsec_fullres_w15_H66d_local5_bb1e4_equal_plus20_ep50_20260812/checkpoint_epoch44.pth"
)
ROOT = EXP / "results/dsec_fullres_w15_H88_local5_a3s_ft5_short_20260818"
LOCK = Path("/tmp/sdformer_h88_local5_a3s.lock")
STATUS = ROOT / "status.log"
CONTRACT = REPO / "neuron_autoresearch/D3_LOCAL5_A3S_IMPLEMENTATION_20260818.md"
PY = Path("/opt/conda/envs/sdformerflow/bin/python")
OPERATOR = EXP / "overlay/models/STSwinNet_SNN/bsa_attention.py"
GPU_IDLE_MIB = 4096
KNOWN_BUSY_PIDS = (
    "run_dsec_fullres_window9",
    "run_date11_followup_queue",
    "run_h86_member_delta_ft15",
    "run_nts11bd_rank2_guardian",
    "run_motion_t5_quotient_short",
    "train.py",
    "formal_eval",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(message: str) -> None:
    line = f"[{datetime.now(timezone.utc).isoformat()}] {message}"
    print(line, flush=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    with STATUS.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return int(float(out.strip().splitlines()[0].strip()))
    except Exception:
        return None


def gpu_idle() -> bool:
    used = gpu_used_mib()
    if used is None:
        return True  # 无 nvidia-smi（CPU 机器）——由队列协调者裁决
    try:
        procs = subprocess.check_output(["ps", "-eo", "cmd"], text=True)
    except Exception:
        procs = ""
    busy = [name for name in KNOWN_BUSY_PIDS if name in procs]
    idle = used < GPU_IDLE_MIB and not busy
    record(
        f"GPU gate: used={used}MiB busy_pids={busy} -> "
        f"{'IDLE' if idle else 'BUSY (short run NOT started)'}"
    )
    return idle


def freeze_contract() -> None:
    payload = {
        "schema": "d3_local5_a3s_operator_contract_v1",
        "status": "OPERATOR_FROZEN_TRAINING",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "d3": [
            "K1 delta=0 bit-identical to deployed Local5 gate (float + Q7)",
            "K2 delta=1/16 == 8 Q7 bins, quant commute (clamp outside)",
            "K3 direction field = 3x3 temporal XOR-gradient argmax, 2bit/px",
            "K4 aligned-lane winner hit rate > baseline (honest metric)",
            "K5 ident-K unique gate 1 -> 3 offset classes",
        ],
        "forbidden": [
            "window size change (Swin partition untouched)",
            "direction field with gradient (fixed bitmap, detach)",
            "motion bias lane (Local5 discipline: binary_motion_xor_alpha ignored)",
        ],
        "parent": {
            "line": "Local5",
            "checkpoint": str(INIT),
            "checkpoint_sha256": sha256(INIT),
            "anchor_aee": "1.2819@ep44",
        },
        "artifacts": {
            "operator_py": {"path": str(OPERATOR), "sha256": sha256(OPERATOR)},
            "config": {"path": str(CONFIG), "sha256": sha256(CONFIG)},
            "contract_md": {"path": str(CONTRACT), "sha256": sha256(CONTRACT)},
        },
    }
    contract_json = CONTRACT.with_suffix(".json")
    contract_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    record(f"FROZE operator contract {contract_json} sha={sha256(contract_json)}")


def main() -> int:
    if not CONFIG.is_file():
        raise FileNotFoundError(f"D3 short config missing: {CONFIG}")
    if not INIT.is_file():
        raise FileNotFoundError(f"Local5 anchor missing: {INIT}")
    if not OPERATOR.is_file():
        raise FileNotFoundError(f"operator missing: {OPERATOR}")
    if not CONTRACT.is_file():
        raise FileNotFoundError(f"implementation note missing: {CONTRACT}")
    ROOT.mkdir(parents=True, exist_ok=True)
    with LOCK.open("w", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("D3 short trainer already active", flush=True)
            return 0
        if (ROOT / "checkpoint_epoch4.pth").is_file():
            record("ALL COMPLETE D3 short already has epoch4")
            return 0
        if not gpu_idle():
            return 0  # GPU 忙：不启动、不排队，留给队列协调者
        freeze_contract()
        env = os.environ.copy()
        env.update(
            {
                "SDFORMER_USE_MLFLOW": "0",
                "SDFORMER_MLFLOW_MODEL_LOGGING": "0",
                "SDFORMER_SNN_BACKEND": "cupy",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            }
        )
        command = [
            str(PY),
            "-u",
            str(EXP / "entrypoints/train.py"),
            "--config",
            str(CONFIG),
            "--prev_runid",
            str(INIT),
            "--save_path",
            str(ROOT / "checkpoint_epoch{}.pth"),
            "--finetune",
            "1",
        ]
        record("START " + " ".join(command))
        log = ROOT / "train.log"
        with log.open("w", encoding="utf-8") as log_handle:
            log_handle.write("$ " + " ".join(command) + "\n")
            log_handle.flush()
            proc = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
            log_handle.write(f"\n[d3-short] exit_code={proc.returncode}\n")
        if proc.returncode:
            raise RuntimeError(f"D3 short train failed; log={log}")
        record("ALL COMPLETE D3 short ft5")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
