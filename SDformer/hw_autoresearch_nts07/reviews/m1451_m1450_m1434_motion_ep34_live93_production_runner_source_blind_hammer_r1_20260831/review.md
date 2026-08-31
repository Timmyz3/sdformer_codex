# M1451 independent blind hammer — M1450/M1434 live93 production runner

Verdict: **FAIL_DO_NOT_CITE**. M1450 must not authorize M1452 or a production launch.

## Finding

The exact M1450 source passes its 29 author tests and all 66 independent positive/static checks. The independent campaign then injected 113 malformed runtime/policy inputs. M1450 rejected 112, but accepted one malformed GPU record:

```text
0, <exact A800 UUID>, NVIDIA A800 80GB PCIe, -1, 81920
```

`inspect_gpu` parses `memory.used=-1` and checks only `used <= 64`. It omits `used >= 0`, so the row is admitted as idle. A real NVIDIA driver is not expected to report negative memory, but this review was explicitly required to achieve zero false negatives. The gate therefore fails.

Minimal repair:

```python
0 <= used <= GPU_USED_LIMIT_MIB
```

M1450 and this failure review are immutable evidence. The repair needs an additive successor runner and fresh blind/release/final namespaces.

## What did close

- Exact M1434/M1435/M1450 pins and protected docs/359.
- Static 259/ATLIF105 minus the exact 12 H60-dead `sn2_q` nodes gives live 247/ATLIF93 and exactly 9,880 ordered records.
- Exact controller PID, start ticks, PPID1, stopped state, cwd, exe, and argv; 17/17 identity mutations rejected.
- Exact A800 UUID/name/capacity and empty compute-app list; 11/12 GPU mutations rejected.
- Eight external SHA bindings under 64 malformed-value attacks; 64/64 rejected.
- Exclusive-lease order, O_EXCL attempt before capture, double seal before success log, and no retry.
- Fail log forbids controller restore and canonical promotion and marks retained hidden staging for quarantine.
- Runner contains no signal/restore, SSH, GPU-launch, capture-launch from this review, or EDA operation.
- Exact-file allowlisting is compatible with the remote dirty tree: unrelated files are ignored, while every deployed prerequisite and authority is SHA-bound.

No SSH, remote preflight, real `nvidia-smi`, GPU work, capture, attempt creation, controller signal/restore, or EDA was performed.
