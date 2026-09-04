# M1007 C1 matched-common-charge/address-timed replay source receipt

Status: `PASS_M1007_MATCHED_REPLAY_SOURCE__NO_FULL_REPLAY_NO_EDA`

M1007 freezes a memory-bounded source path for the future full C1 comparison. It reproduces the frozen M505 dead-write-only 1RW parent recurrence cycle by cycle and emits parent operation/address, issue, forwarding, freeing, queue and pending-response state. Four small directed cases match M505 on cycle, read, write, forward, issue and stall counts.

The source also makes the fairness obligation executable. Candidate, strongest-zero and same-coordinate-bit must make the same include/exclude decision for psum, weight, source, DMA and commit. Included resources require the same capacity, ports, latency and logical access multiset. That equality is only necessary: each schedule still must be merged with the same service model before a total-cycle ratio exists.

The 214,912-byte packing remains blocked. It may be admitted only after complete frozen-ledger coverage shows zero same-cycle conflicts for every paired psum bank group, zero 1RW conflicts for the single weight group and zero overlap between both weight half slots. The synthetic negative oracle demonstrates that the gate rejects conflicts; it says nothing about the real full trace conflict rate.

No 51.84M-row replay, VCS, DC, PT, PTPX, GPU or remote execution occurred. No matched total-cycle result was created. The M528 `1.7467534301x` number remains CPU same-ledger evidence and is not RTL cycle evidence. Capacity is not area or timing, and linear area extrapolation remains prohibited.

Reproduction commands:

```bash
/opt/anaconda3/envs/pytorch310/bin/python3.10 -m unittest system_simulator/tests/test_m1007_c1_matched_common_charge_address_replay_source.py
/opt/anaconda3/envs/pytorch310/bin/python3.10 system_simulator/scripts/check_m1007_c1_matched_common_charge_address_replay_source.py
/opt/anaconda3/envs/pytorch310/bin/python3.10 system_simulator/scripts/m1007_c1_matched_common_charge_address_replay_source.py --self-test
```

`docs/359_DATE终局冻结_20260813.md` remains unchanged at SHA256 `dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
