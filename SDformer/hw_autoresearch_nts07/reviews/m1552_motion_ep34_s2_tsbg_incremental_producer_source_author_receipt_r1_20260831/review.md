# M1552 ep34 S2/TSBG incremental producer source author receipt

Status: **PASS source authoring; independent hammer required; no GPU, capture,
or release.**

M1552 is a streaming hook writer for the existing M1434/M1174 ep34 execution
path. It does not introduce a new checkpoint, model builder, sample loader, or
preprocess path. The frozen M1458 inventory binds 12 FC1, 12 FC2, and eight
PATCH modules, including their exact sample-0 execution order, channel counts,
and first-call input shapes. Loaded model modules must match before hooks are
accepted.

Each hook consumes input in chunks of at most 4,096 logical tokens. It writes
canonical level-9 zlib JSONL directly, retains zero-token rows, omits empty
source groups, and emits support/sign/nonunit bitsets plus nonzero signed codes.
Static weight address, bank, and ordinary-LRU row-buffer keys appear once in
`layers.json`. PATCH contributes only compact S1 magnitude/debt aggregates;
no complete activation or output tensor is saved.

The codebook is diagnostic. It is nearest-even signed-int8 with scale one, but
has `hardware_quant_authority=false`. Exactness ends at the captured codeword
and contributor stream. It is not model bit-exactness, an Acc24 proof, paired
AEE, or an admitted INT8 hardware path.

Python 3.6 and Python 3.12 both pass the ten-attack test. A 40-sample fake-hook
run emits 240 token records and 40 S1 rows; the unchanged M1544 validator
accepts the sealed result. The exact 32 real model names also close against a
fake loaded-model inventory.

The formal full capture is large: 474,720,000 token records. A future runner
must therefore provide an independently reviewed compressed-size estimate,
enforce at most 12 GiB and at least 16 GiB free after the estimate, and do so
before config/model/checkpoint/CUDA load. Failing the gate must stop the run;
silently sampling tokens or reducing S40 is forbidden.

This source exposes only `--describe` and `--source-self-check`.
`production_release()` rejects. No checkpoint, GPU, SSH, capture namespace,
attempt token, or retry was used. A different-author source hammer, separate
remote integration review, explicit one-shot release, and independent result
hammer remain required.
