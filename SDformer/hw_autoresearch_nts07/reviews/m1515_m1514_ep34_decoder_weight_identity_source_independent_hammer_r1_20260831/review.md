# M1515 independent decoder-weight identity hammer

Status: `PASS_M1515_M1514_EP34_DECODER_WEIGHT_IDENTITY_SOURCE_ZERO_FALSE_NEGATIVE__M1516_SOURCE_ONLY`.

M1515 reran the 10 author tests and source self-check, then performed two real read-only CPU checkpoint loads. The checkpoint SHA remained exact before and after reading. An independent audit confirmed the exact `model_state_dict`-only root, 921-key `OrderedDict`, four exact decoder keys and shapes, float32 contiguous little-endian encoding, no bias/suffix aliases/shared storage, all four content hashes, and the 28,560,384-byte total.

All 26 independent checks passed. All 250 mutations were rejected: ten checkpoint-object attacks, six authority-identity attacks, and 234 exact-contract attacks. No payload, GPU, EDA, SSH or remote action occurred.

This review permits only fresh M1516 export/materializer source authoring. It does not authorize execution, materialization, production, automatic retry, or any cycle/traffic/speedup/energy/PPA/Table-A/paper claim.
