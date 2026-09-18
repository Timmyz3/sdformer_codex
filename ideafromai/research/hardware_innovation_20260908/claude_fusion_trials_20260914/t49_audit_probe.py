import json, sys, torch
p = "/root/private_data/work/sdformer_codex/SDformer/hw_autoresearch_nts07/system_handoff/incoming/m2041_ep34_quant_binding_inputs/checkpoint_epoch34.pth"
sd = torch.load(p, map_location="cpu")
if isinstance(sd, dict) and "model_state_dict" in sd:
    sd = sd["model_state_dict"]
elif isinstance(sd, dict) and "state_dict" in sd:
    sd = sd["state_dict"]
keys = [k for k in sd if k.endswith(".thresh")]
print("THRESH_KEYS", len(keys))
paths = sorted(k[: -len(".thresh")] for k in keys)
for x in paths:
    print("PATH", x)
# also count any name containing sn2
print("HAS_SN2", sum(1 for x in paths if "sn2" in x))
