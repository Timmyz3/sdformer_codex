"""H9 SOP/profile entrypoint with ATLIFTernaryPSN and Shiftmax installation."""

from __future__ import annotations

import sys
from pathlib import Path


INIT_ANCHOR = """    model.to(device)
    model.init_weights()
    if args.checkpoint:
        model = load_model(str(args.checkpoint), model, device, remap=args.remap, test=True)
"""

INIT_PATCH = """    model.to(device)
    model.init_weights()
    def _h9_is_overlay_key(key):
        markers = (".linear_v.", ".bn_v.", ".sn_v.", ".spiking_neuron.thresh", ".spiking_neuron.center")
        return any(marker in key for marker in markers)

    def _h9_load_profile_checkpoint(checkpoint, model, device, remap=None):
        from utils.utils import _extract_pretrained_state_dict, load_model as _baseline_load_model, load_pretrained_interpolate, remap_pretrained_keys_swin
        checkpoint = str(checkpoint)
        if not checkpoint:
            return model
        if not Path(checkpoint).is_file():
            return _baseline_load_model(checkpoint, model, device, remap=remap, test=True)
        pretrained_model = torch.load(checkpoint, map_location=device, weights_only=False)
        pretrained_dict = _extract_pretrained_state_dict(pretrained_model, test=True)
        if remap == "v2":
            pretrained_dict = remap_pretrained_keys_swin(model, pretrained_dict)
        elif remap == "v1":
            load_pretrained_interpolate(model, pretrained_dict)
            del pretrained_model
            torch.cuda.empty_cache()
            print("Model restored from local checkpoint " + checkpoint + "\\n")
            return model
        overlay_checkpoint_keys = [key for key in pretrained_dict.keys() if _h9_is_overlay_key(key)]
        incompatible = model.load_state_dict(pretrained_dict, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        overlay_missing = [key for key in missing if _h9_is_overlay_key(key)]
        overlay_unexpected = [key for key in unexpected if _h9_is_overlay_key(key)]
        print(
            f"[H9] profile load audit: checkpoint_overlay_keys={len(overlay_checkpoint_keys)}, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
        if missing:
            print(f"[H9] profile missing keys sample: {missing[:12]}")
        if unexpected:
            print(f"[H9] profile unexpected keys sample: {unexpected[:12]}")
        if overlay_unexpected:
            raise RuntimeError(
                "[H9] overlay checkpoint keys were not registered before profile load: "
                + str(overlay_unexpected[:20])
            )
        if overlay_checkpoint_keys and overlay_missing:
            raise RuntimeError(
                "[H9] checkpoint contains overlay parameters but matching profile model keys are missing: "
                + str(overlay_missing[:20])
            )
        del pretrained_model
        torch.cuda.empty_cache()
        print("Model restored from local checkpoint " + checkpoint + "\\n")
        return model

    if config.get("atlif_ternary_psn", {}).get("enabled", False):
        from models.STSwinNet_SNN.atlif_ternary_psn import atlif_ternary_summary, install_atlif_ternary_psn
        installed_h9 = install_atlif_ternary_psn(model, config.get("atlif_ternary_psn"))
        print(f"[H9] installed ATLIFTernaryPSN before profile load: {len(installed_h9)} modules")
        print(f"[H9] neuron summary before profile load: {atlif_ternary_summary(model)}")
    if config.get("bsa_attention", {}).get("enabled", False):
        from models.STSwinNet_SNN.bsa_attention import install_shiftmax_attention, register_shiftmax_pickle_compat, shiftmax_attention_summary
        register_shiftmax_pickle_compat()
        installed_h9_bsa = install_shiftmax_attention(model, config.get("bsa_attention"))
        print(f"[H9] installed Shiftmax attention before profile load: {len(installed_h9_bsa)} modules")
        print(f"[H9] attention summary before profile load: {shiftmax_attention_summary(model)}")
    if args.checkpoint:
        model = _h9_load_profile_checkpoint(str(args.checkpoint), model, device, remap=args.remap)
        if config.get("atlif_ternary_psn", {}).get("enabled", False):
            from models.STSwinNet_SNN.atlif_ternary_psn import atlif_ternary_summary
            print(f"[H9] neuron summary after profile load: {atlif_ternary_summary(model)}")
        if config.get("bsa_attention", {}).get("enabled", False):
            from models.STSwinNet_SNN.bsa_attention import shiftmax_attention_summary
            print(f"[H9] attention summary after profile load: {shiftmax_attention_summary(model)}")
"""


def main() -> None:
    experiment_root = Path(__file__).resolve().parents[1]
    repo_root = experiment_root.parents[1]
    baseline_root = repo_root / "third_party" / "SDformerFlow"
    overlay_root = experiment_root / "overlay"
    profile_script = repo_root / "tools" / "profile_sops.py"

    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(baseline_root))
    sys.path.insert(0, str(overlay_root))

    source = profile_script.read_text()
    if INIT_ANCHOR not in source:
        raise RuntimeError("Could not patch tools/profile_sops.py for H9 loading")
    source = source.replace(INIT_ANCHOR, INIT_PATCH, 1)
    code = compile(source, str(profile_script), "exec")
    exec(code, {"__name__": "__main__", "__file__": str(profile_script)})


if __name__ == "__main__":
    main()
