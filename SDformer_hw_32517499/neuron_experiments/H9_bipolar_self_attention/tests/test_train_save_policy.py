from pathlib import Path


from neuron_experiments.H9_bipolar_self_attention.entrypoints import train


def test_patched_training_source_compiles_with_sparse_save_policy():
    repo = Path(__file__).resolve().parents[3]
    baseline = repo / "third_party/SDformerFlow/train_flow_parallel_supervised_SNN.py"
    source = train._patch_source(baseline.read_text(encoding="utf-8"), baseline)

    compile(source, str(baseline), "exec")
    assert 'get("save_only_force_epochs", False)' in source
    assert "epoch in force_save_epochs" in source
    assert 'epoch == config["loader"]["n_epochs"] - 1' in source
    assert 'get("state_save_epochs", [])' in source
    assert 'runtime_cfg.get("seed")' in source
    assert "torch.cuda.manual_seed_all" in source


def test_sparse_save_policy_is_opt_in():
    assert 'get("save_only_force_epochs", False)' in train.SAVE_PATCH
    assert "if save_only_force_epochs:" in train.SAVE_PATCH
    assert "epoch_loss < best_loss" in train.SAVE_PATCH
    assert "if should_save_state" in train.STATE_SAVE_PATCH


def test_seed_policy_is_opt_in():
    assert 'runtime_cfg.get("seed")' in train.SEED_PATCH
    assert 'runtime_cfg.get("deterministic", False)' in train.SEED_PATCH
