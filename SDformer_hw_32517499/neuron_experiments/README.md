# Neuron Experiments

This directory contains self-contained SDFormerFlow neuron experiments.

Each experiment keeps its own entrypoints, configs, overlay code, and results.
Do not place experimental neuron code under `third_party/SDformerFlow`.

Expected layout:

```text
neuron_experiments/<experiment_id>/
    README.md
    entrypoints/
        train.py
        eval.py
    configs/
        smoke.yml
        subset.yml
        full.yml
    overlay/
        models/
            __init__.py
            STSwinNet_SNN/
                Spiking_modules.py
                experimental_neurons/
                    __init__.py
                    base.py
                    factory.py
                    single/
                    fused/
    results/
        metrics.md
        run_commands.md
```

The baseline in `third_party/SDformerFlow` is treated as read-only. Experiment
entrypoints and overlay files import unchanged baseline modules when a file is
not present in the experiment overlay.
