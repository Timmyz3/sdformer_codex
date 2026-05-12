# Experiment Template

Copy this template into `neuron_experiments/<experiment_id>/` before wiring a
new neuron experiment.

Use `entrypoints/train.py` and `entrypoints/eval.py` for launch logic. Use
`overlay/` only for Python modules that replace or extend baseline imports.
