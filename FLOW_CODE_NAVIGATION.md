# Optical-flow code navigation

The Git base is `75fe802af6987411a54f7396223d178779ae4aa0`, including the five preserved September AT-LIF repair commits. This source update adds the later optical-flow research code and required small configurations.

- Original model and data loaders: `SDformer/third_party/SDformerFlow/`.
- H9 training, model overlays and experiment configurations: `SDformer/neuron_experiments/H9_bipolar_self_attention/`.
- Native optical-flow baseline and data-contract checks: `research/flow_story_20261012/baseline/` and `research/flow_story_20261012/data_encoding/`.
- Later algorithm and RTL experiments: dated `research/flow_story_20261013` through `research/flow_story_20261026` directories.
- Preserved AT-LIF implementation work: `ideafromai/research/hardware_innovation_20260908/`.

The existing server runtime remains under `/data/flow_a` and `/data/flow_new`. Checkpoints, datasets, bulk replay captures, generated EDA trees, historical result JSON, retrievable upstream clones and optional Python packages are separate resources. Historical scripts can contain absolute runtime paths; source publication is not a claim that every experiment runs without its recorded input assets and environment.
