# H67 Motion vs H81 no-motion training fairness

Status: `PASS_RECIPE_LEVEL_CONTROL_NOT_STEP_PAIRED`.

- Crop training uses the same parent checkpoint and identical config after removing experiment/note and `binary_motion_xor_alpha`; H67 uses `0.25`, H81 uses `0.0`.
- Full-resolution geometry, model, neuron, optimizer, augmentation and evaluation contracts match. H81 runs the registered 40-epoch no-motion control.
- H67's historical full-resolution path is a five-stage audited rescue/continuation, whereas H81 is uninterrupted. This supports a recipe-level control, not a bit-exact step-paired causal claim.
- Final metrics remain pending until H81 valid825 finishes.
