# E4b Official-style TS-LIF

This experiment keeps the E4 official TS-LIF dynamics but changes the training
protocol to better match the TS-LIF source repository:

- Adam instead of AdamW.
- Gradient clipping set to `1.0`.
- TS-LIF dynamics parameters use a separate high learning rate.
- PSN-pretrained SDFormer backbone uses a lower learning rate.
- Baseline files under `third_party/SDformerFlow` remain untouched.

Source TS-LIF reference:

`/root/private_data/work/optimization_sources/neuron_optimization/TSLIF_TS-LIF/TS-LIF`

Official source commit:

`a59826a6c7f62d0f16edbafdbb28db65bebd9f69`
