# G1 Partial Sparse Gate

G1 keeps the PSN baseline architecture and wraps only six low-sensitivity, high-SOP layer0 Swin nodes with scalar hard straight-through gates.

The baseline `third_party/SDformerFlow` tree is not edited.

Target nodes:

- `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.attn.proj_sn`
- `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn1`
- `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.0.mlp.sn2`
- `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.attn.proj_sn`
- `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn1`
- `sttmultires_unet.encoders.swin3d.layers.0.swin_blocks.1.mlp.sn2`

Initial G1a policy:

- gates start closed with `init_logit=-2.0`
- backbone is frozen
- only six gate logits are trainable
- `reg_lambda=0.02` keeps pressure toward fewer open gates
