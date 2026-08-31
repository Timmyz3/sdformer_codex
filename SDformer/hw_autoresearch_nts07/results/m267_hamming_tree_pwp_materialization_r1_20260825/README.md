# M267 exact Hamming-tree PWP materialization

M267 removes the checkpoint-derived fixed12 PWP payload from off-chip storage.
For each 16-source partition it builds a deterministic minimum-Hamming spanning
tree over implicit zero and the 16 M77 patterns, then reconstructs every child
PWP from its parent with signed 96-lane INT8 weight-vector updates.

The proof covers all four PAFT bottleneck Conv operators, 1,728 partitions,
27,648 full 768-lane PWP vectors (221,184 96-lane blocks), and 21,233,664 scalar
PWP values.  Direct summation and tree reconstruction have zero mismatch and the
same canonical digest.  The observed range is `[-1026, 960]`, safely inside
signed12.

The design eliminates the 31,850,496-byte PWP payload.  Including weights,
patterns, and explicit four-byte tree descriptors, traffic/storage per complete
catalog pass falls by 59.729448%.  The largest tree needs 50 source flips; with
eight 96-lane output blocks, the conservative serial metadata/weight load plus
generation bound is 787 cycles, below the frozen M251 960-cycle next-partition
DMA envelope.  M251 reports compute binding on all 17,280 phases, so its admitted
module cycles remain unchanged.

This is an exact derived-payload-elision and reuse result, not a new arithmetic
or system speedup.  Energy requires DRAM/PTPX measurement, and RTL/VCS/DC,
Formality, SRAM macro, paper PPA, and headline admission remain false.
