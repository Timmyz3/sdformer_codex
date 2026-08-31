# M1155HC independent M1153HC first-D0 replay

PASS for the frozen first D0 call only. A different implementation replayed all 4,465,036 psum updates without importing the subject cache or port classes. It exactly reproduces the M1111 baseline at 8,930,072 backing operations and 17,863,747 cycles, and the one-entry direct candidate at 96,000 backing operations and 9,025,999 cycles: 1.979143472097x local speedup and 98.9249806720% backing-operation reduction.

The exact key is `(timestep,destination,output_block)`. All 4,417,036 non-cold references have reuse distance zero and there are 48,000 cold keys. Phase is not silently omitted: it is uniquely encoded by destination parity, and adding it explicitly leaves 48,000 keys. Omitting output block collapses the population to 12,000 keys; omitting timestep collapses it to 4,800 and is invalid.

The one entry is 96 lanes times Acc24 = 2,304 data bits = 288 bytes, plus 16 metadata bits, for 2,320 bits or 290 bytes. A 288-bit interpretation holds only 12 lanes. Existing allocation is 243,200 bytes; adding 290 gives 243,490 bytes, leaving 2,270 bytes inside 240 KiB.

All 48,000 fills, 47,990 dirty evictions, 10 timestep-terminal flushes, 48,000 total writes, and 48,000 dense commits are charged. Terminal symbolic flush has zero mismatches. Descriptor, weight, compute and commit populations are identical across baseline and candidate; 29,622,568 source terms are not substituted for the 4,465,036 psum-update events.

This remains one old-checkpoint D0 CPU call. It authorizes only a D0-D3 one-call-each cross-layer CPU replay, not RTL, VCS, DC, a headline, or system speedup.
