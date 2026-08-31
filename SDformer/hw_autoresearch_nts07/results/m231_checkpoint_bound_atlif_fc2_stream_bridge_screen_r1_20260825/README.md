# M231 checkpoint-bound ATLIF-to-FC2 stream bridge screen

The original H67 ep35 checkpoint has 12/12 scalar float32 FFN `sn2`
thresholds equal to exact `1.0` (`0000803f` little endian).  M162 independently
records the same unit-threshold identity for the completed PAFT checkpoint,
although PAFT hardware accuracy remains rejected because its best validation
result uses sample-statistic dynamic BN.

Across the frozen ten-sample FC2 payload, the 120 FC2 records contain
3,502,080,000 binary input bits, 437,760,000 packed bytes and 143,894,510
events.  A separate packed activation write and read would therefore transfer
875,520,000 bytes.  The M231 bridge instead transposes the existing two-row x
16-channel event word into the M216 four-lane x 96-channel ingress with two
pair slots.  Its state is `4*INPUT_WIDTH` bits: 192/384/768/1536 bytes for the
four H67 FC2 widths.

The M218 mean K8 service demand is 2.44x to 15.39x slower than the event-word
producer rate across stages, so the bounded stream is worth implementing.  It
is only a mean-rate screen; finite-buffer trace cycles are not yet admitted.
The traffic number is on-chip packed activation write-plus-read elision, not
DRAM traffic, energy, full-FFN speedup or system speedup.

M167 rank-3 accuracy and PAFT hardware accuracy remain unpromoted.  The bridge
is also valid for any exact binary ATLIF producer with the same event-word
boundary and does not depend on accepting either approximation.
