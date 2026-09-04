# M1574 — M1565 permit provenance successor author receipt

M1574 closes the two M1565 P0 seams without touching a production result or
running checkpoint, GPU, SSH, capture, RTL, or EDA work.  The public production
issuer and its closure-bound authority now accept only the resolved output path.
They derive the exact 32-layer/40-sample inventory internally and obtain free
space only through `shutil.disk_usage` on the resolved output parent.

Production and synthetic permits are separate exact closure-bound types.  Their
provenance is encoded by type and by a fixed receipt value; it is not accepted as
a constructor/issuer argument.  `production_inventory=True` requires the exact
production type before permit consumption or namespace creation.  The regression
therefore rejects a synthetic permit even when it carries the exact frozen 32
layers and 40 samples, and rejects a production permit in synthetic mode.

Both CPython 3.10.18 and 3.6.8 pass the same 30-attack synthetic regression.  A
controlled `disk_usage` witness confirms the production path performs the real
query and rejects equality at the strict 16-GiB post-result boundary.  The
44,640,000 FC-token and 430,080,000 PATCH-token-equivalent production estimate is
unchanged; no production payload was generated or modified.

This is author evidence, not an independent admission.  It authorizes only a
fresh different-author dual-runtime rehammer.  Remote-wrapper authoring and any
actual capture still require subsequent separately sealed gates; no performance,
accuracy, RTL/EDA, energy, or paper claim is admitted here.
