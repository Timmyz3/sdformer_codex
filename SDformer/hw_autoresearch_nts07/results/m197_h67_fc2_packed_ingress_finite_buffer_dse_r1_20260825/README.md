# M197 packed-ingress finite-buffer DSE

M197 replays the 120 frozen H67 FC2 payloads with F={1,2,4,8}
nonzero96 descriptors accepted per fill cycle and B={2,3,4} window buffers.
It exactly reproduces all M196 F1 wall-cycle points.

The selected F2/B2 screen takes 89,013,553 cycles, or 1.096550x versus the
legacy W1/F1/B2 point.  The fair decomposition is important: F2 packing alone
gives 1.056447x, while pair fusion adds only 1.037961x versus an iso-width W1
frontend.  F4/B2 and F8/B2 reach 1.129507x and 1.140761x versus legacy, with
clear diminishing returns.

This DSE begins at a precompacted nonzero-descriptor stream.  It therefore
does not prove that a real bitmap scanner/compactor can sustain F2.  F2 is an
investigation point, not an admitted RTL, physical, FC2, FFN or system speedup.
