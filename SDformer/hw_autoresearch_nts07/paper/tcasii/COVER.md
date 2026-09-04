# TCAS-II cover letter draft (fill names before send)

**To:** Editor-in-Chief, IEEE Transactions on Circuits and Systems II: Express Briefs

**Manuscript:** Exact Product Capture under a Finite Single-Port Parent Store for Event-Driven Spiking Optical Flow

**Article type:** Express Brief (regular)

**EDICS:** Digital circuits and systems; neuromorphic / neural-network circuits (please pick the current portal codes)

Dear Editor,

Please consider this Express Brief on a digital 28-nm execution island for a frozen binary-event optical-flow network.  Sparse firing does not by itself reduce parent-state traffic when the working set is finite and the store is single-ported.  The brief’s sole circuit contribution, C1, captures exact repeated products under that contract: a child row reuses only a still-resident exact-subset parent, issues the XOR residual, reconstructs the signed product by addition, elides dead writes, and commits atomically.

On 51.84 million same-ledger source rows the cycle model reduces component time by 40.99% (1.6945×) versus strongest-zero skipping, while same-coordinate bit skipping is 1.003×.  The mapped nine-SRAM island occupies 166,514 µm², meets 3-ns setup/hold, reports a 64-row/253-cycle mixed-corner energy of 29.08 mW / 22.07 nJ, and passes 16,549 mapped-to-mapped Formality compare points.  These are prelayout, pre-macro-extracted component results; we do not claim silicon measurements or whole-network FPS.

The manuscript is 5 pages in the required 4.5+0.5 format, has not been submitted elsewhere, and is not a simultaneous conference submission.  Related weight-broadcast work is omitted because its hold and power are not closed.

Sincerely,
[Corresponding author, ORCID, e-mail]
