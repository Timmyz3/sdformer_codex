# TCAS-II cover letter draft (fill names before send)

**To:** Editor-in-Chief, IEEE Transactions on Circuits and Systems II: Express Briefs

**Manuscript:** Finite-Lifetime Single-Port Product Capture and Context-Safe Weight Broadcast for Event-Driven Spiking Optical Flow

**Article type:** Express Brief (regular)

**EDICS:** Digital circuits and systems; neuromorphic / neural-network circuits (please pick the current portal codes)

Dear Editor,

Please consider this Express Brief on two exact digital 28-nm execution islands for a frozen binary-event optical-flow network.  Sparse firing does not by itself reduce parent-state or weight-delivery traffic under finite capacity and port constraints.  C1 captures repeated products only while an exact-subset parent remains resident in a single-1RW store; C2 combines an equal-bandwidth typed-K8 datapath with context-safe token-set broadcast (TSBG), sharing a weight delivery while preserving private signed products, destinations, and Acc24 state.

On 51.84 million same-ledger source rows C1's cycle model reduces component time by 40.99% (1.6945×) versus strongest-zero skipping.  Its mapped nine-SRAM island occupies 166,514 µm² and meets 3-ns setup/hold.  Across 2,880 fixed real-activity workloads spanning all 12 FC1 and all 12 FC2 layer identities, same-port/cache TSBG VCS reduces post-load execution by 45.49% (1.8345×) and scalar weight reads by 58.13%; the matched logic-area increment is 0.0118%.  K8 provides 4.541× directed-throughput/logic-area versus equal-bandwidth K1×8.  These are component-level results; we do not claim silicon measurements, whole-network speedup, or FPS.

The manuscript is formatted as 5 pages in the required 4.5+0.5 format and will not be a simultaneous conference submission.  C2/TSBG post-route hold and matched power remain disclosed as open unless the final independently reviewed campaigns close them.  Author metadata and the statement that the work is not under review elsewhere must be confirmed immediately before upload.

Sincerely,
[Corresponding author, ORCID, e-mail]
