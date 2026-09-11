# Q05 — Event-camera optical flow (independent, round 2)

Read only: `round2_absorb/SCOPE.md`, `round2_absorb/PROBLEM.md`, `IDENTITY_ATLIF.md`. No round-1 catalogs.

**Perspective.** Event-camera 2D optical flow, DSEC valid825 AEE. After absorb, the spike path is binary \(\{0,1\}\times W\), so Prosperity / Gustav / FireFly are legal complete priors on that path. DSEC *support* (who fires, in 2-D and along time) is not i.i.d. Bernoulli: it is edge-clustered, stereo-constrained, and it *slides* with motion. That structure can drive skip, packing, and last-use. **Current-frame final flow is not a free oracle** — it is the answer; using it to skip work that produces it is illegal. Legal side information: previous-frame flow, C12 motion features already in the forward, H67 coarser-scale state, ep34 / calibration geometry, pre-threshold T10 mix, residual/PED/I24 as a *different* tensor.

**Locked identity (not a contribution).** Official AT-LIF \(o=\theta\cdot H(m-\theta)\in\{0,\theta\}\). Inference: layer-shared \(\theta\) absorbed \(W\leftarrow\theta W\). Inter-layer spike path is binary select-add. Residual / PED / I24 is not analog AT-LIF amplitude. Noncausal T10 mix (PSN / lifting CSE) is continuous arithmetic *before* threshold.

**Forbidden in X.** Continuous non-absorbable \(\theta g\); HBG-RP non-absorbable int8 payload; CIM; OpenROAD-as-PPA; FPS products; 18-module laundry lists; “we skip zeros” with no DSEC/task object Prosperity does not already skip.

**Venue gates (kill every idea that misses them).** TCAS-II 5 pages. Same-port / state / backpressure *net service* \(\ge 15\%\) vs the copied prior. AEE abs \(\le 1.259\) and relative \(\le +0.005\) vs ordinary \(1.219801338\). Lifting raw \(1.232979368\) already fails the relative gate; AEE slack is not a dumping ground.

**How to read A / B / X.** A = copy the prior *whole* onto the post-absorb binary GeMM. B = hole that exists only under this freeze. X = the increment that is not a reskin of A after absorb. If ablating X (equal-density random mask, same-index temporal reuse, single last-use) leaves service and AEE unchanged, the idea is a reskin and dies.

---

## Q05-I1 — Motion-translated NRV packing (previous flow / C12, never current final flow)

**id:** Q05-I1

**one sentence:** Pack post-absorb binary spikes into Gustav-style NRV/CPTB groups along a *translated* support predicted from previous-frame flow or in-graph C12, so product-sparsity reuse follows sliding DSEC edges instead of the same neuron index.

**A:** GustavSNN NRV + CPTB and Prosperity product sparsity on the absorbed \(\{0,1\}\times W\) path (same-index co-fire groups, skip spike=0 or weight=0). FireFly-S bitmap AND as the inner product primitive.

**B:** Freeze task is event OF: support *moves*. Classification SNNs that Gustav/Prosperity target reuse neuron \(i\) at \(t\) and \(t{+}1\). DSEC edges at \(t{+}1\) live near \((x+u,y+v)\) with \((u,v)\) from *past* motion, not at the same index. Patch r1 / projection GeMM still pays full spatial fan-out unless the packer is motion-aligned. Current-frame final flow is unavailable as a packer input.

**X:** The extra operand is a warp of *previous* binary support (or C12 motion field already produced) used only to *order and group* current binary columns for reuse and to prefetch \(W\). It is not a third sparsity that copies the current spike bitmap (that is A). It is not analog \(\theta\). Residual/PED is not packed in this NRV.

**controls:**
- A-only: Gustav NRV on current bitmap, same-index across T, no warp.
- Warp-from-previous-flow vs warp-from-C12 vs warp-from-identity (zero motion).
- Equal-density random permutation of the packed groups (destroys translation, keeps density).
- Forbidden oracle: warp by current-frame *final* flow (must be worse-or-equal in a table that is *not* the letter claim; if it is the only thing that works, kill).
- Same-port always-ready vs long-backpressure traces; live NRV state bytes.

**kill-gate:** Net same-port/state/backpressure service \(<15\%\) vs A-only; AEE abs \(>1.259\) or relative \(>+0.005\); random pack \(\approx\) translated pack (reskin); identity warp \(\approx\) previous-flow warp (no motion structure); only the illegal current-final-flow warp clears 15%.

**two-sentence pitch:** After \(\theta\) is absorbed, OF spikes are binary, so Gustav/Prosperity apply — but DSEC support slides, so same-index NRV misses the co-live columns. Packing along previous-frame / C12 translation reuses those columns and frees same-port bursts without reading the current answer.

**objection:** Previous flow is wrong on occlusions and independent movers; mis-packs will either fetch useless \(W\) (no service) or drop live spikes (AEE).

**assumptions:** Previous-frame flow or C12 exists at pack time; mis-pack is a *schedule* error not a numerical skip of live spikes (live bitmap still gates the MAC; warp only orders tiles). Support correlation under ego-motion is high enough that translated groups are tighter than same-index groups.

**predictions:** Translated NRV group size \(>\) same-index NRV on projection/C12 layers; same-port occupancy drops \(\ge 15\%\) on always-ready; AEE unchanged if live bitmap remains the numerical gate. Random pack returns occupancy to A-only.

**disconfirmers:** Measured co-liveness after previous-flow warp \(\le\) same-index co-liveness; C12 not ready before the GeMM it would pack; packer state exceeds the SRAM the skip was supposed to free.

---

## Q05-I2 — 2-D edge-run encoding of DSEC support, not a flat FireFly bitmap

**id:** Q05-I2

**one sentence:** Encode post-absorb spike tiles as 2-D runs / 8-connected edge chains (the DSEC spatial structure that survives AT-LIF), and stream select-add along those runs so same-port bursts beat FireFly’s flat bitmap AND.

**A:** FireFly-S spike/weight bitmap AND plus Prosperity product sparsity on the absorbed binary GeMM. Gustav CPTB last-use on the 1-D neuron axis.

**B:** Freeze spatial shape is \(120\times 160\) (BN capture \(10\times 96\times 120\times 160\)) over driving events: after threshold, ones lie on thin edge chains, not i.i.d. bits and not a 1-D CSR of channels. Flat bitmap AND still walks empty 2-D holes inside a tile. Same-port service cares about burst length and last-use of a tile, which 1-D CPTB does not see.

**X:** The structure operand is 2-D connectivity of the *binary* support (run-length along edges, chain last-use). It is not weight sparsity (Prosperity already has that) and not “the bitmap itself.” Residual / PED / I24 stays a dense/continuous raster; only the spike GeMM is chain-scheduled. No analog payload on the chain.

**controls:**
- A-only: FireFly AND on \(N\)-bit flat tiles (e.g. \(8\times 8\) or channel-major).
- Run-length along \(x\), along gradient of previous support, along 8-connected chains.
- Shuffle 2-D ones inside each tile (keep density, kill geometry).
- Dual-path off: do not chain-schedule residual add.

**kill-gate:** Service \(<15\%\) vs A-only; AEE gate fail; shuffled-ones occupancy \(\approx\) chain occupancy (then X is just density = FireFly); chain metadata bytes \(>\) skipped same-port cycles; residual chained “for free” changes I24/PED 0-diff.

**two-sentence pitch:** Binary DSEC spikes are FireFly-legal, but they are *edges*, so a flat AND still pays holes inside the tile. Streaming select-add along 2-D runs lengthens same-port bursts and shortens tile last-use without touching the residual tensor.

**objection:** After T10 mix and several layers, support may be blob-like; edge-run degenerates to generic CSR and the idea becomes a FireFly reskin.

**assumptions:** Post-absorb ones in projection / motion layers remain anisotropic (high 2-D run length vs i.i.d. of same density). Chain metadata fits in the tile header Gustav already pays for indices.

**predictions:** Mean run length \(\gg 1\) on early motion layers; same-port idle inside FireFly tiles drops when walking runs; shuffle-inside-tile restores A-only occupancy; AEE bit-match vs ordinary if numerical GeMM is unchanged.

**disconfirmers:** Run-length histogram \(\approx\) geometric(\(p\)) with \(p=\) density (no extra 2-D structure); H67/ep34 layers already dense enough that AND occupancy \(>85\%\); chain state increases backpressure on the long-backpressure trace (both 8088).

---

## Q05-I3 — Epipolar-band tile mask from ep34 / calibration, not from current flow

**id:** Q05-I3

**one sentence:** Gate binary GeMM tiles whose 2-D support cannot meet the other-camera event band under DSEC epipolar geometry (ep34 / calibration), as a *task-structure zero* outside Prosperity product sparsity.

**A:** Prosperity skip when spike=0 or weight=0; FireFly AND of those two bitmaps. Copy both fully on the absorbed path.

**B:** DSEC is stereo driving; freeze names ep34. Generic SNN priors have no stereo geometry. A tile can be spike-nonzero in a feature map and still be incapable of a stereo correspondence (off-band). That zero is not in the spike bitmap and not in \(W\). Current-frame final flow / disparity is the answer and must not supply the band.

**X:** Third mask = epipolar band from calibration (+ optional *previous*-frame disparity to *narrow* the band). AND with the FireFly bitmap. Residual/PED path is not epipolar-gated (continuous tensor, different consumer). Pre-threshold T10 mix is not replaced by the mask; the mask sits *after* absorb on binary tiles.

**controls:**
- A-only: FireFly/Prosperity, no geometry mask.
- Calibration-only band (wide) vs previous-disparity-narrowed band vs illegal current-final-disparity band.
- Random bands of equal pixel count.
- Left-only vs both cameras (if the student is stereo).

**kill-gate:** Service \(<15\%\) vs A-only (wide band may skip too little); AEE gate fail (narrow band skips live correspondence); random equal-area band matches geometry band (reskin); only illegal current-final disparity hits 15% and AEE.

**two-sentence pitch:** After absorb, extra ANDs are free in FireFly, but the zeros that matter for stereo OF are *off-epipolar*, which product sparsity never sees. A calibration/ep34 band is not the current flow field, so it does not oracle the answer.

**objection:** Feature-map spikes are not events; epipolar on image coordinates may not transfer to \(120\times 160\) channels after T10 mix, so the band is either all-ones (no skip) or wrong (AEE).

**assumptions:** At least one binary GeMM (cost / correlation / motion C12 / ep34) is still indexed by image-plane tiles. Calibration is static. Previous disparity is optional and never required to meet AEE.

**predictions:** Calibration-only band skips a measurable tile fraction on stereo layers with AEE 0-diff; previous-disparity narrows skip further without breaking \(+0.005\); random bands of the same area move AEE first.

**disconfirmers:** ep34 is not a stereo-indexed GeMM in the student; after mix, support fills the band; same-port bottleneck is residual/BN not the stereo GeMM, so tile skip does not move the venue metric.

---

## Q05-I4 — Dual last-use: binary GeMM gate vs residual / PED / I24

**id:** Q05-I4

**one sentence:** Keep Gustav CPTB on the absorbed binary spike path, but give the continuous residual/PED/I24 path its own last-use, because the freeze’s two consumers die at different pipeline stages and a single spike last-use cannot free same-port state.

**A:** Gustav CPTB / last-use and Prosperity product sparsity for *one* spike consumer. FireFly bitmap lifetime = GeMM lifetime.

**B:** Freeze splits objects: consumer A = binary gate after absorb; consumer B = continuous residual / PED / I24 (two-window integer gates already 0-diff vs model). Patch r1 residual chain is historically expensive (old proxy \(\sim 35\%\) of activity-weighted dots; proxy \(\neq\) new cycle share). One CPTB timed to GeMM either holds residual SRAM too long or drops residual too early.

**X:** Two last-use maps, two free events, one shared SRAM port. Not “analog AT-LIF amplitude shared by two MACs.” Not int8 spike payload. Binary path still copies Gustav; residual path is the extra object those papers do not have.

**controls:**
- A-only: single last-use = max(GeMM, residual) (conservative, holds state).
- Single last-use = GeMM-only (aggressive, risks residual).
- Dual last-use as specified.
- 0-diff check on I24/PED q24 vs model; AEE.

**kill-gate:** Dual last-use net service \(<15\%\) vs conservative single last-use; AEE/I24 0-diff broken; GeMM-only last-use already \(\ge 15\%\) *and* 0-diff (then dual is unnecessary); last-use logic becomes a third consumer that increases backpressure on the 8088 trace.

**two-sentence pitch:** Prosperity/Gustav legally own the binary OF spikes after absorb, but they have one death time. The letter’s residual/PED tensor is a second live object; two last-uses are what actually returns same-port state on r1.

**objection:** If residual always outlives GeMM, dual last-use reduces to “free spikes earlier,” which is a Gustav footnote, not a letter.

**assumptions:** There exist tiles where GeMM last-use \(\neq\) residual last-use by enough cycles to matter on the same-port trace. I24/PED remains a separate integer path (already 0-diff). No new quantizer.

**predictions:** Histogram of (residual_last − gemm_last) is not a delta at 0 on r1/projection; dual last-use drops live lines \(\ge 15\%\) vs conservative single; I24 0-diff holds; AEE holds.

**disconfirmers:** Last-use difference \(\approx 0\) everywhere; the 35% residual proxy does not show up in same-port cycles; separate integer consumer model already only \(-5.78\%\) and dual last-use cannot add the rest without breaking 0-diff.

---

## Q05-I5 — Pre-threshold T10 mix as lookahead packer for post-absorb binary GeMM

**id:** Q05-I5

**one sentence:** Use the already-paid noncausal T10 mix (PSN / lifting CSE, continuous, *before* threshold) to predict which spatial neighbors will co-fire, and prefetch/pack \(W\) for the *next* absorbed binary GeMM — Prosperity only sees the bitmap after the pulse.

**A:** Prosperity/FireFly wait for the 0/1 bitmap, then AND/select-add. Gustav NRV built from observed spikes.

**B:** Freeze: noncausal T10 mix happens before threshold, then absorb. That mix is continuous arithmetic the SNN priors do not contain. Co-firing of binary spikes is the *image* of a known stencil, not independent Bernoulli. Lifting already cuts source CSE \(260\to 159\) add/sub; the leftover hole is *schedule* of the post-absorb GeMM, not more CSE algebra.

**X:** Lookahead = T10 mix output / pre-threshold \(m\) (computed anyway) \(\to\) predicted co-support groups \(\to\) NRV/prefetch for binary \(W\). Numerical MAC still uses post-threshold 0/1 after absorb (identity lock). Not sending analog \(o\) downstream. Not skipping the mix. Not treating \(\theta\) as a payload.

**controls:**
- A-only: NRV from actual bitmap, no lookahead.
- Lookahead from T10 mix vs from previous-layer bitmap vs from noise.
- Prefetch-only (never drop a live spike) vs speculative skip (illegal if it changes GeMM).
- Combine with lifting CSE on/off to show the two sit on opposite sides of the threshold.

**kill-gate:** Speculative skip needed to hit 15% *and* it moves AEE; prefetch-only service \(<15\%\) (then X is extra state, negative); lookahead groups \(\approx\) bitmap NRV (reskin: mix does not predict extra co-fire); mix not finished before the GeMM it would pack.

**two-sentence pitch:** Gustav/Prosperity are correct *after* absorb, but this net’s co-support is made *before* threshold by a noncausal mix they do not have. Using that mix as a packer lookahead — while still MACing binary spikes — is the increment.

**objection:** If \(m\) is ready, the threshold is cheap and the bitmap is ready in the next cycle; lookahead buys nothing on the same-port trace.

**assumptions:** \(W\) fetch / NRV build is on the same-port critical path and can start during mix+threshold. T10 stencil induces neighbor co-fire above i.i.d. at the same firing rate. Lifting CSE and this packer do not share the same adder resource in a way that serializes them back to 8088.

**predictions:** Mutual information between mix-neighbor energy and next-layer co-fire \(>\) that of the previous bitmap; prefetch cuts \(W\)-port stall \(\ge 15\%\) on always-ready; AEE 0-diff because the MAC gate remains the true spike.

**disconfirmers:** Threshold+bitmap is same-cycle with mix commit; neighbor co-fire \(\approx\) i.i.d.; lifting already occupies the port the prefetcher wanted; extra mix-side state cancels GeMM skip.

---

## Q05-I6 — Dense projection BN vs sparse binary GeMM on the same port

**id:** Q05-I6

**one sentence:** Schedule full-frame projection BN (real batch stats over \(10\times 96\times 120\times 160\)) as a dense reduction that *does not* follow Prosperity skip, while the absorbed projection GeMM *does*, so same-port / backpressure service comes from un-contending the two, not from skipping BN as if it were a spike MAC.

**A:** Prosperity/FireFly skip zeros on binary GeMM. Gustav last-use of spike tiles.

**B:** Freeze names native projection conv + BN + residual add, and captured BN uses *actual* batch stats on a full \(120\times 160\) raster. After absorb the conv is sparse select-add; BN is still a dense two-pass (or streaming) reduction over every spatial site. Copying Prosperity onto BN is wrong: zeros in spikes are not zeros in the BN tensor. Same-port fusion that treats both as spike-GeMM is the hole (always-ready already showed \(-22.83\%\) *part generic fusion*; long backpressure both 8088).

**X:** Two schedules, one port: sparse write-hits from binary GeMM; dense BN scan that owns the raster and the stats accumulators. Last-use of BN stats \(\neq\) last-use of spike tiles. Residual add is a third epoch, not merged into FireFly AND. No CIM. No analog AT-LIF.

**controls:**
- A-only: FireFly/Prosperity on GeMM; BN still naive full-frame after GeMM, same port, serial.
- Illegal: skip BN at spike zeros (must show AEE/BN-stat drift).
- Dual schedule: GeMM sparse + BN dense overlapped vs serial.
- Stats: actual batch vs frozen running (only as an ablation; letter keeps actual if that is the student).

**kill-gate:** Overlap service \(<15\%\) vs serial A-only; skipping BN at zeros is what “works” (illegal, AEE/stat fail); BN-dense stream increases long-backpressure (8088 stays 8088); GeMM already so sparse that BN dominates and overlap cannot reach 15%.

**two-sentence pitch:** Prosperity legally skips the absorbed projection GeMM; it cannot skip this net’s full-frame BN. The letter is the same-port schedule that lets dense BN and sparse binary GeMM coexist without pretending BN is a spike.

**objection:** If BN is fused into \(W\) (scale absorbed like \(\theta\)), there is no dense BN left and the idea dies — and \(\theta\)-absorb does *not* automatically absorb batch-stat BN.

**assumptions:** Projection BN is live at inference with actual batch stats as captured; it shares the measured same-port with projection GeMM / residual add. Batch \(N=10\) stats are not pre-absorbed into \(W\).

**predictions:** Illegal zero-skip BN moves AEE beyond \(+0.005\) or changes stats; legal overlap cuts same-port occupancy \(\ge 15\%\) without AEE move; BN accumulator last-use is full-frame even when GeMM tile last-use is sparse.

**disconfirmers:** Student inference already folds BN into \(W\); BN is on a separate port in the measured traces; overlap requires double-buffer state that cancels occupancy gain.

---

## Q05-I7 — H67 coarse-to-fine ROI on binary correlation/C12, not current-frame final flow

**id:** Q05-I7

**one sentence:** Restrict post-absorb binary GeMM for H67 / C12 matching to a search ROI from the *previous pyramid scale* or previous frame, copying FireFly AND *inside* that ROI, without reading current-frame final flow.

**A:** FireFly bitmap AND + Prosperity product sparsity over the *full* spatial/search domain of the layer.

**B:** Freeze names motion C12 / H67 / ep34. Optical-flow matching has a search neighborhood; SNN classification priors do not. Full-domain AND still scans empty correspondence tiles. Current-frame *final* flow would shrink the ROI to a cheat. Coarse-scale state is an in-graph prefix, not the answer.

**X:** ROI = warp of coarser H67 field (or previous-frame flow) \(\to\) per-tile search window \(\to\) FireFly AND only in-window. Live spikes outside the window are a numerical risk and must be counted in AEE kill. Residual/BN not ROI-gated. Not analog payload.

**controls:**
- A-only: full-domain FireFly/Prosperity.
- ROI from previous scale, from previous frame, from zero (center-only), from illegal current final flow.
- Window radii sweep; report AEE vs service.
- Equal-area random windows.

**kill-gate:** Any legal ROI that holds AEE \(\le 1.259\) and \(\le +0.005\) fails 15% service; any ROI that hits 15% fails AEE; random equal-area windows match H67 ROI (reskin); only current-final-flow ROI satisfies both gates.

**two-sentence pitch:** Binary matching after absorb is exactly FireFly, but OF matching is local in a coarse-to-fine window those papers never have. The window must come from H67’s own coarser state, not from the current final flow.

**objection:** DSEC independent movers and occlusions need large windows; the legal ROI grows back to full domain and X vanishes.

**assumptions:** H67 (or equivalent) is coarse-to-fine in the student; a coarser field exists before the finer binary GeMM. C12 search is spatially indexed. ROI is a *schedule/mask*, with a documented policy for out-of-window spikes (must not silently drop if that breaks AEE).

**predictions:** Previous-scale ROI radius that keeps AEE also cuts tile GeMMs enough for \(\ge 15\%\) same-port service on H67/C12; random windows of that area break AEE first; current-final ROI is only marginally tighter (so the letter does not need the oracle).

**disconfirmers:** No coarser field is ready (single-scale student); C12 is not a search GeMM; out-of-window mass is high and any hold-AEE policy fills the ROI; service bottleneck is r1 residual not H67.

---

## Q05-I8 — Dual-zero skip on r1 residual add (binary projection support AND continuous increment)

**id:** Q05-I8

**one sentence:** Skip r1 residual-*add* tiles only when the absorbed binary projection support is empty *and* the continuous residual increment is 0 at I24/PED q24, so Prosperity’s spike-zero skip is not illegally applied to a different tensor.

**A:** Prosperity/FireFly skip when the binary spike is 0 on the projection GeMM. Gustav last-use of those spike tiles.

**B:** Freeze: r1 residual chain historically expensive; native residual add exists; I24/PED q24 already 0-diff vs model. Residual add is \(y = x + F(x)\) with \(F\) fed by binary GeMM after absorb but \(x\) (and possibly PED) continuous. Spike=0 does not imply \(x=0\) and does not imply \(F(x)=0\) after BN. Applying A to the add is a reskin that breaks identity.

**X:** Dual-zero predicate: (post-absorb spike support in the tile = 0) \(\land\) (quantized residual increment = 0 at the already-0-diff I24/PED grid). Then skip add *and* free both last-uses (ties to I4 without becoming 18 modules: this idea is the *predicate*; I4 is the *lifetime*). Not non-absorbable int8 spikes. Not \(\theta g\).

**controls:**
- A-only: skip projection GeMM on spike=0; always perform residual add.
- Illegal: skip add on spike=0 alone.
- Dual-zero skip as specified.
- I24/PED 0-diff vs model; AEE; same-port.

**kill-gate:** Dual-zero skip rate too low to move same-port \(\ge 15\%\); illegal spike-only skip is the only one that hits 15% (AEE or 0-diff dies); dual-zero set equals spike-zero set (then residual increment is always 0 when spikes are 0 — reskin, and must be *shown* as A-only); 0-diff breaks.

**two-sentence pitch:** After absorb, Prosperity already skips empty projection GeMM; it must not skip the residual add. Skipping add only on dual-zero tiles is the OF residual object those priors do not have, and I24/PED q24 is the already-measured 0-diff grid.

**objection:** If BN/residual make \(F(x)\) nonzero almost everywhere, dual-zero never fires and r1 stays expensive.

**assumptions:** I24/PED q24 0-diff remains the numerical definition of “increment is 0.” Dual-zero is exact, not a threshold on analog residue. Same-port actually services residual adds (else skip does not move the venue metric).

**predictions:** Dual-zero tile fraction on r1 is strictly smaller than spike-zero fraction; illegal spike-only skip moves AEE or I24; legal dual-zero keeps 0-diff and AEE; if dual-zero fraction is large, same-port service can clear 15% *on traces where residual add is live*.

**disconfirmers:** Dual-zero fraction \(\approx 0\); residual add is not on the measured same-port; spike-zero \(\equiv\) increment-zero so X collapses to A; combining this predicate with I4/I6 exceeds 5 pages / looks like a module pile.

---

## Cross-idea constraints (not extra modules)

- **Oracle rule:** any table that needs current-frame *final* flow to pass both gates is dead. Previous frame, C12, coarser H67, ep34/calibration, T10 mix, I24/PED are the legal X sources.
- **Identity rule:** numerical GeMM on the spike path stays \(\{0,1\}\times W\) after absorb. Residual/PED/I24 never becomes “AT-LIF analog amplitude.”
- **Prior rule:** A is copied whole; X is only DSEC/task/dual-path/T10/BN objects those papers do not have.
- **Metric rule:** MAC skip that does not move same-port / state / backpressure is not a letter. Always-ready 6938→5354 is partly generic fusion; long backpressure both 8088 is the hostile trace.
- **AEE rule:** ordinary \(1.219801338\); abs cap \(1.259\); relative cap \(+0.005\). Lifting raw already fails relative; do not spend AEE slack on schedule approximations that change the math.
- **Page rule:** at most one of {I1, I2} packing encodings, at most one geometry mask {I3, I7}, dual last-use I4, and at most one of {I5, I6, I8} as the second object. Two objects + copied A is a letter; eight ideas here are alternatives, not a stack.
- **Reskin test (all ideas):** equal-density random mask / shuffle / same-index reuse / single last-use must *hurt* service or AEE relative to X. If it does not, X was Prosperity after absorb and the idea is withdrawn.
