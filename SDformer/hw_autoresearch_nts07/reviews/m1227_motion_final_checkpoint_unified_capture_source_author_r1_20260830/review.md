# M1227 final-checkpoint unified capture source author receipt

Status: `PASS_SOURCE_AUTHOR__INERT__M1228_DIFFERENT_AUTHOR_REQUIRED`.

M1227 is a source-only successor bound to the M1224 root-cause audit. It preserves the loaded model's static 259-module inventory, including 105 ATLIF modules, while enforcing the observed H60 runtime contract: 247 live modules (93 ATLIF) execute exactly once for each sample and the twelve statically installed `attn.sn_v.spiking_neuron` leaves execute zero times. The 12 attention captures per sample are audited independently, not folded into or inferred from the unified-hook arithmetic.

Every completed sample is written to a new temporary forensic directory, its members and manifest are fsynced, and the directory is atomically renamed below random staging. These snapshots never promote themselves. Canonical publication remains gated on all 40 samples and exact populations: 9,880 ordered records, 480 attention records, 640 retained payload files, 7,360 execution records, 79 operator rows at 40 calls, and 93 live ATLIF rows at 40 calls, followed by a self-verified recursive double seal.

The selected checkpoint is deliberately not fixed by this source. A future launch must bind a double-sealed final-selection result containing the exact epoch, checkpoint and configuration identities. M1227 has a new result, attempt-marker and log namespace and contains no ep29 equality gate.

The controlled suite passed 15 tests, including missing/duplicate/live/dead call mutations, attention Cartesian mutations, atomic-collision rejection, payload deletion, seal tamper, and source-only launch rejection. Import isolation loaded neither Torch, NumPy nor the M1174 substrate.

No GPU, checkpoint load, capture, remote action, EDA, release, hardware metric, or paper result was produced. A different author must perform M1228 and explicitly authorize any production release.
