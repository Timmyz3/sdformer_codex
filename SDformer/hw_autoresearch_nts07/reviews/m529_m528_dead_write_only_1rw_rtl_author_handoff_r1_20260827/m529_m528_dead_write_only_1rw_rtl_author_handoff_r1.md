# M529 / M528 dead-write-only 1RW RTL author handoff

Status: complete source-only package; no compile, simulation, synthesis, formal, power, CPU-production, or GPU run was performed or authorized.

The package implements one structure only: a dynamic 64-row M504-compatible exact-subset matcher, two 64×32 ping-pong directories and live maps, stable `(popcount(original), row_id)` execution, one earliest-parent lookahead, immediate-next deadline hold, a no-consume-credit two-entry response queue, and a read-XOR-write parent scratch built from exactly nine generated 128×128-bit 1RW macros. Dead finals suppress only the scratch write; every live row still writes, including same-address new-value forwards. Psum commit and row completion remain atomic.

The M474 arithmetic boundary is now explicit: every non-synthetic signed12 source lane must be a sign extension of signed INT8, and the 16-bit source mask permits at most 16 sources per row. Format, row-output, psum-output, and protocol failures block the entire accepted event.

Macro binding is fail-closed:

- VCS must compile the checksum-verified foundry slow `.v` (`8343acf...`) and may not substitute a local SRAM model.
- DC/STA must link the checksum-verified slow `.db` (`cd8c205...`), must not compile the behavioral `.v`, and must reject inferred parent registers or a macro count other than nine.
- Formality must match the nine macro instances as black boxes/cutpoints and compare control, address, and all nine contiguous 128-bit slice boundaries.

No combined PVRF, single-use store elision, concurrent read/write scratch, second lookahead, second structural variant, register-array scratch fallback, decoder, or full-network scheduler is present.

Final source identities are in `m529_m528_dead_write_only_1rw_rtl_author_handoff_r1.json` (SHA256 `86dd591948fc7d09850110f280e2f884185e9368d2a0508a66b7a2ae4f119d5a`). The exact-SHA VCS runner is source only and self-blocks until a separate independent static PASS and double-sealed launch admission exist.

Open physical risks are intentionally not hidden: the 64-way matcher, two 1152-bit response slots, directory/bitmap logic, resident-psum cut, and nine-macro integration have no VCS/DC/STA/Formality/PTPX evidence yet. Therefore all RTL verification, PPA, energy, full-network, system-speedup, and paper-headline claims remain false.

