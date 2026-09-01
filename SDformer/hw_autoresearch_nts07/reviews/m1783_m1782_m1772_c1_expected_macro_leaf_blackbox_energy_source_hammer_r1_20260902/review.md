# M1783 independent source hammer

## Verdict

PASS, 99/100. P0/P1/P2 are all zero. M1784 may be authored; M1783 alone does not authorize or execute a campaign.

## Independent findings

- The sealed M1772 failure and private forensic SHA values match. Mapped VCS completed a 253-cycle public-port workload and produced a 117,690-form-per-tag, all-TX-zero SAIF. PrimeTime linked successfully, then the old broad black-box gate stopped before `read_saif`; no power result exists.
- M1782 preserves the black-box gate and narrows legality to exactly the fixed nine `u_parent_scratch/g_slice_[0..8]__u_parent_sram` leaves with ref `TS1N28HPCPHVTB128X128M4S`, `is_hierarchical=false`, and `is_black_box=true`. It does not admit an arbitrary set of nine black boxes.
- Ten independent inventory mutations were rejected: missing, extra, same-count replacement, wrong ref, hierarchical, false black-box attribute, duplicate name, both header corruptions, and malformed row.
- The mapped netlist independently contains exactly the nine expected SRAM instances. The live Tcl performs its set gate after link and before SAIF annotation. Whole-component reporting is unchanged; there is no macro subtraction or datasheet-energy addition.
- A future campaign is fresh1: one UNIT_DELAY mapped compile, one simulation, one SAIF, and one PTPX. No M1772 private binary, SAIF, or work directory is reused.
- Public two-bank warmup, 117,690-form/TX-zero SAIF validation, exact 100% net and leaf annotation, and power conservation remain in the execution chain.
- The author suite and the independent hammer pass under CPython 3.6.8 and 3.10.18. No EDA, license query, attempt, result, or release was created. `docs/359` remains frozen.

## Boundary

This is source authorization only. A separately pinned and double-sealed M1784 is required before one fresh run. Any resulting component power remains non-citable until a different-author result hammer passes.
