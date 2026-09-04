# M1116C C1 full-storage/common-charge source author receipt

Verdict: **GO to a different-author source hammer only.** No VCS, DC, PT,
Formality, PTPX, GPU, remote, or full replay was run.

## Source package

- Additive wrapper: `rtl_m1116c_c1_full_storage_boundary/m1116c_m935_c1_full_storage_common_charge_boundary.sv`
- Exact mapping: `dc_handoff/manifests/m1116c_c1_full_storage_boundary_mapping_r1.tsv`
- Synthesis-only filelist: `dc_handoff/filelists/date_m1116c_m935_c1_full_storage_common_charge_dc.f`
- 3.000-ns zero-exception SDC: `dc_handoff/constraints/date_m1116c_m935_c1_full_storage_common_charge_3ns.sdc`
- Manifest-derived DC Tcl: `dc_handoff/scripts/run_dc_m1116c_m935_c1_full_storage_common_charge_candidate.tcl`
- Static checker and eight bounded tests under `verif_m1116c...` and `system_simulator/tests`.

## Physical mapping boundary

The package deliberately does not instantiate 93 or 105 area-only macros.
Frozen M935 remains byte-identical and retains its nine live parent macros:

- parent: `18,432 B`, nine internal foundry macros, included in candidate DC;
- psum: `122,880 B`, live addressed external common-charge ports;
- weight: `49,152 B`, live addressed weight/product-service ports;
- metadata/reserve: `24,448 B`, non-live identical external common charge only.

The ranges cover bytes 0 through 214,911 exactly once. Total represented
ledger capacity is `214,912 B`; external common charge is `196,480 B`; budget
margin is `30,848 B`. Only nine physical macros exist in this source top.

The DC Tcl derives counts and capacity from the TSV, reports standard-cell
logic, internal parent macro area and physical DC total separately, and labels
external common-charge area `UNMODELED_EXCLUDED`. Therefore this is not a
physically integrated 214,912-B point or full-storage total PPA.

Source-only validation passes 8/8 tests, connects all 59 frozen M935 ports
exactly once, finds zero TB/SVA/attack/behavioral-macro filelist members and
zero timing exceptions. The wrapper itself still requires independent source
hammering and later functional verification before any launch.

`docs/359` remains `dedde7ce...`.
