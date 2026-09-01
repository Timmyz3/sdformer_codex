# M1766 M1757 failure attestation

I performed a read-only audit after the unique M1757 campaign had already
failed.  I did not invoke VCS, `simv`, PrimeTime, PTPX, a license query, or any
other EDA action.  I did not modify M1757/M1758/M1759 sources, sealed evidence,
the consumed-attempt namespace, the failure quarantine, the unsealed private
build, or docs/359.

The mapped simulation log contains exactly one public-port PASS token and one
counter line.  The generated SAIF is present and now bound by hash, but the
canonical result is absent and PTPX was never entered.  The existing failure
JSON records `saif_files=0` because M1757 increments that counter only after
`validate_saif`; this receipt distinguishes the one generated file from zero
validated files.

The private build is still `UNSEALED_DO_NOT_CITE`.  Binding its current compile
log, runtime log and SAIF hashes is forensic evidence, not promotion of that
directory.  Its file mtimes precede the failure JSON and show no observed later
rewrite at audit time; this metadata is not used as an independent provenance
proof.  The sealed M1759 authority, consumed-attempt seal, failure seal and the
exact runner control flow establish the one-attempt lineage.

The accompanying independent M1766 diagnosis proves that comment-aware parsing
is necessary but insufficient: nonzero TX remains.  No parser-only successor
may use this SAIF to launch PTPX.
