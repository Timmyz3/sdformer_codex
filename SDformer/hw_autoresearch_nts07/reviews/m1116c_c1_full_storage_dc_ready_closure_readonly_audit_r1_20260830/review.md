# M1116C — C1 full-storage DC-ready closure audit

Verdict: **STOP current DC launch; GO to exactly one additive source-authoring
successor.** No RTL was changed and no EDA was run.

## Assertion boundary

M934's rule remains the default: any assertion/fatal/error stops launch. M959
is usable only with its mandatory qualifier: the UNIT_DELAY run deliberately
injects the frozen M923 wrong-parent fault and therefore contains exactly one
expected `ap_candidate_after_active` failure at 10,168,500 ps, with zero
unexpected assertion failures. M963 narrowly supersedes M934 for that one
identity/time/attack tuple only. It does not turn M959 into a clean or
zero-assertion regression.

The production DC filelist has two synthesizable sources and contains no TB,
SVA or negative-test injector. Thus the expected negative assertion is
functional admission evidence, not an event that may execute or be masked in
production DC.

## Why the current source is stopped

The exact M935 RTL, M962 filelist/SDC, TSMC28 setup/hold views and 3.000-ns
coordinate are intact. M1006 is a valid component reference: 68,421.148925
um2 logic plus 78,825.243164 um2 for nine parent macros equals 147,246.39209
um2, with setup WNS +0.001795 ns.

That top represents only 18,432 B of parent SRAM. The admitted M1102/M1114
capacity coordinate is 214,912 B within a 245,760-B budget. Therefore 196,480
B remain outside the current physical top. Parent+psum+weight imply 93 known
macros / 190,464 B, while another 24,448 B of metadata and reserve still lack
an exact physical/common-charge mapping.

## Only allowed next step

Author one additive M1116C storage-boundary wrapper/package without editing the
frozen M935 RTL. It must bind live storage ports, map every byte exactly once,
derive macro counts from that mapping, and report logic and each macro class
separately. Dummy/tied-off area macros and pairing the old component area with
the 1.75917x CPU ratio are forbidden.

After source authoring: independent source hammer, zero-argument launcher,
different-author final launcher hammer/external tuple, at most one root DC
attempt, then an independent result hammer. Old M955/M962 identities may not
be retried or relabeled.
