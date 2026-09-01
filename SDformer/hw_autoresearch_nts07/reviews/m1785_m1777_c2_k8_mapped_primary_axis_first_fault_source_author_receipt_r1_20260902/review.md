# M1785 source-only M1777 K8 mapped first-fault audit

Status: `PASS_M1785_SOURCE_AUTHORING__READY_FOR_DIFFERENT_AUTHOR_M1786_HAMMER__NO_EDA`.

M1777 is sealed as a K8 case-0 mapped simulation failure at 27,000 ps. The root event is the unchanged M1684 exact `$isunknown` check on `{protocol_error, numeric_overflow, stale_response_seen, endpoint_fault[7:0]}`. One source cover matched before termination; endpoint, result, and done covers did not. The later Python checker traceback is downstream of that simulator termination and is not the root cause. There are zero checked SAIF files and zero PTPX runs, so no partial power or energy axis is citable.

The existing fatal collapses eleven bits into one message, so a specific unknown field is not honestly proven yet. M1785 preserves the exact M1684 and M1334 monitors and adds a non-driving public-field observer. It records source, endpoint, result, done, top fault, endpoint fault, and status/header fields with stable numeric codes. It samples continuously on fault transitions and at `posedge + 1 ps`; the latter carries forward M1594's race-free observation method without reusing the old-netlist conclusion. Six named registered fault taps are supplemental cone evidence only.

The seven K8 compile warnings are half-adder instances with unused carry outputs. They are not suppressed, but neither their text nor static connectivity proves them to be the public fault root. The eight endpoint fault registers are explicitly reset in the exact M1334 memory model. The M1609/M1613 evidence proves the isolated compactor boundary, not every integrated K8 service/adapter/public mapped cone.

No repair is selected before localization. If a public combinational fault output is unknown while its registered taps remain binary, the owning RTL module must expose sampled sticky state while retaining illegal-request gating, followed by fresh synthesis. Numeric or functional-output X/Z requires a real reset or valid-isolation repair. Testbench masking, `initreg`, `force`, assertion suppression, and X-to-zero coercion remain forbidden.

Author checks passed under CPython 3.6 and 3.10: 10/10 mutations on each interpreter, byte-identical source-check output, contract double seal, and M1777 failure double seal. This author ran no license query, VCS, simv, SAIF, PTPX, DC, or Formality and created no M1785 attempt/result. A different M1786 author must independently hammer the package before the one future VCS compile plus one simv localization attempt.
