# M518 r11 fixed-wrapper author handoff

Date: 2026-08-27  
Status: `AUTHOR_HANDOFF_ONLY__INDEPENDENT_STATIC_REVIEW_REQUIRED__NO_TOOL_AUTHORIZATION`

## Sole repair

r10 consumed its one-shot authorization at a pre-tool environment-format gate.
The effective launch environment was not preserved, so the sealed review does
not guess whether the cause was an unset variable, whitespace, or a mistyped
name. r10 is immutable and must not be retried.

r11 changes only the runner launch plumbing and adds one immutable wrapper:

- before any result `mkdir`, the runner reports only each SHA environment
  variable name, presence, length and lowercase-hex-64 regex result; it never
  prints the supplied value;
- unset, wrong length, wrong format, live-SHA mismatch, missing admission,
  failed member/outer seal, and semantic mismatch are distinct fail-closed
  gates;
- `authorized_invocations` must satisfy Python
  `type(value) is int and value == 1`, so JSON Boolean `true` is rejected;
- the wrapper rejects a present `M518_RUN_DIR`, validates both admission seals,
  computes the live runner, wrapper and admission SHAs, validates all admission
  semantics, clears caller SHA overrides, and exports the exact three variable
  names itself. The operator must not retype SHA assignments.

No launch admission exists at this handoff. A different reviewer must first
return a P0/P1-zero source-only review; root may then create and double-seal the
one-shot r11 admission. The only permitted later operator entry point is the
fixed wrapper.

## Frozen identity

```text
RTL       8a7ec11843b1b9c13c22ab679f69d70f73a8f5874f9ccee51c8873f4f7f142d6
SVA       89d4d711e2913e49ed14d3368c786f069cf11b2ec3f89371dd8582358917c1f5
TB        8877512040c0677de58bc88c1cacd8056bb6f20026c24e3794f633682d962e56
filelist  09e435600ded03f79ff4eb1462135ce67d4987725e07111b230fbbd1a2f22fea
contract  f0b8b2379138fa52d4abfe0b82884e8bfaf10d7a83ae7f1bc04badb071903690
identity  a355d6ce053fc064aef21850de677c633d61e54637f869b75fd35da1690c754b
runner    4e50a78cae0a4a05cad50865468e8321897d7ce74d851212551d5ccfa4d660a8
wrapper   798f433ff0ee790058b86b781e01de9fd021c0947cdf49c8bfcc0e95480c3650
```

The 42-entry runner SHA map has zero mismatch. Both scripts pass `bash -n`;
their four embedded Python blocks compile statically. The canonical r11 result
path and r11 admission are absent.

## Author execution boundary

The author did not execute the wrapper, runner, wrong-TB negative control,
VCS, DC, Formality, PT, PTPX, or any open-source EDA. This package admits no
compile, simulation, numeric, cycle, physical, power, energy, speedup, PPA,
system, or headline claim.

`docs/359` remains SHA256
`dedde7ce44c3e595098f25ce6550dc0f6dfd66ce7227bcffd3dab0426a7bdfc4`.
