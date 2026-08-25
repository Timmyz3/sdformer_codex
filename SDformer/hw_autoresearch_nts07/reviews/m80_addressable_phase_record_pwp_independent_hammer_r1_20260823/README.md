# M80 addressable phase-record PWP independent hammer R1

Immutable targets:

- M80 analyzer: `d367f9f73e6b12a956c1a1983f3f9710cf5af8a0c8f4f6b7ba039a65da188a12`
- M80 result: `dec76e2afa2b91420df514157a8ba9ca0f10ccae03004c84cee2e82b9d72a7da`

The independent oracle does not import the M80 or M78 analyzers.  It rebuilds
all widths directly from the pinned M72 centers, M41 INT8 weight payloads, and
the signed-range equation.
