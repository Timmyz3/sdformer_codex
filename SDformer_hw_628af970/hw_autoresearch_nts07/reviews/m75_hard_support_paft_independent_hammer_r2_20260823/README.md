# M75 hard-support / admission independent hammer R2

R2 targets the following immutable inputs:

- `pattern_paft.py`: `bf15a2ea328a16d1d8c676de11a041fc3768c6717bcf73a21c1c6e3c2378087f`
- production validator: `d882af175785cdcfb3a6ec5478039969a465bf156abae2e201d040b2208d59cd`
- M75 r5 receipt: `9832aa7c96a8a8699cde2bd29e249c124d29684ed42d7e2669e8b8c164fd7aae`

R1 remains sealed unchanged.  R2 uses an independently written arithmetic
oracle plus isolated local negative tests against the production loader.  It
does not use a network, modify training artifacts, or launch training.
