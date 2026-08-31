# M1208 ep29 unified capture symlink-root successor — source author receipt

Status: `PASS_SOURCE_ONLY__FRESH_DIFFERENT_AUTHOR_HAMMERS_REQUIRED`.

The additive repair changes only the sample resolver used during the inherited M1180 capture. It permits exactly the repository component `data/Datasets/DSEC` to be an absolute symlink when both the raw link target and canonical resolved target equal `/root/private_data/SothisAI/dataset/Console/DSEC/main/DSEC`. Every other component and every leaf remains non-symlink; each of the forty leaves retains exact byte-count and SHA-256 admission.

The predecessor M1180 attempt is immutable and never retried. M1208 owns disjoint attempt, result, log, and PASS namespaces. The resolver override is restored in `finally`.

Twelve controlled tests cover original rejection, exact successor admission, target drift, traversal, nested and leaf symlinks, content drift, namespace separation, and restoration. This receipt does not authorize remote launch. A fresh different-author source hammer, release package, release hammer, secure transfer, and one exact remote launch are still required.
