#!/usr/bin/env python3
"""Identity-only selection: no events, GT, features, predictions or quality."""
import csv
import copy
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from source_contract import HERE, require, sha, source_path

SEED = "flow-r10-block2-k1-check4-per-sequence-20261015:"
SEQUENCES = ("zurich_city_01_a", "zurich_city_05_a")
MIN_ID_GAP = 3


def inputs():
    prior = json.loads(source_path("r6_manifest").read_text())
    r9 = json.loads(source_path("r9_membership").read_text())
    canonical = json.loads(source_path("canonical").read_text())["records"]
    with source_path("roles").open(newline="") as stream:
        roles = {row["target_id"]: row["auxiliary_role"] for row in csv.DictReader(stream)}
    return canonical, roles, prior, r9


def select(canonical, roles, prior, r9):
    require(tuple(sorted(prior["check_sequences"])) == SEQUENCES, "Old development sequences differ")
    require(len(r9["target_ids"]) == len(set(r9["target_ids"])) == 16, "R9 fixed identity union differs")
    excluded = (set(prior["excluded_ids"]) | set(prior["groups"]["fit"])
                | set(prior["groups"]["check"]) | set(r9["target_ids"]))
    require(len({row["target_id"] for row in canonical}) == len(canonical), "Duplicate canonical ID")
    chosen = []
    for sequence in SEQUENCES:
        pool = [row for row in canonical if row["sequence"] == sequence
                and row["protocol_split"] == "train"
                and roles.get(row["target_id"]) == "gate_development"
                and row["target_id"] not in excluded]
        ordered = sorted(pool, key=lambda row: (
            hashlib.sha256((SEED + row["target_id"]).encode()).hexdigest(), row["target_id"]))
        accepted = []
        for row in ordered:
            index = int(row["target_id"].rsplit("_", 1)[1])
            if all(abs(index - int(other["target_id"].rsplit("_", 1)[1])) >= MIN_ID_GAP
                   for other in accepted):
                accepted.append(row)
            if len(accepted) == 4:
                break
        require(len(accepted) == 4, "Insufficient members for the frozen spacing rule")
        chosen.extend(accepted)
    require(len(chosen) == len({row["target_id"] for row in chosen}) == 8, "Fixed check8 differs")
    return chosen, sorted(excluded)


def verify_membership(path=None):
    path = Path(path) if path else HERE / "membership_v1.json"
    member = json.loads(path.read_text())
    canonical, roles, prior, r9 = inputs()
    rows, excluded = select(canonical, roles, prior, r9)
    require(member["schema"] == "flow_round10_identity_selection_v1", "Unknown R10 membership")
    require(member["seed"] == SEED and member["minimum_within_check_sequence_ID_gap"] == MIN_ID_GAP,
            "Frozen identity rule differs")
    require(member["groups"] == {"fit": prior["groups"]["fit"], "check": [r["target_id"] for r in rows]},
            "R10 selected identities/order differ")
    require(member["excluded_ids"] == excluded, "R10 exclusions differ")
    for key, digest in member["source_sha256"].items():
        require(sha(source_path(key)) == digest, "Membership provenance differs: " + key)
    return member


def main():
    canonical, roles, prior, r9 = inputs()
    rows, excluded = select(canonical, roles, prior, r9)
    paths = [HERE / "membership_v1.json", HERE / "selection_manifest_v1.json"]
    require(all(not path.exists() for path in paths), "Refuse membership overwrite")
    member = {"schema": "flow_round10_identity_selection_v1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "groups": {"fit": prior["groups"]["fit"], "check": [r["target_id"] for r in rows]},
              "excluded_ids": excluded, "seed": SEED,
              "minimum_within_check_sequence_ID_gap": MIN_ID_GAP,
              "rule": "SHA256(seed+ID), ID tiebreak; greedy first4 per 01a/05a; gap>=3 only within this new check batch; no GT/input/outcome read",
              "source_sha256": {key: sha(source_path(key)) for key in
                                  ("r6_manifest", "r9_membership", "fit20", "check64", "dev32", "canonical", "roles")},
              "selector_source_sha256": sha(__file__),
              "scope": "Reused fit20; identity-new check8 in two historically exposed, backbone-seen training sequences; no official or independent-sequence validation"}
    paths[0].write_text(json.dumps(member, indent=2) + "\n")
    # Keep the original exclusion field required by the frozen R6 validator.
    # The stronger R10 rule above is independently revalidated before prepare.
    compat = copy.deepcopy(prior)
    compat.update(created_utc=member["created_utc"], groups=member["groups"], check_seed=SEED,
                  check_rule=member["rule"], r10_membership_sha256=sha(paths[0]),
                  compatibility_only=True,
                  scope="R6 numerical/data preflight compatibility view only. Original exclusion field retained for frozen API; authoritative stronger R10 exclusions are in membership_v1.json.")
    paths[1].write_text(json.dumps(compat, indent=2) + "\n")
    verify_membership()
    print(json.dumps({"fit": len(member["groups"]["fit"]), "check": member["groups"]["check"],
                      "excluded": len(excluded), "membership_sha256": sha(paths[0])}, indent=2))


if __name__ == "__main__":
    main()
