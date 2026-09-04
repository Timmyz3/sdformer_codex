#!/usr/bin/env python3
"""Independent adversarial re-hammer for the M1167 r3 source-only binder.

This harness deliberately creates only tiny synthetic files.  It never opens a
network connection, imports a GPU framework, copies a production checkpoint, or
starts EDA.  The author test module is not imported.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[3]
HW = ROOT / "hw_autoresearch_nts07"
SOURCE = HW / "scripts/build_m1167_motion_final_checkpoint_selection_rebind_binder_r3.py"
CONTRACT = HW / "contracts/m1167_motion_final_checkpoint_selection_rebind_binder_source_r3_20260830.json"
EPOCHS = (9, 14, 19, 24, 29)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    spec = importlib.util.spec_from_file_location("m1165_independent_target", SOURCE)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


M = load_source()
CORE = M.R1


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run = root / "run"
        self.run.mkdir()
        self.config = root / "deployment.yml"
        self.config.write_text("name: independent-m1165\n", encoding="utf-8")
        self.ranking = self.run / "profile_ranking_valid825.md"
        self.aee = {9: "1.90", 14: "1.40", 19: "1.40", 24: "1.25", 29: "1.10"}
        self.policy = CORE.RunPolicy(
            run_dir=self.run,
            config=self.config,
            ranking=self.ranking,
            config_sha256=digest(self.config),
        )
        for epoch in EPOCHS:
            self.write_epoch(epoch)
        self.write_ranking((29, 24, 14, 19, 9))

    def checkpoint(self, epoch: int) -> Path:
        return self.run / f"checkpoint_epoch{epoch}.pth"

    def profile(self, epoch: int) -> Path:
        return self.run / "standard_valid825" / f"epoch{epoch}" / "spike_profile.json"

    def write_epoch(self, epoch: int) -> None:
        checkpoint = self.checkpoint(epoch)
        checkpoint.write_bytes((f"independent-checkpoint-{epoch}:" * 7).encode("ascii"))
        checkpoint_stat = checkpoint.stat()
        profile = self.profile(epoch)
        profile.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "samples": 825,
            "artifact_identity": {
                "config_path": str(self.config.resolve()),
                "config_sha256": digest(self.config),
                "checkpoint_path": str(checkpoint.resolve()),
                "checkpoint_size": checkpoint_stat.st_size,
                "checkpoint_mtime_ns": checkpoint_stat.st_mtime_ns,
                "checkpoint_sha256": digest(checkpoint),
            },
            "checkpoint_load_audit": {
                "checkpoint": str(checkpoint.resolve()),
                "checkpoint_overlay_keys": 210,
                "model_overlay_keys": 210,
                "missing_count": 0,
                "unexpected_count": 0,
                "overlay_missing_count": 0,
                "overlay_unexpected_count": 0,
            },
            "module_counts": {"ATLIFTernaryPSN": 105, "ShiftmaxAttention": 12},
            "metrics": {
                "AEE": self.aee[epoch],
                "AAE": "6.25",
                "AAE_Benchmark": "6.00",
                "AEE_PE1": "0.41",
                "AEE_PE2": "0.21",
                "AEE_PE3": "0.11",
                "AEE_outliers": "0.09",
                "DSEC_Fl": "5.5",
            },
            "total_spikes": 61_000_000_000 + epoch,
            "global_firing_rate": 0.0479,
            "dense_flops": 1_000_000.0,
            "effective_flops": 200_000.0,
            "energy_uj": 54_000.0,
        }
        profile.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def read_profile(self, epoch: int) -> dict[str, Any]:
        return json.loads(self.profile(epoch).read_text(encoding="utf-8"))

    def mutate(self, epoch: int, fn: Callable[[dict[str, Any]], None]) -> None:
        payload = self.read_profile(epoch)
        fn(payload)
        self.profile(epoch).write_text(
            json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
        )

    def write_ranking(self, epochs: tuple[int, ...], mode: str = "aee") -> None:
        lines = [
            "# independent ranking",
            "",
            f"Ranking mode: `{mode}`.",
            "",
            "| rank | epoch | AEE |",
            "|---:|---:|---:|",
        ]
        for rank, epoch in enumerate(epochs, 1):
            lines.append(f"| {rank} | {epoch} | {self.aee[epoch]} |")
        self.ranking.write_text("\n".join(lines) + "\n", encoding="utf-8")


def fresh() -> tuple[tempfile.TemporaryDirectory[str], Fixture]:
    temp = tempfile.TemporaryDirectory()
    return temp, Fixture(Path(temp.name))


def require_reject(label: str, mutate: Callable[[Fixture], None]) -> str:
    temp, fixture = fresh()
    try:
        mutate(fixture)
        try:
            M.build(fixture.policy)
        except CORE.BinderError:
            return label
        raise AssertionError(f"attack accepted: {label}")
    finally:
        temp.cleanup()


def main() -> int:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert digest(SOURCE) == contract["source_identity"]["r3_builder"]["sha256"]
    assert contract["inherited_r2_gates"]["selection"].startswith("minimum exact")

    passed: list[str] = []
    temp, fixture = fresh()
    try:
        result = M.build(fixture.policy)
        assert result["selected"]["epoch"] == 29
        assert [row["epoch"] for row in result["five_checkpoint_metric_table"]] == list(EPOCHS)
        assert [row["id"] for row in result["e0_e8_invalidation_and_rebind_targets"]] == [
            f"E{index}" for index in range(9)
        ]
        boundary = result["claim_boundary"]
        assert boundary["independent_hammer_required_before_hardware_rebind"] is True
        assert boundary["hardware_rebind_authorized"] is False
        assert boundary["hardware_replay_complete"] is False
        assert boundary["hardware_speedup"] is False
        assert boundary["system_speedup"] is False
        assert boundary["power_or_energy"] is False
        selected = result["selected"]
        assert selected["accuracy_metrics"]["AEE"] == "1.10"
        assert selected["activity"]["total_spikes"] == 61_000_000_029
        assert selected["activity"]["energy_scope"] == "spike_activity_proxy_not_hardware_energy"
        output = fixture.root / "sealed"
        M.write_receipt(output, result)
        manifest = output / "SHA256SUMS"
        assert (output / "SHA256SUMS.seal.sha256").read_text(encoding="utf-8") == (
            f"{digest(manifest)}  SHA256SUMS\n"
        )
        assert "NO_HARDWARE_REBIND_AUTHORITY" in (
            output / "RUN_COMPLETE.txt"
        ).read_text(encoding="utf-8")
        passed.append("golden-five-profile-selection-freeze-and-seal")
    finally:
        temp.cleanup()

    # Missing/stale/tampered population and identity attacks.
    attacks: list[tuple[str, Callable[[Fixture], None]]] = [
        ("extra-epoch09-alias", lambda f: (f.run / "standard_valid825" / "epoch09").mkdir()),
        ("extra-epoch009-alias", lambda f: (f.run / "standard_valid825" / "epoch009").mkdir()),
        ("extra-epoch99", lambda f: (f.run / "standard_valid825" / "epoch99").mkdir()),
        ("extra-ordinary-file", lambda f: (f.run / "standard_valid825" / "EXTRA.txt").write_text("x\n", encoding="utf-8")),
        ("missing-profile", lambda f: f.profile(29).unlink()),
        ("missing-checkpoint", lambda f: f.checkpoint(24).unlink()),
        ("samples-824", lambda f: f.mutate(19, lambda p: p.__setitem__("samples", 824))),
        ("samples-float-825", lambda f: f.mutate(19, lambda p: p.__setitem__("samples", 825.0))),
        ("stale-config", lambda f: f.config.write_text("stale\n", encoding="utf-8")),
        ("tampered-checkpoint", lambda f: f.checkpoint(29).write_bytes(b"tampered")),
        ("stale-checkpoint-sha", lambda f: f.mutate(14, lambda p: p["artifact_identity"].__setitem__("checkpoint_sha256", "0" * 64))),
        ("stale-checkpoint-size", lambda f: f.mutate(14, lambda p: p["artifact_identity"].__setitem__("checkpoint_size", 1))),
        ("stale-checkpoint-mtime", lambda f: f.mutate(14, lambda p: p["artifact_identity"].__setitem__("checkpoint_mtime_ns", 1))),
        ("stale-checkpoint-path", lambda f: f.mutate(14, lambda p: p["artifact_identity"].__setitem__("checkpoint_path", "/tmp/forged.pth"))),
        ("ranking-mode-candidate", lambda f: f.write_ranking((29, 24, 14, 19, 9), "candidate")),
        ("ranking-mixed-mode-declarations", lambda f: f.ranking.write_text(
            "Ranking mode: `candidate`.\nRanking mode: `aee`.\n"
            + "\n".join(f"| {rank} | {epoch} | x |" for rank, epoch in enumerate((29, 24, 14, 19, 9), 1))
            + "\n",
            encoding="utf-8",
        )),
        ("ranking-incomplete", lambda f: f.write_ranking((29, 24, 14, 19))),
        ("ranking-wrong-order", lambda f: f.write_ranking((24, 29, 14, 19, 9))),
        ("module-atlif-104", lambda f: f.mutate(9, lambda p: p["module_counts"].__setitem__("ATLIFTernaryPSN", 104))),
        ("module-atlif-float-105", lambda f: f.mutate(9, lambda p: p["module_counts"].__setitem__("ATLIFTernaryPSN", 105.0))),
        ("module-attention-11", lambda f: f.mutate(9, lambda p: p["module_counts"].__setitem__("ShiftmaxAttention", 11))),
        ("module-attention-float-12", lambda f: f.mutate(9, lambda p: p["module_counts"].__setitem__("ShiftmaxAttention", 12.0))),
        ("module-extra-key", lambda f: f.mutate(9, lambda p: p["module_counts"].__setitem__("Other", 1))),
        ("overlay-keys-209", lambda f: f.mutate(9, lambda p: p["checkpoint_load_audit"].__setitem__("checkpoint_overlay_keys", 209))),
        ("overlay-keys-float-210", lambda f: f.mutate(9, lambda p: p["checkpoint_load_audit"].__setitem__("checkpoint_overlay_keys", 210.0))),
        ("model-overlay-keys-209", lambda f: f.mutate(9, lambda p: p["checkpoint_load_audit"].__setitem__("model_overlay_keys", 209))),
        ("model-overlay-keys-float-210", lambda f: f.mutate(9, lambda p: p["checkpoint_load_audit"].__setitem__("model_overlay_keys", 210.0))),
        ("identity-size-float", lambda f: f.mutate(9, lambda p: p["artifact_identity"].__setitem__("checkpoint_size", float(p["artifact_identity"]["checkpoint_size"])))),
        ("identity-mtime-bool", lambda f: f.mutate(9, lambda p: p["artifact_identity"].__setitem__("checkpoint_mtime_ns", True))),
        ("identity-extra-key", lambda f: f.mutate(9, lambda p: p["artifact_identity"].__setitem__("extra", 1))),
        ("total-spikes-zero", lambda f: f.mutate(9, lambda p: p.__setitem__("total_spikes", 0))),
        ("effective-greater-dense", lambda f: f.mutate(9, lambda p: p.__setitem__("effective_flops", 2_000_000.0))),
        ("energy-zero", lambda f: f.mutate(9, lambda p: p.__setitem__("energy_uj", 0.0))),
    ]
    for counter in ("missing_count", "unexpected_count", "overlay_missing_count", "overlay_unexpected_count"):
        attacks.append((f"load-{counter}-one", lambda f, key=counter: f.mutate(24, lambda p: p["checkpoint_load_audit"].__setitem__(key, 1))))
        attacks.append((f"load-{counter}-bool-false", lambda f, key=counter: f.mutate(24, lambda p: p["checkpoint_load_audit"].__setitem__(key, False))))
        attacks.append((f"load-{counter}-string-zero", lambda f, key=counter: f.mutate(24, lambda p: p["checkpoint_load_audit"].__setitem__(key, "0"))))
        attacks.append((f"load-{counter}-float-zero", lambda f, key=counter: f.mutate(24, lambda p: p["checkpoint_load_audit"].__setitem__(key, 0.0))))
    for label, mutate in attacks:
        passed.append(require_reject(label, mutate))

    # Non-finite JSON and string forms, independently of the author's cases.
    for label, replacement in (("aee-json-nan", "NaN"), ("aee-string-infinity", '"Infinity"')):
        def attack(f: Fixture, replacement: str = replacement) -> None:
            path = f.profile(9)
            text = path.read_text(encoding="utf-8").replace('"AEE": "1.90"', f'"AEE": {replacement}')
            path.write_text(text, encoding="utf-8")
        passed.append(require_reject(label, attack))
    passed.append(require_reject(
        "activity-json-nan",
        lambda f: f.profile(9).write_text(
            f.profile(9).read_text(encoding="utf-8").replace(
                '"global_firing_rate": 0.0479', '"global_firing_rate": NaN'
            ),
            encoding="utf-8",
        ),
    ))

    # Tie: epoch 14 and 19 tie, and 14 must win deterministically.
    temp, fixture = fresh()
    try:
        fixture.aee[29] = "1.80"
        fixture.aee[14] = fixture.aee[19] = "1.00"
        fixture.write_epoch(14)
        fixture.write_epoch(19)
        fixture.write_epoch(29)
        fixture.write_ranking((14, 19, 24, 29, 9))
        assert M.build(fixture.policy)["selected"]["epoch"] == 14
        passed.append("tie-selects-lowest-epoch")
    finally:
        temp.cleanup()

    # Population/ranking policy itself is immutable.
    temp, fixture = fresh()
    try:
        wrong_epochs = CORE.RunPolicy(
            run_dir=fixture.run,
            config=fixture.config,
            ranking=fixture.ranking,
            config_sha256=digest(fixture.config),
            epochs=(9, 14, 19, 24),
        )
        try:
            M.build(wrong_epochs)
        except CORE.BinderError:
            passed.append("policy-four-epochs-rejected")
        else:
            raise AssertionError("four-epoch policy accepted")
        wrong_mode = CORE.RunPolicy(
            run_dir=fixture.run,
            config=fixture.config,
            ranking=fixture.ranking,
            config_sha256=digest(fixture.config),
            ranking_mode="candidate",
        )
        try:
            M.build(wrong_mode)
        except CORE.BinderError:
            passed.append("policy-candidate-mode-rejected")
        else:
            raise AssertionError("candidate policy accepted")
    finally:
        temp.cleanup()

    print(json.dumps({
        "status": "PASS_M1165_INDEPENDENT_M1167_R3_SOURCE_REHAMMER",
        "checks": len(passed),
        "passed": passed,
        "source_sha256": digest(SOURCE),
        "contract_sha256": digest(CONTRACT),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
