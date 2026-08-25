"""M51-r2 fail-closed additions to the frozen r1 streaming writer."""

from __future__ import print_function

import json
from pathlib import Path

from h67_binary_input_trace import (  # noqa: F401
    EXPECTED_TARGET_PLAN_SHA256, ExactBinaryInputTraceWriter, require,
    sha256_path, strict_json, write_binary_value_chunks,
)


MEMORY_KEYS = (
    "memory_allocated_bytes",
    "memory_reserved_bytes",
    "max_memory_allocated_bytes",
    "max_memory_reserved_bytes",
)


def require_c_order_contiguous(tensor):
    """Reject a layout that would require a whole-call contiguous copy."""
    require(hasattr(tensor, "is_contiguous") and
            bool(tensor.is_contiguous()),
            "non-contiguous hook input rejected; whole-call copy forbidden")
    return tensor


def validate_memory_snapshot(snapshot, expected_phase):
    require(isinstance(snapshot, dict) and
            snapshot.get("phase") == expected_phase and
            snapshot.get("cuda_available") is True and
            snapshot.get("capture_device_type") == "cuda",
            "invalid CUDA memory snapshot identity")
    for key in MEMORY_KEYS:
        require(isinstance(snapshot.get(key), int) and
                not isinstance(snapshot.get(key), bool) and
                snapshot[key] >= 0,
                "invalid CUDA memory field: {}".format(key))
    require(snapshot["max_memory_allocated_bytes"] >=
            snapshot["memory_allocated_bytes"] and
            snapshot["max_memory_reserved_bytes"] >=
            snapshot["memory_reserved_bytes"],
            "CUDA maximum/current memory inconsistency")


class ExactBinaryInputTraceWriterR2(ExactBinaryInputTraceWriter):
    """Adds bounded-layout and completion telemetry to the r1 protocol."""

    def bind_run_context(self, context):
        require("capture_memory" not in context,
                "capture memory must be recorded through the r2 API")
        super(ExactBinaryInputTraceWriterR2, self).bind_run_context(context)

    def record_capture_memory(self, before, after):
        require(not self.closed and not self.aborted and
                self.run_context is not None and
                "capture_memory" not in self.run_context,
                "capture memory already recorded or writer unavailable")
        validate_memory_snapshot(before, "BEFORE_CAPTURE")
        validate_memory_snapshot(after, "AFTER_FINAL_SYNCHRONIZE")
        require(after["max_memory_allocated_bytes"] >=
                before["max_memory_allocated_bytes"] and
                after["max_memory_reserved_bytes"] >=
                before["max_memory_reserved_bytes"],
                "capture peak memory decreased after reset baseline")
        self.run_context["capture_memory"] = {
            "peak_stats_reset_before_capture": True,
            "before": dict(before),
            "after": dict(after),
        }

    def close(self):
        require(self.run_context is not None and
                "capture_memory" in self.run_context,
                "refusing PASS manifest without CUDA memory telemetry")
        require(self.run_context.get("cuda_synchronization") == {
            "before_capture": 1,
            "per_sample_post_forward": 10,
            "final_pre_manifest": 1,
        }, "refusing PASS manifest without exact CUDA synchronization counts")
        return super(ExactBinaryInputTraceWriterR2, self).close()

    def abort(self, reason, failure_memory=None):
        require(not self.closed and not self.aborted,
                "second abort/closed writer")
        manifest = self.output_root / "manifest.json"
        require(not manifest.exists(),
                "refusing FAILED alongside a published PASS manifest")
        removed = []
        for partial in sorted(self.output_root.rglob("*.partial")):
            require(partial.is_file(), "unexpected partial path type")
            removed.append(str(partial.relative_to(self.output_root)))
            partial.unlink()
        receipt = self.output_root / "FAILED.json"
        require(not receipt.exists(), "failure receipt already exists")
        payload = {
            "schema": "m51_binary_input_trace_failure_r2_v1",
            "status": "FAIL_CLOSED_PARTIAL_CLEANED_NO_PASS_MANIFEST",
            "reason": str(reason),
            "completed_records": len(self.records),
            "partial_files_removed": removed,
            "partial_files_remaining": len(list(
                self.output_root.rglob("*.partial"))),
            "manifest_written": False,
            "failure_memory": failure_memory,
        }
        receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
        self.aborted = True
        return receipt
