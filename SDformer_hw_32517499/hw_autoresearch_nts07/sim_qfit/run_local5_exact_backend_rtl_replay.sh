#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT_DIR="${INPUT_DIR:-results/local5_fullres_bb1e4_postg0_profile100_20260805}"
VECTOR_DIR="${VECTOR_DIR:-tb_qfit/vectors/local5_bb1e4_active_projection_postg0_all4800}"
RESULT_DIR="${RESULT_DIR:-results/local5_bb1e4_exact_backend_rtl_all4800_20260810}"
mkdir -p "$RESULT_DIR"
rm -f "$RESULT_DIR/report.json" "$RESULT_DIR/report.md" \
  "$RESULT_DIR/result_sha256.txt" "$RESULT_DIR/complete.json"

if [[ "${REUSE_VECTORS:-0}" == "1" ]]; then
  python - "$VECTOR_DIR/manifest.json" <<'PY'
import json
import hashlib
import sys
from pathlib import Path

def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

manifest_path = Path(sys.argv[1])
manifest = json.load(manifest_path.open(encoding="utf-8"))
if (
    manifest.get("schema") != "local5_active_projection_postg0_vectors_v1"
    or manifest.get("selection", {}).get("method") != "manifest_order_all_groups"
    or int(manifest.get("selection", {}).get("groups", 0)) <= 0
):
    raise SystemExit("拒绝复用：vector manifest不是有效的all-groups合同")
groups = int(manifest["selection"]["groups"])
shape = manifest.get("shape", {})
if shape != {
    "height": 15, "width": 15, "planes": 2, "sources": 450,
    "head_dim": 32, "out_dim": 2,
}:
    raise SystemExit("拒绝复用：vector shape合同失配")
expected = {
    "input_valid": (groups * 450, 5),
    "input_active": (groups * 450, 5),
    "input_k": (groups * 450, 32),
    "input_gates": (groups * 450, 45),
    "input_weights": (groups * 32 * 2, 8),
    "expected_acc": (groups * 450 * 2, 32),
    "expected_active": (groups, 16),
    "expected_terms": (groups, 32),
    "expected_updates": (groups, 32),
}
artifacts = manifest.get("artifacts", {})
if set(artifacts) != set(expected):
    raise SystemExit("拒绝复用：vector artifact集合不完整")
for name, row in artifacts.items():
    path = manifest_path.parent / row["file"]
    entries, width = expected[name]
    if (
        not path.is_file()
        or path.stat().st_size <= 0
        or row.get("entries") != entries
        or row.get("width") != width
        or sum(1 for _ in path.open("rb")) != entries
        or sha256(path) != row["sha256"]
    ):
        raise SystemExit(f"拒绝复用：artifact SHA失配: {path}")
for path_key, sha_key in (
    ("source_manifest", "source_manifest_sha256"),
    ("source_payload", "source_payload_sha256"),
):
    path = Path(manifest[path_key])
    if not path.is_file() or sha256(path) != manifest[sha_key]:
        raise SystemExit(f"拒绝复用：source SHA失配: {path}")
PY
  echo "REUSE validated vector manifest: $VECTOR_DIR/manifest.json" \
    | tee "$RESULT_DIR/vector_generation.log"
else
  python scripts/generate_local5_active_projection_postg0_vectors.py \
    --input-dir "$INPUT_DIR" --output-dir "$VECTOR_DIR" \
    --all-groups --weight-mode synthetic \
    | tee "$RESULT_DIR/vector_generation.log"
fi

GROUP_COUNT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["selection"]["groups"])' "$VECTOR_DIR/manifest.json")"
if [[ "$GROUP_COUNT" -le 0 ]]; then
  echo "非法group数: $GROUP_COUNT" >&2
  exit 1
fi

RTL=(
  tb_qfit/tb_qfit_local5_active_projection_postg0.sv
  rtl_qfit/qfit_local5_1rw_active_projection_tile.sv
  rtl_qfit/qfit_dual_color_relation_frontier_sync.sv
  rtl_qfit/qfit_dual_color_word_skipper_index.sv
  rtl_qfit/qfit_sync_relation_bank.sv
  rtl_qfit/qfit_fakeram45_relation_bank_450.sv
  rtl_qfit/qfit_source_multicast_term_builder_fifo2.sv
  rtl_qfit/qfit_source_multicast_term_builder.sv
  rtl_qfit/qfit_local5_1rw_projection_backend.sv
  rtl_qfit/qfit_local5_color_map.sv
  rtl_qfit/qfit_direct_1rw_acc_bank.sv
  rtl_qfit/qfit_gasr2c_acc_bank.sv
  rtl_qfit/qfit_single_port_acc_memory.sv
)

{
  verilator --version
  python --version
  /usr/bin/time --version
  uname -a
} >"$RESULT_DIR/tool_versions.txt"

sha256sum "${RTL[@]}" \
  scripts/generate_local5_active_projection_postg0_vectors.py \
  scripts/summarize_local5_exact_backend_rtl_replay.py \
  sim_qfit/run_local5_exact_backend_rtl_replay.sh \
  "$VECTOR_DIR/manifest.json" \
  >"$RESULT_DIR/source_sha256.txt"

python - "$RESULT_DIR/compile_contract.json" "$GROUP_COUNT" "$VECTOR_DIR" <<'PY'
import json
import sys
from pathlib import Path

value = {
    "schema": "local5_exact_backend_compile_contract_v1",
    "top": "tb_qfit_local5_active_projection_postg0",
    "simulator": "verilator --binary --timing -Wno-fatal",
    "groups": int(sys.argv[2]),
    "vector_dir": str(Path(sys.argv[3]).resolve()),
    "common_parameters": {"NEW_1RW_BACKEND": 1, "RELATION_READ_LATENCY": 1},
    "modes": {"direct": 0, "gasr_reset": 1},
    "cycle_scope": "projection_start through flush/done; excludes post-done Acc32 readback",
}
Path(sys.argv[1]).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

for mode in 0 1; do
  name="direct"
  if [[ "$mode" == "1" ]]; then
    name="gasr_reset"
  fi
  obj="$RESULT_DIR/verilator_${name}_obj"
  verilator --binary --timing -Wno-fatal \
    --top-module tb_qfit_local5_active_projection_postg0 \
    -Mdir "$obj" -GNEW_1RW_BACKEND=1 -GMODE="$mode" \
    -GGROUPS="$GROUP_COUNT" -GRUN_GROUPS="$GROUP_COUNT" "${RTL[@]}" \
    >"$RESULT_DIR/${name}_compile.log" 2>&1
  /usr/bin/time -v "$obj/Vtb_qfit_local5_active_projection_postg0" \
    +VECTOR_DIR="$VECTOR_DIR" \
    >"$RESULT_DIR/${name}_all_groups.log" \
    2>"$RESULT_DIR/${name}_time.log"
done

python scripts/summarize_local5_exact_backend_rtl_replay.py \
  --manifest "$VECTOR_DIR/manifest.json" \
  --direct-log "$RESULT_DIR/direct_all_groups.log" \
  --gasr-log "$RESULT_DIR/gasr_reset_all_groups.log" \
  --output-dir "$RESULT_DIR"

sha256sum "$RESULT_DIR"/{vector_generation.log,tool_versions.txt,source_sha256.txt,compile_contract.json,direct_compile.log,direct_all_groups.log,direct_time.log,gasr_reset_compile.log,gasr_reset_all_groups.log,gasr_reset_time.log,report.json,report.md} \
  >"$RESULT_DIR/result_sha256.txt.tmp"
mv "$RESULT_DIR/result_sha256.txt.tmp" "$RESULT_DIR/result_sha256.txt"

python - "$RESULT_DIR/complete.json" "$GROUP_COUNT" "$RESULT_DIR/result_sha256.txt" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

result_hashes = Path(sys.argv[3])
value = {
    "schema": "local5_exact_backend_rtl_replay_complete_v1",
    "status": "PASS",
    "groups": int(sys.argv[2]),
    "result_sha256_file": str(result_hashes.resolve()),
    "result_sha256_file_sha256": hashlib.sha256(result_hashes.read_bytes()).hexdigest(),
}
Path(sys.argv[1]).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
PY

echo "PASS Local5 exact backend RTL replay groups=$GROUP_COUNT"
