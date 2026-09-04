#!/usr/bin/env python3
"""Build the fail-closed M520 H67 paper metric registry.

The registry is deliberately not a speedup calculator.  It inventories values at
their measured scope, leaves missing paper-table cells null, and refuses stale or
ambiguous evidence before writing a sealed output directory.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence
from typing import Set, Tuple, Union


STATUS = "REGISTRY_READY__SYSTEM_TABLE_BLOCKED"
SCHEMA = "m520_h67_paper_metric_registry_v1"
CONFIG_SCHEMA = "m520_h67_paper_metric_registry_config_v1"
CONTRACT_SCHEMA = "m520_h67_paper_metric_registry_contract_v1"
ROW_IDS = [
    "fixed_dense",
    "exact_bit_sparse",
    "prosperity_official_external_iso_workload",
    "phi_like",
    "ours_c1",
    "ours_c2",
    "ours_c3",
    "ours_a1",
]
METRICS = [
    ("compute_cycles", "cycle/selected_window"),
    ("preprocess_stall_cycles", "cycle/selected_window"),
    ("mem_stall_cycles", "cycle/selected_window"),
    ("total_cycles", "cycle/selected_window"),
    ("sram_read_bytes", "byte/selected_window"),
    ("sram_write_bytes", "byte/selected_window"),
    ("dram_read_bytes", "byte/selected_window"),
    ("dram_write_bytes", "byte/selected_window"),
    ("logic_energy_pj", "pJ/selected_window"),
    ("sram_energy_pj", "pJ/selected_window"),
    ("dram_energy_pj", "pJ/selected_window"),
    ("logic_area_um2", "um2"),
    ("macro_area_um2", "um2"),
    ("AEE", "pixel"),
    ("Fl_percent", "percent"),
]
METRIC_UNITS = dict(METRICS)
ROOT = Path(__file__).resolve().parents[3]
OFFICIAL_PROSPERITY_ROOT = Path(
    "/home/zhumd/work/literature_artifacts/Prosperity"
).resolve()


class RegistryError(RuntimeError):
    """Fail-closed registry validation error."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RegistryError(message)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unique_object(items: Iterable[Sequence[Any]]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in items:
        require(key not in result, "duplicate JSON key: " + key)
        result[key] = value
    return result


def strict_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=lambda token: (_ for _ in ()).throw(
                RegistryError("non-standard JSON token: " + token)
            ),
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RegistryError(f"cannot read strict JSON {path}: {exc}") from exc
    require(isinstance(data, dict), "JSON root must be an object: " + str(path))
    return data


def json_pointer(document: Any, pointer: str) -> Any:
    require(pointer == "" or pointer.startswith("/"),
            "invalid JSON pointer: " + pointer)
    current = document
    if not pointer:
        return current
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            require(token.isdigit(), "non-numeric array index in pointer: " + pointer)
            index = int(token)
            require(index < len(current), "array index out of range: " + pointer)
            current = current[index]
        else:
            require(isinstance(current, dict) and token in current,
                    "missing JSON pointer: " + pointer)
            current = current[token]
    return current


def numeric(value: Any, context: str) -> Union[int, float]:
    require(not isinstance(value, bool), context + " is boolean, not numeric")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise RegistryError(context + " is not numeric") from exc
    require(math.isfinite(number), context + " is non-finite")
    require(number >= 0, context + " is negative")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and number.is_integer():
        return int(number)
    return number


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_sources(
    config: Mapping[str, Any], repo_root: Path = ROOT
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Any]]:
    sources = config.get("sources")
    require(isinstance(sources, dict) and sources, "sources must be a nonempty object")
    resolved: Dict[str, Dict[str, Any]] = {}
    documents: Dict[str, Any] = {}
    for source_id, entry in sources.items():
        require(isinstance(source_id, str) and source_id,
                "source id must be a nonempty string")
        require(isinstance(entry, dict), "source entry must be an object: " + source_id)
        require(set(entry) == {"path", "sha256", "format", "role"},
                "source fields drift: " + source_id)
        raw_path = entry["path"]
        require(isinstance(raw_path, str) and raw_path, "invalid source path: " + source_id)
        path = Path(raw_path)
        path = (repo_root / path).resolve() if not path.is_absolute() else path.resolve()
        require(_inside(path, repo_root.resolve()) or
                _inside(path, OFFICIAL_PROSPERITY_ROOT),
                "source escapes allowed roots: " + source_id)
        require(path.is_file() and not path.is_symlink(),
                "source is not a regular non-symlink file: " + source_id)
        observed = sha256(path)
        require(entry["sha256"] == observed,
                f"source SHA mismatch: {source_id}: {observed}")
        require(entry["format"] in {"json", "text", "csv", "python"},
                "unsupported source format: " + source_id)
        require(isinstance(entry["role"], str) and entry["role"],
                "source role missing: " + source_id)
        resolved[source_id] = {
            "path": raw_path,
            "sha256": observed,
            "format": entry["format"],
            "role": entry["role"],
        }
        if entry["format"] == "json":
            documents[source_id] = strict_json(path)
    return resolved, documents


def _row_metadata(row: Mapping[str, Any]) -> Dict[str, str]:
    required = [
        "workload", "checkpoint", "sequence", "operator_scope",
        "resource_scope", "default_blocking_reason",
    ]
    for key in required:
        require(isinstance(row.get(key), str) and row[key].strip(),
                f"row {row.get('row_id')} missing {key}")
    return {key: row[key] for key in required[:-1]}


def _source_ref(
    resolved: Mapping[str, Mapping[str, Any]], source_id: str,
    pointers: Sequence[str],
) -> Dict[str, Any]:
    source = resolved[source_id]
    return {
        "source_id": source_id,
        "path": source["path"],
        "sha256": source["sha256"],
        "json_pointers": list(pointers),
    }


def _null_metric(row: Mapping[str, Any], metric_name: str) -> Dict[str, Any]:
    metadata = _row_metadata(row)
    return {
        "metric_id": f"{row['row_id']}.{metric_name}",
        "value": None,
        "unit": METRIC_UNITS[metric_name],
        "numerator": None,
        "denominator": None,
        **metadata,
        "evidence_class": "MISSING_OR_INCOMPATIBLE_EVIDENCE",
        "source": None,
        "admission": {
            "status": "BLOCKED",
            "system_table_eligible": False,
            "claim_boundary": "No paper-comparable value is admitted for this cell.",
        },
        "blocking_reason": row["default_blocking_reason"],
    }


def _validate_spec_common(spec: Mapping[str, Any], rows: Mapping[str, Any]) -> None:
    require(spec.get("row_id") in rows, "metric spec has unknown row")
    require(spec.get("metric") in METRIC_UNITS, "metric spec has unknown metric")
    require("speedup" not in str(spec.get("metric", "")).lower(),
            "speedup metrics are forbidden")


def build_registry(
    config: Mapping[str, Any], repo_root: Path = ROOT
) -> Dict[str, Any]:
    require(config.get("schema") == CONFIG_SCHEMA, "config schema drift")
    require(config.get("status") == STATUS, "config status must remain blocked")
    require(config.get("system_speedup_generated") is False,
            "system_speedup_generated must be false")
    row_entries = config.get("rows")
    require(isinstance(row_entries, list), "rows must be a list")
    require([row.get("row_id") for row in row_entries] == ROW_IDS,
            "fixed baseline row order/population drift")
    rows = {row["row_id"]: row for row in row_entries}
    require(len(rows) == len(ROW_IDS), "duplicate baseline row")
    for row in row_entries:
        _row_metadata(row)

    blockers = config.get("system_table_blockers")
    require(isinstance(blockers, list) and len(blockers) >= 5 and
            all(isinstance(item, str) and item.strip() for item in blockers),
            "at least five explicit system-table blockers are required")
    blocker_text = " ".join(blockers).lower()
    for token in ("decoder", "phi", "schedule", "macro", "multi-sequence"):
        require(token in blocker_text, "missing mandatory blocker token: " + token)

    resolved, documents = resolve_sources(config, repo_root)
    registry_rows: List[Dict[str, Any]] = []
    cell_index: Dict[Tuple[str, str], MutableMapping[str, Any]] = {}
    for row in row_entries:
        cells = [_null_metric(row, metric) for metric, _ in METRICS]
        registry_rows.append({
            "row_id": row["row_id"],
            "display_name": row.get("display_name", row["row_id"]),
            "metrics": cells,
        })
        for cell in cells:
            cell_index[(row["row_id"], cell["metric_id"].split(".", 1)[1])] = cell

    seen: Set[Tuple[str, str]] = set()
    for spec in config.get("populated_metrics", []):
        require(isinstance(spec, dict), "populated metric spec must be an object")
        _validate_spec_common(spec, rows)
        key = (spec["row_id"], spec["metric"])
        require(key not in seen, "duplicate populated metric: " + ".".join(key))
        seen.add(key)
        required = {
            "row_id", "metric", "source_id", "json_pointers", "aggregation",
            "divisor", "numerator_unit", "denominator_unit",
            "denominator_definition", "evidence_class", "claim_boundary",
        }
        require(set(spec) == required, "populated metric fields drift: " + ".".join(key))
        source_id = spec["source_id"]
        require(source_id in documents, "populated metric source must be JSON: " + source_id)
        pointers = spec["json_pointers"]
        require(isinstance(pointers, list) and pointers,
                "populated metric needs JSON pointers: " + ".".join(key))
        values = [numeric(json_pointer(documents[source_id], pointer),
                          source_id + pointer) for pointer in pointers]
        require(spec["aggregation"] in {"single", "sum"}, "unsupported aggregation")
        require((spec["aggregation"] == "single") == (len(values) == 1),
                "single aggregation/pointer cardinality mismatch")
        numerator = values[0] if len(values) == 1 else sum(values)
        divisor = numeric(spec["divisor"], "metric divisor")
        require(divisor > 0, "metric divisor must be positive")
        value = numerator / divisor
        if isinstance(numerator, int) and isinstance(divisor, int) and numerator % divisor == 0:
            value = numerator // divisor
        for name in ("numerator_unit", "denominator_unit",
                     "denominator_definition", "evidence_class", "claim_boundary"):
            require(isinstance(spec[name], str) and spec[name].strip(),
                    "missing populated metric metadata: " + name)
        cell = cell_index[key]
        cell.update({
            "value": value,
            "numerator": {
                "value": numerator,
                "unit": spec["numerator_unit"],
                "definition": "Aggregation of the pinned source JSON pointer(s).",
            },
            "denominator": {
                "value": divisor,
                "unit": spec["denominator_unit"],
                "definition": spec["denominator_definition"],
            },
            "evidence_class": spec["evidence_class"],
            "source": _source_ref(resolved, source_id, pointers),
            "admission": {
                "status": "INVENTORY_ONLY__NOT_SYSTEM_TABLE_ADMITTED",
                "system_table_eligible": False,
                "claim_boundary": spec["claim_boundary"],
            },
            "blocking_reason": None,
        })

    for spec in config.get("blocked_metric_evidence", []):
        require(isinstance(spec, dict), "blocked metric spec must be an object")
        _validate_spec_common(spec, rows)
        key = (spec["row_id"], spec["metric"])
        require(key not in seen, "duplicate metric spec: " + ".".join(key))
        seen.add(key)
        required = {
            "row_id", "metric", "source_id", "json_pointers",
            "evidence_class", "claim_boundary", "blocking_reason",
        }
        require(set(spec) == required, "blocked metric fields drift: " + ".".join(key))
        source_id = spec["source_id"]
        require(source_id in documents, "blocked metric source must be JSON: " + source_id)
        pointers = spec["json_pointers"]
        require(isinstance(pointers, list) and pointers, "blocked metric needs pointers")
        for pointer in pointers:
            numeric(json_pointer(documents[source_id], pointer), source_id + pointer)
        require(all(isinstance(spec[name], str) and spec[name].strip() for name in
                    ("evidence_class", "claim_boundary", "blocking_reason")),
                "blocked metric metadata missing")
        cell = cell_index[key]
        cell.update({
            "evidence_class": spec["evidence_class"],
            "source": _source_ref(resolved, source_id, pointers),
            "admission": {
                "status": "BLOCKED_INCOMPATIBLE_UNIT_OR_SCOPE",
                "system_table_eligible": False,
                "claim_boundary": spec["claim_boundary"],
            },
            "blocking_reason": spec["blocking_reason"],
        })

    result = {
        "schema": SCHEMA,
        "date": config.get("date"),
        "status": STATUS,
        "system_speedup_generated": False,
        "claim_boundary": (
            "Evidence inventory only. No row has complete common-resource, full-network, "
            "decoder-complete cycle/memory/energy/area evidence; no system comparison or "
            "system speedup is admitted."
        ),
        "metric_order": [name for name, _ in METRICS],
        "baseline_row_order": ROW_IDS,
        "source_inventory": resolved,
        "system_table_blockers": blockers,
        "rows": registry_rows,
    }
    validate_registry(result)
    return result


def validate_registry(registry: Mapping[str, Any]) -> None:
    require(registry.get("schema") == SCHEMA and registry.get("status") == STATUS,
            "registry identity/status drift")
    require(registry.get("system_speedup_generated") is False,
            "registry generated a system speedup")
    require(registry.get("metric_order") == [name for name, _ in METRICS],
            "metric order drift")
    require(registry.get("baseline_row_order") == ROW_IDS, "row order drift")
    rows = registry.get("rows")
    require(isinstance(rows, list) and [row.get("row_id") for row in rows] == ROW_IDS,
            "registry row population drift")
    metric_ids: Set[str] = set()
    for row in rows:
        metrics = row.get("metrics")
        require(isinstance(metrics, list) and
                [cell.get("metric_id") for cell in metrics] ==
                [f"{row['row_id']}.{name}" for name, _ in METRICS],
                "registry metric population/order drift: " + row["row_id"])
        for cell in metrics:
            metric_id = cell["metric_id"]
            require(metric_id not in metric_ids, "duplicate metric_id: " + metric_id)
            metric_ids.add(metric_id)
            metric_name = metric_id.split(".", 1)[1]
            require(cell.get("unit") == METRIC_UNITS[metric_name],
                    "metric unit drift: " + metric_id)
            for field in ("workload", "checkpoint", "sequence", "operator_scope",
                          "resource_scope", "evidence_class"):
                require(isinstance(cell.get(field), str) and cell[field].strip(),
                        f"{metric_id} missing {field}")
            admission = cell.get("admission")
            require(isinstance(admission, dict) and
                    admission.get("system_table_eligible") is False and
                    isinstance(admission.get("claim_boundary"), str) and
                    admission["claim_boundary"].strip(),
                    "metric admission boundary invalid: " + metric_id)
            if cell.get("value") is None:
                require(cell.get("numerator") is None and cell.get("denominator") is None,
                        "null metric carries numerator/denominator: " + metric_id)
                require(isinstance(cell.get("blocking_reason"), str) and
                        cell["blocking_reason"].strip(),
                        "null metric lacks blocking_reason: " + metric_id)
            else:
                numeric(cell["value"], metric_id)
                require(cell.get("blocking_reason") is None,
                        "numeric metric carries blocking_reason: " + metric_id)
                require(isinstance(cell.get("numerator"), dict) and
                        isinstance(cell.get("denominator"), dict) and
                        isinstance(cell.get("source"), dict),
                        "numeric metric lacks derivation/provenance: " + metric_id)
                require(cell["source"].get("sha256") and
                        cell["source"].get("json_pointers"),
                        "numeric metric lacks source SHA/pointers: " + metric_id)
    require(len(metric_ids) == len(ROW_IDS) * len(METRICS),
            "registry cell count drift")


def _report(registry: Mapping[str, Any]) -> str:
    lines = [
        "# M520 H67 paper metric registry v1",
        "",
        f"Status: `{STATUS}`",
        "",
        "This is a provenance-checked evidence inventory, not a system performance table. "
        "No system speedup is generated.",
        "",
        "| Row | Numeric cells | Blocked/null cells |",
        "|---|---:|---:|",
    ]
    for row in registry["rows"]:
        count = sum(cell["value"] is not None for cell in row["metrics"])
        lines.append(f"| {row['display_name']} | {count} | {len(METRICS) - count} |")
    lines.extend(["", "## Blocking gates", ""])
    lines.extend(f"- {item}" for item in registry["system_table_blockers"])
    lines.extend([
        "",
        "## Hard boundary",
        "",
        "The official Prosperity row is external support-tile iso-workload evidence only; "
        "its local ratio is intentionally absent. C1/C2/C3/A1 values retain component or "
        "included-scope labels. Null means blocked, never zero.",
        "",
    ])
    return "\n".join(lines)


def _seal(directory: Path, names: Sequence[str]) -> None:
    manifest = "".join(
        f"{sha256(directory / name)}  {name}\n" for name in sorted(names)
    )
    manifest_path = directory / "SHA256SUMS"
    manifest_path.write_text(manifest, encoding="utf-8")
    (directory / "SHA256SUMS.seal.sha256").write_text(
        f"{sha256(manifest_path)}  SHA256SUMS\n", encoding="utf-8"
    )


def write_output(registry: Mapping[str, Any], output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    require(output_dir.parent.is_dir() and not output_dir.exists(),
            "output must be a new child of an existing directory")
    staging = Path(tempfile.mkdtemp(prefix=".m520_registry_", dir=output_dir.parent))
    try:
        json_name = "m520_h67_paper_metric_registry_v1.json"
        (staging / json_name).write_text(
            json.dumps(registry, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (staging / "REPORT.md").write_text(_report(registry), encoding="utf-8")
        _seal(staging, [json_name, "REPORT.md"])
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_contract(contract: Mapping[str, Any], contract_path: Path,
                    config_path: Path, output_dir: Path) -> None:
    require(contract.get("schema") == CONTRACT_SCHEMA and
            contract.get("status") == STATUS,
            "contract schema/status drift")
    script_entry = contract.get("builder", {})
    config_entry = contract.get("config", {})
    require(script_entry.get("sha256") == sha256(Path(__file__).resolve()),
            "contract builder SHA drift")
    require(config_entry.get("sha256") == sha256(config_path),
            "contract config SHA drift")
    require((ROOT / script_entry.get("path", "")).resolve() == Path(__file__).resolve(),
            "contract builder path drift")
    require((ROOT / config_entry.get("path", "")).resolve() == config_path.resolve(),
            "contract config path drift")
    for name in ("test", "docs359"):
        entry = contract.get(name, {})
        pinned_path = (ROOT / entry.get("path", "")).resolve()
        require(pinned_path.is_file() and entry.get("sha256") == sha256(pinned_path),
                "contract pinned file drift: " + name)
    require((ROOT / contract["canonical_output_directory"]).resolve() ==
            output_dir.resolve(), "non-canonical M520 output directory")
    require(contract_path.is_file(), "contract is not a file")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    contract_path = args.contract.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    contract = strict_json(contract_path)
    verify_contract(contract, contract_path, config_path, output_dir)
    config = strict_json(config_path)
    registry = build_registry(config)
    registry["generator"] = {
        "builder": {
            "path": str(Path(__file__).resolve().relative_to(ROOT)),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "config": {
            "path": str(config_path.relative_to(ROOT)),
            "sha256": sha256(config_path),
        },
        "contract": {
            "path": str(contract_path.relative_to(ROOT)),
            "sha256": sha256(contract_path),
        },
    }
    write_output(registry, output_dir)
    print(json.dumps({
        "status": registry["status"],
        "rows": len(registry["rows"]),
        "metrics": len(registry["rows"]) * len(METRICS),
        "system_speedup_generated": False,
        "output_dir": str(output_dir),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
