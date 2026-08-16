#!/usr/bin/env python3
"""Stable logical transaction identities for Local5 EREP v3."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCHEMA = "local5_erep_identity_service_v3"
DEFAULT_SEED = 20260810
DELAY_VALUES = 4
SERVICE_LATENCY_OFFSET = 1

REQUIRED_FIELDS_BY_KIND: dict[str, tuple[str, ...]] = {
    "relation": (
        "sample",
        "stage",
        "block",
        "window",
        "input_head",
        "source_id",
    ),
    "epoch_read": (
        "sample",
        "stage",
        "block",
        "window",
        "stripe",
        "input_head",
        "output_tile",
        "source_id",
    ),
    "weight": (
        "sample",
        "stage",
        "block",
        "window",
        "input_head",
        "output_tile",
        "lane",
        "out",
    ),
    "final": (
        "sample",
        "stage",
        "block",
        "window",
        "output_tile",
        "source_id",
        "out",
    ),
}

# These names couple service behavior to candidate-dependent issue order.
FORBIDDEN_IDENTITY_FIELDS = frozenset(
    {"transaction_index", "global_transaction_index", "global_index"}
)


def _validate_json_object_keys(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON object key at {path} must be a string")
            _validate_json_object_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_json_object_keys(child, f"{path}[{index}]")


def canonical_json(value: Any) -> str:
    """Return sorted-key, compact, non-ASCII-preserving canonical JSON."""

    _validate_json_object_keys(value)
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"value is not canonical JSON data: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Encode :func:`canonical_json` as strict UTF-8 bytes."""

    try:
        return canonical_json(value).encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("canonical JSON contains an invalid Unicode surrogate") from exc


def _validate_service_config(schema: str, seed: int) -> None:
    if not isinstance(schema, str) or not schema:
        raise ValueError("schema must be a non-empty string")
    try:
        schema.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError("schema contains an invalid Unicode surrogate") from exc
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _reject_forbidden_fields(value: Any, path: str = "identity") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if key in FORBIDDEN_IDENTITY_FIELDS:
                raise ValueError(f"{path}.{key} is a forbidden global index field")
            _reject_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{path}[{index}]")


def _normalize_identity(kind: str, identity: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    if kind not in REQUIRED_FIELDS_BY_KIND:
        allowed = ", ".join(sorted(REQUIRED_FIELDS_BY_KIND))
        raise ValueError(f"unknown transaction kind {kind!r}; expected one of: {allowed}")
    if not isinstance(identity, Mapping):
        raise ValueError("identity must be a JSON object")

    encoded = canonical_json_bytes(dict(identity))
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise ValueError("identity must be a JSON object")

    missing = [
        field for field in REQUIRED_FIELDS_BY_KIND[kind] if field not in normalized
    ]
    if missing:
        raise ValueError(f"{kind} identity is missing required fields: {', '.join(missing)}")

    _reject_forbidden_fields(normalized)
    if "occurrence" in normalized:
        occurrence = normalized["occurrence"]
        if (
            isinstance(occurrence, bool)
            or not isinstance(occurrence, int)
            or occurrence < 0
        ):
            raise ValueError("identity.occurrence must be a non-negative integer")
    return normalized, encoded


def validate_identity(kind: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a JSON-normalized copy of a complete identity."""

    normalized, _ = _normalize_identity(kind, identity)
    return normalized


def _length_prefix(parts: Iterable[bytes]) -> bytes:
    framed = bytearray()
    for part in parts:
        if not isinstance(part, bytes):
            raise TypeError("hash frame components must be bytes")
        framed.extend(len(part).to_bytes(8, "big"))
        framed.extend(part)
    return bytes(framed)


def _hash_material_from_identity_json(
    kind: str,
    identity_json: bytes,
    *,
    seed: int,
    schema: str,
) -> bytes:
    # Tags and uint64 lengths make every component boundary independently auditable.
    return _length_prefix(
        (
            b"schema",
            schema.encode("utf-8"),
            b"seed",
            canonical_json_bytes(seed),
            b"kind",
            kind.encode("utf-8"),
            b"identity",
            identity_json,
        )
    )


def transaction_hash_material(
    kind: str,
    identity: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> bytes:
    """Return the unambiguous bytes hashed for a transaction."""

    _validate_service_config(schema, seed)
    _, identity_json = _normalize_identity(kind, identity)
    return _hash_material_from_identity_json(
        kind, identity_json, seed=seed, schema=schema
    )


def transaction_digest(
    kind: str,
    identity: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> str:
    """Return the SHA-256 logical transaction digest."""

    return hashlib.sha256(
        transaction_hash_material(kind, identity, seed=seed, schema=schema)
    ).hexdigest()


def transaction_delay(
    kind: str,
    identity: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> int:
    """Return the deterministic service delay in the inclusive range 0..3."""

    digest = hashlib.sha256(
        transaction_hash_material(kind, identity, seed=seed, schema=schema)
    ).digest()
    return int.from_bytes(digest[:8], "big") % DELAY_VALUES


@dataclass(frozen=True)
class Transaction:
    """An immutable, validated logical transaction service result."""

    schema: str
    seed: int
    kind: str
    identity_json: bytes
    delay: int

    def __post_init__(self) -> None:
        _validate_service_config(self.schema, self.seed)
        if not isinstance(self.identity_json, bytes):
            raise ValueError("identity_json must be bytes")
        try:
            decoded = json.loads(self.identity_json)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("identity_json must contain valid UTF-8 JSON") from exc
        normalized, encoded = _normalize_identity(self.kind, decoded)
        if encoded != self.identity_json:
            raise ValueError("identity_json is not canonical")
        expected = int.from_bytes(
            hashlib.sha256(
                _hash_material_from_identity_json(
                    self.kind,
                    encoded,
                    seed=self.seed,
                    schema=self.schema,
                )
            ).digest()[:8],
            "big",
        ) % DELAY_VALUES
        if self.delay != expected:
            raise ValueError("delay does not match schema/seed/kind/identity")
        if normalized != decoded:
            raise AssertionError("canonical identity normalization changed JSON value")

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(self.identity_json)

    @property
    def digest(self) -> str:
        material = _hash_material_from_identity_json(
            self.kind,
            self.identity_json,
            seed=self.seed,
            schema=self.schema,
        )
        return hashlib.sha256(material).hexdigest()

    @property
    def occurrence(self) -> int | None:
        return self.identity.get("occurrence")

    @property
    def response_latency_cycles(self) -> int:
        """Return the registered service latency, in the closed range 1..4."""

        return self.delay + SERVICE_LATENCY_OFFSET

    def response_cycle(self, accept_cycle: int) -> int:
        """Return the response edge for a request accepted at ``accept_cycle``."""

        if (
            isinstance(accept_cycle, bool)
            or not isinstance(accept_cycle, int)
            or accept_cycle < 0
        ):
            raise ValueError("accept_cycle must be a non-negative integer")
        return accept_cycle + self.response_latency_cycles

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "seed": self.seed,
            "kind": self.kind,
            "identity": self.identity,
            "delay": self.delay,
            "response_latency_cycles": self.response_latency_cycles,
            "transaction_digest": self.digest,
        }

    def canonical_record_bytes(self) -> bytes:
        return canonical_json_bytes(
            {
                "schema": self.schema,
                "seed": self.seed,
                "kind": self.kind,
                "identity": self.identity,
                "delay": self.delay,
            }
        )


def make_transaction(
    kind: str,
    identity: Mapping[str, Any],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> Transaction:
    """Create a stable transaction after validating the complete identity."""

    _validate_service_config(schema, seed)
    _, identity_json = _normalize_identity(kind, identity)
    material = _hash_material_from_identity_json(
        kind, identity_json, seed=seed, schema=schema
    )
    delay = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % DELAY_VALUES
    return Transaction(schema, seed, kind, identity_json, delay)


def _base_identity_json(transaction: Transaction) -> bytes:
    identity = transaction.identity
    identity.pop("occurrence", None)
    return canonical_json_bytes({"kind": transaction.kind, "identity": identity})


def _validate_ledger(
    transactions: Iterable[Transaction],
    *,
    seed: int,
    schema: str,
) -> tuple[Transaction, ...]:
    _validate_service_config(schema, seed)
    rows = tuple(transactions)
    groups: dict[bytes, list[Transaction]] = {}
    for row in rows:
        if not isinstance(row, Transaction):
            raise ValueError("ledger entries must be Transaction instances")
        if row.schema != schema or row.seed != seed:
            raise ValueError("ledger mixes schema or seed values")
        groups.setdefault(_base_identity_json(row), []).append(row)

    for group in groups.values():
        if len(group) <= 1:
            continue
        occurrences = [row.occurrence for row in group]
        if any(value is None for value in occurrences):
            raise ValueError(
                "repeated logical identity requires explicit occurrence on every copy"
            )
        if len(set(occurrences)) != len(occurrences):
            raise ValueError(
                "repeated logical identity occurrence values must be unique"
            )
    return rows


def _ledger_hash(
    domain: bytes,
    records: Iterable[bytes],
    *,
    seed: int,
    schema: str,
) -> str:
    rows = tuple(records)
    material = _length_prefix(
        (
            b"local5-erep-ledger-v3",
            domain,
            schema.encode("utf-8"),
            canonical_json_bytes(seed),
            canonical_json_bytes(len(rows)),
            *rows,
        )
    )
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class LedgerDigests:
    transaction_count: int
    ordered_digest: str
    multiset_digest: str

    @property
    def ordered_ledger_digest(self) -> str:
        return self.ordered_digest

    @property
    def unordered_multiset_digest(self) -> str:
        return self.multiset_digest

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_count": self.transaction_count,
            "ordered_ledger_digest": self.ordered_digest,
            "unordered_multiset_digest": self.multiset_digest,
        }


def ledger_digests(
    transactions: Iterable[Transaction],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> LedgerDigests:
    """Digest a ledger both in issue order and as an order-independent multiset."""

    rows = _validate_ledger(transactions, seed=seed, schema=schema)
    records = tuple(row.canonical_record_bytes() for row in rows)
    return LedgerDigests(
        transaction_count=len(rows),
        ordered_digest=_ledger_hash(
            b"ordered", records, seed=seed, schema=schema
        ),
        multiset_digest=_ledger_hash(
            b"multiset", sorted(records), seed=seed, schema=schema
        ),
    )


def comparable_transaction_map(
    transactions: Iterable[Transaction],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> dict[str, int]:
    """Return a candidate-order-independent logical-identity-to-delay map."""

    rows = _validate_ledger(transactions, seed=seed, schema=schema)
    result: dict[str, int] = {}
    for row in rows:
        key = canonical_json({"kind": row.kind, "identity": row.identity})
        if key in result:
            raise ValueError("ledger contains a duplicate complete transaction identity")
        result[key] = row.delay
    return result


@dataclass(frozen=True)
class CandidateComparison:
    common_transaction_count: int
    left_only_transaction_count: int
    right_only_transaction_count: int
    delay_mismatches: tuple[str, ...]

    @property
    def common_delays_match(self) -> bool:
        return not self.delay_mismatches


def compare_candidate_ledgers(
    left: Iterable[Transaction],
    right: Iterable[Transaction],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> CandidateComparison:
    """Compare all common logical transactions independent of candidate order."""

    left_rows = _validate_ledger(left, seed=seed, schema=schema)
    right_rows = _validate_ledger(right, seed=seed, schema=schema)

    left_occurrences: dict[bytes, tuple[int | None, ...]] = {}
    right_occurrences: dict[bytes, tuple[int | None, ...]] = {}
    for rows, groups in (
        (left_rows, left_occurrences),
        (right_rows, right_occurrences),
    ):
        pending: dict[bytes, list[int | None]] = {}
        for row in rows:
            pending.setdefault(_base_identity_json(row), []).append(row.occurrence)
        groups.update({key: tuple(values) for key, values in pending.items()})

    for base_key in set(left_occurrences).intersection(right_occurrences):
        combined = left_occurrences[base_key] + right_occurrences[base_key]
        if any(value is None for value in combined) and any(
            value is not None for value in combined
        ):
            raise ValueError(
                "common logical identity uses inconsistent occurrence encoding"
            )

    left_map = comparable_transaction_map(left_rows, seed=seed, schema=schema)
    right_map = comparable_transaction_map(right_rows, seed=seed, schema=schema)
    common = set(left_map).intersection(right_map)
    mismatches = tuple(
        sorted(key for key in common if left_map[key] != right_map[key])
    )
    return CandidateComparison(
        common_transaction_count=len(common),
        left_only_transaction_count=len(set(left_map).difference(right_map)),
        right_only_transaction_count=len(set(right_map).difference(left_map)),
        delay_mismatches=mismatches,
    )


@dataclass(frozen=True)
class IdentityService:
    seed: int = DEFAULT_SEED
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        _validate_service_config(self.schema, self.seed)

    def delay(self, kind: str, identity: Mapping[str, Any]) -> int:
        return transaction_delay(
            kind, identity, seed=self.seed, schema=self.schema
        )

    def transaction(self, kind: str, identity: Mapping[str, Any]) -> Transaction:
        return make_transaction(
            kind, identity, seed=self.seed, schema=self.schema
        )

    def ledger_digests(self, transactions: Iterable[Transaction]) -> LedgerDigests:
        return ledger_digests(
            transactions, seed=self.seed, schema=self.schema
        )

    def comparable_map(self, transactions: Iterable[Transaction]) -> dict[str, int]:
        return comparable_transaction_map(
            transactions, seed=self.seed, schema=self.schema
        )

    def compare_candidates(
        self,
        left: Iterable[Transaction],
        right: Iterable[Transaction],
    ) -> CandidateComparison:
        return compare_candidate_ledgers(
            left, right, seed=self.seed, schema=self.schema
        )


# Explicit aliases keep call sites concise without introducing alternate semantics.
build_transaction = make_transaction
delay_for_transaction = transaction_delay
