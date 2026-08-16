#!/usr/bin/env python3
"""Stable logical transaction identities for Local5 EREP v4."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


SCHEMA = "local5_erep_identity_service_v4"
DEFAULT_SEED = 20260810
DELAY_VALUES = 4
SERVICE_LATENCY_OFFSET = 1

# This is the complete frozen identity schema. Fields not listed here are invalid.
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

IDENTITY_FIELDS_BY_KIND = REQUIRED_FIELDS_BY_KIND

# These fields previously allowed identity to depend on issue order or candidate.
FORBIDDEN_IDENTITY_FIELDS = frozenset(
    {
        "occurrence",
        "candidate_private",
        "transaction_index",
        "global_transaction_index",
        "global_index",
    }
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
                raise ValueError(f"{path}.{key} is a forbidden identity field")
            _reject_forbidden_fields(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_forbidden_fields(child, f"{path}[{index}]")


def _normalize_identity(
    kind: str, identity: Mapping[str, Any]
) -> tuple[dict[str, Any], bytes]:
    if kind not in REQUIRED_FIELDS_BY_KIND:
        allowed = ", ".join(sorted(REQUIRED_FIELDS_BY_KIND))
        raise ValueError(f"unknown transaction kind {kind!r}; expected one of: {allowed}")
    if not isinstance(identity, Mapping):
        raise ValueError("identity must be a JSON object")

    copied = dict(identity)
    _reject_forbidden_fields(copied)
    encoded = canonical_json_bytes(copied)
    normalized = json.loads(encoded)
    if not isinstance(normalized, dict):
        raise ValueError("identity must be a JSON object")

    expected_fields = frozenset(REQUIRED_FIELDS_BY_KIND[kind])
    actual_fields = frozenset(normalized)
    missing = sorted(expected_fields - actual_fields)
    unexpected = sorted(actual_fields - expected_fields)
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(unexpected)}")
        raise ValueError(
            f"{kind} identity fields must exactly match the frozen schema; "
            + "; ".join(details)
        )

    sample = normalized["sample"]
    if not isinstance(sample, str) or not sample:
        raise ValueError("identity.sample must be a non-empty string")
    for field in REQUIRED_FIELDS_BY_KIND[kind]:
        if field == "sample":
            continue
        value = normalized[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"identity.{field} must be a non-negative integer")
    return normalized, encoded


def validate_identity(kind: str, identity: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a JSON-normalized copy of an exact frozen identity."""

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
    """Return the length-prefixed UTF-8 bytes hashed for a transaction."""

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
        if (
            isinstance(self.delay, bool)
            or not isinstance(self.delay, int)
            or self.delay != expected
        ):
            raise ValueError("delay does not match schema/seed/kind/identity")
        if normalized != decoded:
            raise AssertionError("canonical identity normalization changed JSON value")

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(self.identity_json)

    @property
    def identity_key(self) -> str:
        return canonical_json({"kind": self.kind, "identity": self.identity})

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
    """Create a stable transaction after validating its exact frozen identity."""

    _validate_service_config(schema, seed)
    _, identity_json = _normalize_identity(kind, identity)
    material = _hash_material_from_identity_json(
        kind, identity_json, seed=seed, schema=schema
    )
    delay = int.from_bytes(hashlib.sha256(material).digest()[:8], "big") % DELAY_VALUES
    return Transaction(schema, seed, kind, identity_json, delay)


def _identity_key_bytes(transaction: Transaction) -> bytes:
    return canonical_json_bytes(
        {"kind": transaction.kind, "identity": transaction.identity}
    )


def _validate_ledger(
    transactions: Iterable[Transaction],
    *,
    seed: int,
    schema: str,
) -> tuple[Transaction, ...]:
    _validate_service_config(schema, seed)
    rows = tuple(transactions)
    for row in rows:
        if not isinstance(row, Transaction):
            raise ValueError("ledger entries must be Transaction instances")
        if row.schema != schema or row.seed != seed:
            raise ValueError("ledger mixes schema or seed values")
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
            b"local5-erep-ledger-v4",
            domain,
            schema.encode("utf-8"),
            canonical_json_bytes(seed),
            canonical_json_bytes(len(rows)),
            *rows,
        )
    )
    return hashlib.sha256(material).hexdigest()


@dataclass(frozen=True)
class IdentityMultiplicity:
    """Auditable multiplicity for one exact logical identity."""

    kind: str
    identity_json: bytes
    transaction_digest: str
    delay: int
    multiplicity: int

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(self.identity_json)

    @property
    def identity_key(self) -> str:
        return canonical_json({"kind": self.kind, "identity": self.identity})

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "identity": self.identity,
            "transaction_digest": self.transaction_digest,
            "delay": self.delay,
            "multiplicity": self.multiplicity,
        }


def _identity_multiplicities(
    rows: Iterable[Transaction],
) -> tuple[IdentityMultiplicity, ...]:
    groups: dict[bytes, list[Transaction]] = {}
    for row in rows:
        groups.setdefault(_identity_key_bytes(row), []).append(row)

    audits = []
    for key in sorted(groups):
        group = groups[key]
        representative = group[0]
        delays = {row.delay for row in group}
        digests = {row.digest for row in group}
        if len(delays) != 1 or len(digests) != 1:
            raise ValueError("identical logical identities have inconsistent service results")
        audits.append(
            IdentityMultiplicity(
                kind=representative.kind,
                identity_json=representative.identity_json,
                transaction_digest=representative.digest,
                delay=representative.delay,
                multiplicity=len(group),
            )
        )
    return tuple(audits)


@dataclass(frozen=True)
class LedgerDigests:
    transaction_count: int
    ordered_digest: str
    multiset_digest: str
    identity_multiplicities: tuple[IdentityMultiplicity, ...]

    @property
    def ordered_ledger_digest(self) -> str:
        return self.ordered_digest

    @property
    def unordered_multiset_digest(self) -> str:
        return self.multiset_digest

    @property
    def identity_count(self) -> int:
        return len(self.identity_multiplicities)

    @property
    def multiplicity_by_identity(self) -> dict[str, int]:
        return {
            audit.identity_key: audit.multiplicity
            for audit in self.identity_multiplicities
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "transaction_count": self.transaction_count,
            "ordered_ledger_digest": self.ordered_digest,
            "unordered_multiset_digest": self.multiset_digest,
            "identity_multiplicities": [
                audit.as_dict() for audit in self.identity_multiplicities
            ],
        }


def ledger_digests(
    transactions: Iterable[Transaction],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> LedgerDigests:
    """Audit issue order, multiset contents, and every identity multiplicity."""

    rows = _validate_ledger(transactions, seed=seed, schema=schema)
    records = tuple(row.canonical_record_bytes() for row in rows)
    return LedgerDigests(
        transaction_count=len(rows),
        ordered_digest=_ledger_hash(b"ordered", records, seed=seed, schema=schema),
        multiset_digest=_ledger_hash(
            b"multiset", sorted(records), seed=seed, schema=schema
        ),
        identity_multiplicities=_identity_multiplicities(rows),
    )


def comparable_transaction_map(
    transactions: Iterable[Transaction],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> dict[str, int]:
    """Return one deterministic delay per unique logical identity."""

    rows = _validate_ledger(transactions, seed=seed, schema=schema)
    return {
        audit.identity_key: audit.delay for audit in _identity_multiplicities(rows)
    }


def transaction_multiplicity_map(
    transactions: Iterable[Transaction],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> dict[str, int]:
    """Return the exact count of each logical identity in a ledger."""

    rows = _validate_ledger(transactions, seed=seed, schema=schema)
    return {
        audit.identity_key: audit.multiplicity
        for audit in _identity_multiplicities(rows)
    }


@dataclass(frozen=True)
class MultiplicityDifference:
    """Left and right counts for an identity whose multiplicity differs."""

    kind: str
    identity_json: bytes
    left_multiplicity: int
    right_multiplicity: int

    @property
    def identity(self) -> dict[str, Any]:
        return json.loads(self.identity_json)

    @property
    def identity_key(self) -> str:
        return canonical_json({"kind": self.kind, "identity": self.identity})

    @property
    def delta(self) -> int:
        return self.left_multiplicity - self.right_multiplicity

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "identity": self.identity,
            "left_multiplicity": self.left_multiplicity,
            "right_multiplicity": self.right_multiplicity,
            "left_minus_right": self.delta,
        }


IdentityMultiplicityDifference = MultiplicityDifference


@dataclass(frozen=True)
class CandidateComparison:
    common_identity_count: int
    common_transaction_count: int
    left_only_transaction_count: int
    right_only_transaction_count: int
    delay_mismatches: tuple[str, ...]
    multiplicity_differences: tuple[MultiplicityDifference, ...]

    @property
    def common_delays_match(self) -> bool:
        return not self.delay_mismatches

    @property
    def multiplicities_match(self) -> bool:
        return not self.multiplicity_differences

    @property
    def multiplicity_mismatches(self) -> tuple[MultiplicityDifference, ...]:
        return self.multiplicity_differences

    @property
    def common_multiplicity_differences(self) -> tuple[MultiplicityDifference, ...]:
        return tuple(
            difference
            for difference in self.multiplicity_differences
            if difference.left_multiplicity and difference.right_multiplicity
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "common_identity_count": self.common_identity_count,
            "common_transaction_count": self.common_transaction_count,
            "left_only_transaction_count": self.left_only_transaction_count,
            "right_only_transaction_count": self.right_only_transaction_count,
            "delay_mismatches": list(self.delay_mismatches),
            "multiplicity_differences": [
                difference.as_dict() for difference in self.multiplicity_differences
            ],
        }


def compare_candidate_ledgers(
    left: Iterable[Transaction],
    right: Iterable[Transaction],
    *,
    seed: int = DEFAULT_SEED,
    schema: str = SCHEMA,
) -> CandidateComparison:
    """Compare candidate ledgers as multisets of exact logical identities."""

    left_rows = _validate_ledger(left, seed=seed, schema=schema)
    right_rows = _validate_ledger(right, seed=seed, schema=schema)
    left_groups = {
        audit.identity_key.encode("utf-8"): audit
        for audit in _identity_multiplicities(left_rows)
    }
    right_groups = {
        audit.identity_key.encode("utf-8"): audit
        for audit in _identity_multiplicities(right_rows)
    }

    common_keys = set(left_groups).intersection(right_groups)
    delay_mismatches = tuple(
        key.decode("utf-8")
        for key in sorted(common_keys)
        if left_groups[key].delay != right_groups[key].delay
    )

    differences = []
    common_transaction_count = 0
    left_only_transaction_count = 0
    right_only_transaction_count = 0
    for key in sorted(set(left_groups).union(right_groups)):
        left_audit = left_groups.get(key)
        right_audit = right_groups.get(key)
        left_count = left_audit.multiplicity if left_audit is not None else 0
        right_count = right_audit.multiplicity if right_audit is not None else 0
        common_transaction_count += min(left_count, right_count)
        left_only_transaction_count += max(left_count - right_count, 0)
        right_only_transaction_count += max(right_count - left_count, 0)
        if left_count != right_count:
            representative = left_audit if left_audit is not None else right_audit
            assert representative is not None
            differences.append(
                MultiplicityDifference(
                    kind=representative.kind,
                    identity_json=representative.identity_json,
                    left_multiplicity=left_count,
                    right_multiplicity=right_count,
                )
            )

    return CandidateComparison(
        common_identity_count=len(common_keys),
        common_transaction_count=common_transaction_count,
        left_only_transaction_count=left_only_transaction_count,
        right_only_transaction_count=right_only_transaction_count,
        delay_mismatches=delay_mismatches,
        multiplicity_differences=tuple(differences),
    )


@dataclass(frozen=True)
class IdentityService:
    seed: int = DEFAULT_SEED
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        _validate_service_config(self.schema, self.seed)

    def delay(self, kind: str, identity: Mapping[str, Any]) -> int:
        return transaction_delay(kind, identity, seed=self.seed, schema=self.schema)

    def transaction(self, kind: str, identity: Mapping[str, Any]) -> Transaction:
        return make_transaction(kind, identity, seed=self.seed, schema=self.schema)

    def ledger_digests(self, transactions: Iterable[Transaction]) -> LedgerDigests:
        return ledger_digests(transactions, seed=self.seed, schema=self.schema)

    def comparable_map(self, transactions: Iterable[Transaction]) -> dict[str, int]:
        return comparable_transaction_map(
            transactions, seed=self.seed, schema=self.schema
        )

    def multiplicity_map(self, transactions: Iterable[Transaction]) -> dict[str, int]:
        return transaction_multiplicity_map(
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
