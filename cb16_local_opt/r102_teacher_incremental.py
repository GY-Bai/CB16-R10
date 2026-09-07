from __future__ import annotations

"""P0 incremental Teacher compilation for R10.2/R2.

The frozen R6 Teacher mathematics are preserved.  This module removes two pure
recomputation classes:

* compiled Teacher evidence is content-addressed and reused across replay/runs when
  source truth, protocols, implementation and numerical runtime identity match;
* normalization and target/context geometry are computed once per target parent and
  reused across the complete action/risk grid.

Scheduling parameters are deliberately excluded from scientific identity.  Cache
publication and reuse are fail-closed.
"""

import gzip
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .probabilistic_teacher_r5 import weighted_quantile
from .probabilistic_teacher_r6 import (
    DependenceAwarePredictiveLawR6,
    DependenceAwareProbabilisticTeacherR6,
    DependenceAwareTeacherConfigR6,
    DependenceAwareTeacherEvidenceR6,
    EvidenceAdmissionReceiptR6,
    TeacherIndexR6,
    _softmax,
    canonical_hash,
)
from .r102_common import canonical_json_bytes, sha256_file, sha256_obj
from .r102_evidence_cache import ParentContextR102


COMPILED_TEACHER_CACHE_SCHEMA = "CB16_R102_COMPILED_TEACHER_AUTHORITY_V2"
COMPILED_TEACHER_IDENTITY_SCHEMA = "CB16_R102_COMPILED_TEACHER_IDENTITY_V2"
COMPILED_TEACHER_PAYLOAD_SCHEMA = "CB16_R102_COMPILED_TEACHER_PAYLOAD_V2"
COMPILED_TEACHER_RECEIPT_SCHEMA = "CB16_R102_COMPILED_TEACHER_RECEIPT_V2"
COMPILER_SEMANTICS = "R6_EXACT_DISTRIBUTIONAL_TEACHER__P0_GEOMETRY_REUSE_V2"
NONFINITE_TAG = "__cb16_nonfinite_float_v1__"


class ExactIncrementalTeacherR6(DependenceAwareProbabilisticTeacherR6):
    """Exact R6 Teacher with target-local geometry reuse."""

    def _prepare_geometry(
        self,
        *,
        target_features: tuple[float, ...] | np.ndarray,
        train_deps: Sequence[str],
        index: TeacherIndexR6,
        feature_override: Mapping[str, tuple[float, ...]] | None = None,
    ) -> dict[str, float]:
        mean, std = self._normalization(
            train_deps=train_deps,
            index=index,
            feature_override=feature_override,
        )
        target = (np.asarray(target_features, dtype=np.float64) - mean) / std
        distances: dict[str, float] = {}
        for dep in train_deps:
            for parent_id in index.parents_by_dependence_group[dep]:
                feat = (
                    feature_override[parent_id]
                    if feature_override is not None
                    else index.rows_by_parent[parent_id][0].context_features
                )
                z = (np.asarray(feat, dtype=np.float64) - mean) / std
                distances[parent_id] = float(np.sqrt(np.mean((z - target) ** 2)))
        return distances

    def _predictive_law_from_geometry(
        self,
        *,
        train_deps: Sequence[str],
        index: TeacherIndexR6,
        direction: int,
        risk: float,
        distance_by_parent: Mapping[str, float],
    ) -> DependenceAwarePredictiveLawR6 | None:
        candidates = []
        for dep in train_deps:
            best = None
            for parent_id in index.parents_by_dependence_group[dep]:
                match = [
                    s for s in index.rows_by_parent[parent_id]
                    if s.direction == direction
                    and abs(s.requested_risk - risk) <= 1e-12
                ]
                if not match:
                    continue
                if len(match) != 1:
                    raise RuntimeError("DUPLICATE_ACTION_BRANCH_WITHIN_PARENT")
                row = (
                    float(distance_by_parent[parent_id]),
                    parent_id,
                    float(match[0].realized_utility),
                )
                if best is None or row[:2] < best[:2]:
                    best = row
            if best is not None:
                candidates.append((best[0], dep, best[1], best[2]))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        selected = candidates[: min(self.config.k_dependence_groups, len(candidates))]
        distances = np.asarray([x[0] for x in selected], dtype=np.float64)
        utility = np.asarray([x[3] for x in selected], dtype=np.float64)
        deps = [x[1] for x in selected]
        weights = np.exp(
            -0.5 * (distances / self.config.distance_temperature) ** 2
        ) + 1e-12
        weights /= weights.sum()
        quantiles = weighted_quantile(
            utility,
            weights,
            np.asarray(self.config.quantile_levels, dtype=np.float64),
        )
        mean_utility = float(np.sum(weights * utility))
        variance = float(np.sum(weights * (utility - mean_utility) ** 2))
        effective_n = float(1.0 / np.sum(weights ** 2))
        return DependenceAwarePredictiveLawR6(
            direction=direction,
            requested_risk=float(risk),
            mean_utility=mean_utility,
            std_utility=math.sqrt(max(0.0, variance)),
            quantile_levels=self.config.quantile_levels,
            quantiles=tuple(float(x) for x in quantiles),
            effective_dependence_n=effective_n,
            unique_dependence_groups=len(deps),
            nearest_distance=float(distances[0]),
            max_distance_used=float(distances[-1]),
            support_dependence_group_hash=canonical_hash(deps),
        )

    def compile_one(
        self,
        *,
        target_parent: str,
        index: TeacherIndexR6,
        eligible_train_dependence_groups: set[str] | None = None,
        feature_override: Mapping[str, tuple[float, ...]] | None = None,
        target_feature_override: tuple[float, ...] | None = None,
    ) -> DependenceAwareTeacherEvidenceR6:
        rows = index.rows_by_parent[target_parent]
        train_deps = self.train_dependence_groups_for_target(
            target_parent=target_parent,
            index=index,
            eligible_train_dependence_groups=eligible_train_dependence_groups,
        )
        target_features = (
            target_feature_override
            if target_feature_override is not None
            else rows[0].context_features
        )
        grid = sorted(
            {(s.direction, float(s.requested_risk)) for s in rows},
            key=lambda x: (x[0], x[1]),
        )
        laws = []
        if train_deps:
            distances = self._prepare_geometry(
                target_features=target_features,
                train_deps=train_deps,
                index=index,
                feature_override=feature_override,
            )
            for direction, risk in grid:
                law = self._predictive_law_from_geometry(
                    train_deps=train_deps,
                    index=index,
                    direction=direction,
                    risk=risk,
                    distance_by_parent=distances,
                )
                if law is not None:
                    laws.append(law)

        best_by_direction = {}
        for direction in (-1, 0, 1):
            candidates = [law for law in laws if law.direction == direction]
            if candidates:
                best_by_direction[direction] = max(
                    candidates,
                    key=lambda law: (law.mean_utility, -law.requested_risk),
                )

        reasons = []
        if len(train_deps) < self.config.min_train_dependence_groups:
            reasons.append("INSUFFICIENT_TRAIN_DEPENDENCE_GROUPS")
        if set(best_by_direction) != {-1, 0, 1}:
            reasons.append("ACTION_GRID_SUPPORT_INCOMPLETE")
        minimum_effective_n = min(
            (law.effective_dependence_n for law in best_by_direction.values()),
            default=0.0,
        )
        maximum_nearest = max(
            (law.nearest_distance for law in best_by_direction.values()),
            default=float("inf"),
        )
        if minimum_effective_n < self.config.min_effective_dependence_n:
            reasons.append("INSUFFICIENT_EFFECTIVE_DEPENDENCE_SUPPORT")
        if maximum_nearest > self.config.max_nearest_distance:
            reasons.append("TARGET_OUTSIDE_SUPPORTED_CONTEXT")

        if set(best_by_direction) == {-1, 0, 1}:
            means = np.asarray([
                best_by_direction[-1].mean_utility,
                best_by_direction[0].mean_utility,
                best_by_direction[1].mean_utility,
            ])
            probs = _softmax(means, self.config.direction_softmax_temperature)
            risks = np.asarray([
                best_by_direction[-1].requested_risk,
                0.0,
                best_by_direction[1].requested_risk,
            ])
            risk_target = float(np.sum(probs * risks))
        else:
            probs = np.asarray([1 / 3, 1 / 3, 1 / 3], dtype=np.float64)
            risk_target = 0.0

        admission = EvidenceAdmissionReceiptR6(
            status="ADMITTED" if not reasons else "EVIDENCE_NOT_READY",
            lane=self.config.lane,
            unique_train_dependence_groups=len(train_deps),
            minimum_action_effective_dependence_n=float(minimum_effective_n),
            maximum_action_nearest_distance=float(maximum_nearest),
            reasons=tuple(reasons),
            protocol_hash=self.config.content_hash,
        )
        target_dep = index.parent_dependence_group[target_parent]
        return DependenceAwareTeacherEvidenceR6(
            evidence_id=f"R6E:{target_parent}:{self.config.content_hash[:12]}",
            parent_id=target_parent,
            student_context_object_id=rows[0].student_context_object_id,
            target_dependence_group_id=target_dep,
            timestamp=rows[0].timestamp,
            teacher_version=self.config.teacher_version,
            teacher_protocol_hash=self.config.content_hash,
            train_dependence_group_hash=canonical_hash(train_deps),
            action_laws=tuple(laws),
            direction_target_probs=tuple(float(x) for x in probs),
            requested_risk_target=risk_target,
            direction_weight=self.config.direction_weight if admission.admitted else 0.0,
            sizing_weight=self.config.sizing_weight if admission.admitted else 0.0,
            admission=admission,
        )


def _encode_cache_json(obj: Any) -> Any:
    """Encode legal Teacher infinities while keeping JSON strict and deterministic.

    Rejected early-support evidence can legitimately contain +inf nearest-distance.
    NaN is never legitimate and remains a hard failure.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            raise RuntimeError("R102_COMPILED_TEACHER_NAN_REFUSED")
        if math.isinf(obj):
            return {NONFINITE_TAG: "POS_INF" if obj > 0 else "NEG_INF"}
        return obj
    if isinstance(obj, dict):
        return {str(k): _encode_cache_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_encode_cache_json(v) for v in obj]
    return obj


def _decode_cache_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        if set(obj) == {NONFINITE_TAG}:
            tag = obj[NONFINITE_TAG]
            if tag == "POS_INF":
                return float("inf")
            if tag == "NEG_INF":
                return float("-inf")
            raise RuntimeError("R102_COMPILED_TEACHER_NONFINITE_TAG_INVALID")
        return {k: _decode_cache_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_decode_cache_json(v) for v in obj]
    return obj


def _evidence_from_obj(obj: Mapping[str, Any]) -> DependenceAwareTeacherEvidenceR6:
    laws = tuple(
        DependenceAwarePredictiveLawR6(
            direction=int(x["direction"]),
            requested_risk=float(x["requested_risk"]),
            mean_utility=float(x["mean_utility"]),
            std_utility=float(x["std_utility"]),
            quantile_levels=tuple(float(v) for v in x["quantile_levels"]),
            quantiles=tuple(float(v) for v in x["quantiles"]),
            effective_dependence_n=float(x["effective_dependence_n"]),
            unique_dependence_groups=int(x["unique_dependence_groups"]),
            nearest_distance=float(x["nearest_distance"]),
            max_distance_used=float(x["max_distance_used"]),
            support_dependence_group_hash=str(x["support_dependence_group_hash"]),
        )
        for x in obj["action_laws"]
    )
    admission_obj = obj["admission"]
    admission = EvidenceAdmissionReceiptR6(
        status=str(admission_obj["status"]),
        lane=str(admission_obj["lane"]),
        unique_train_dependence_groups=int(admission_obj["unique_train_dependence_groups"]),
        minimum_action_effective_dependence_n=float(
            admission_obj["minimum_action_effective_dependence_n"]
        ),
        maximum_action_nearest_distance=float(
            admission_obj["maximum_action_nearest_distance"]
        ),
        reasons=tuple(str(x) for x in admission_obj["reasons"]),
        protocol_hash=str(admission_obj["protocol_hash"]),
    )
    return DependenceAwareTeacherEvidenceR6(
        evidence_id=str(obj["evidence_id"]),
        parent_id=str(obj["parent_id"]),
        student_context_object_id=str(obj["student_context_object_id"]),
        target_dependence_group_id=str(obj["target_dependence_group_id"]),
        timestamp=int(obj["timestamp"]),
        teacher_version=str(obj["teacher_version"]),
        teacher_protocol_hash=str(obj["teacher_protocol_hash"]),
        train_dependence_group_hash=str(obj["train_dependence_group_hash"]),
        action_laws=laws,
        direction_target_probs=tuple(float(x) for x in obj["direction_target_probs"]),
        requested_risk_target=float(obj["requested_risk_target"]),
        direction_weight=float(obj["direction_weight"]),
        sizing_weight=float(obj["sizing_weight"]),
        admission=admission,
    )


def _write_deterministic_gzip_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = _encode_cache_json(obj)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(
                filename="", mode="wb", fileobj=raw, compresslevel=4, mtime=0
            ) as gz:
                gz.write(canonical_json_bytes(encoded))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        encoded = json.loads(handle.read())
    return _decode_cache_json(encoded)


def teacher_authority_identity(
    *,
    source_identity: Mapping[str, Any],
    train_config: DependenceAwareTeacherConfigR6,
    val_config: DependenceAwareTeacherConfigR6,
) -> dict[str, Any]:
    parents_sha = str(source_identity.get("parents_sha256", ""))
    branches_sha = str(source_identity.get("branches_sha256", ""))
    if len(parents_sha) != 64 or len(branches_sha) != 64:
        raise RuntimeError("R102_TEACHER_SOURCE_IDENTITY_MISSING_SHA256")
    r6_path = Path(
        __import__("cb16_local_opt.probabilistic_teacher_r6", fromlist=["x"]).__file__
    ).resolve()
    r5_path = Path(
        __import__("cb16_local_opt.probabilistic_teacher_r5", fromlist=["x"]).__file__
    ).resolve()
    return {
        "schema": COMPILED_TEACHER_IDENTITY_SCHEMA,
        "source_parents_sha256": parents_sha,
        "source_branches_sha256": branches_sha,
        "train_teacher_protocol_hash": train_config.content_hash,
        "validation_teacher_protocol_hash": val_config.content_hash,
        "teacher_r6_file_sha256": sha256_file(r6_path),
        "weighted_quantile_r5_file_sha256": sha256_file(r5_path),
        "incremental_compiler_file_sha256": sha256_file(Path(__file__).resolve()),
        "numpy_version": np.__version__,
        "compiler_semantics": COMPILER_SEMANTICS,
        "payload_encoding": "STRICT_JSON_WITH_TAGGED_INFINITY_V1",
        "scheduler_parameters_in_scientific_identity": False,
    }


def _compile_exact(
    *,
    samples,
    parents: Mapping[str, ParentContextR102],
    train_config: DependenceAwareTeacherConfigR6,
    val_config: DependenceAwareTeacherConfigR6,
    workers: int,
    threads_per_worker: int,
    max_in_flight: int,
):
    train_parent_ids = sorted(
        p.parent_id for p in parents.values() if p.split == "TRAIN"
    )
    val_parent_ids = sorted(
        p.parent_id for p in parents.values() if p.split == "VALIDATION"
    )
    train_groups = {
        p.dependence_group_id for p in parents.values() if p.split == "TRAIN"
    }
    if int(workers) <= 1:
        train_teacher = ExactIncrementalTeacherR6(train_config)
        val_teacher = ExactIncrementalTeacherR6(val_config)
        # TeacherIndexR6 is a pure function of samples and independent of config.
        index = train_teacher.index(samples)
        train_evidence = [
            train_teacher.compile_one(
                target_parent=parent_id,
                index=index,
                eligible_train_dependence_groups=train_groups,
            )
            for parent_id in train_parent_ids
            if parent_id in index.rows_by_parent
        ]
        val_evidence = [
            val_teacher.compile_one(
                target_parent=parent_id,
                index=index,
                eligible_train_dependence_groups=train_groups,
            )
            for parent_id in val_parent_ids
            if parent_id in index.rows_by_parent
        ]
        return train_evidence, val_evidence

    from .r102_teacher_parallel import compile_teacher_evidence_process_farm_r102

    return compile_teacher_evidence_process_farm_r102(
        samples=samples,
        train_parent_ids=train_parent_ids,
        val_parent_ids=val_parent_ids,
        train_groups=train_groups,
        train_config=train_config,
        val_config=val_config,
        workers=int(workers),
        threads_per_worker=int(threads_per_worker),
        max_in_flight=int(max_in_flight),
        incremental_geometry=True,
    )


def compile_teacher_evidence_incremental(
    *,
    samples,
    parents: Mapping[str, ParentContextR102],
    source_identity: Mapping[str, Any],
    cache_root: str | Path,
    train_config: DependenceAwareTeacherConfigR6,
    val_config: DependenceAwareTeacherConfigR6,
    workers: int | None = None,
    threads_per_worker: int | None = None,
    max_in_flight: int | None = None,
) -> tuple[
    list[DependenceAwareTeacherEvidenceR6],
    list[DependenceAwareTeacherEvidenceR6],
    dict[str, Any],
]:
    if workers is None or threads_per_worker is None or max_in_flight is None:
        from .r102_runtime_authority import load_r102_runtime_parallelism

        runtime = load_r102_runtime_parallelism(
            Path(__file__).resolve().parents[1], live_environment_check=False
        )
        workers = runtime.teacher_workers if workers is None else workers
        threads_per_worker = (
            runtime.teacher_threads_per_worker
            if threads_per_worker is None
            else threads_per_worker
        )
        max_in_flight = (
            runtime.h72_max_in_flight if max_in_flight is None else max_in_flight
        )

    identity = teacher_authority_identity(
        source_identity=source_identity,
        train_config=train_config,
        val_config=val_config,
    )
    authority_hash = sha256_obj(identity)
    root = Path(cache_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload_path = root / f"{authority_hash}.teacher.json.gz"
    manifest_path = root / f"{authority_hash}.manifest.json"

    if manifest_path.exists():
        if not payload_path.is_file():
            raise RuntimeError("R102_COMPILED_TEACHER_MANIFEST_WITHOUT_PAYLOAD")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != COMPILED_TEACHER_CACHE_SCHEMA:
            raise RuntimeError("R102_COMPILED_TEACHER_SCHEMA_MISMATCH")
        if (
            manifest.get("authority_hash") != authority_hash
            or manifest.get("identity") != identity
        ):
            raise RuntimeError("R102_COMPILED_TEACHER_IDENTITY_CONFLICT")
        observed_payload_sha = sha256_file(payload_path)
        if observed_payload_sha != manifest.get("payload_sha256"):
            raise RuntimeError("R102_COMPILED_TEACHER_PAYLOAD_HASH_MISMATCH")
        started = time.perf_counter()
        payload = _load_gzip_json(payload_path)
        if payload.get("schema") != COMPILED_TEACHER_PAYLOAD_SCHEMA:
            raise RuntimeError("R102_COMPILED_TEACHER_PAYLOAD_SCHEMA_MISMATCH")
        if payload.get("authority_hash") != authority_hash:
            raise RuntimeError("R102_COMPILED_TEACHER_PAYLOAD_IDENTITY_MISMATCH")
        train_evidence = [_evidence_from_obj(x) for x in payload["train"]]
        val_evidence = [_evidence_from_obj(x) for x in payload["validation"]]
        receipt = {
            "schema": COMPILED_TEACHER_RECEIPT_SCHEMA,
            "mode": "REUSED_VERIFIED_AUTHORITY",
            "authority_hash": authority_hash,
            "payload_path": str(payload_path),
            "payload_sha256": observed_payload_sha,
            "train_count": len(train_evidence),
            "validation_count": len(val_evidence),
            "compile_seconds": 0.0,
            "load_seconds": time.perf_counter() - started,
            "scientific_semantics_changed": False,
            "scheduler_parameters_in_scientific_identity": False,
        }
        return train_evidence, val_evidence, receipt

    started = time.perf_counter()
    train_evidence, val_evidence = _compile_exact(
        samples=samples,
        parents=parents,
        train_config=train_config,
        val_config=val_config,
        workers=int(workers),
        threads_per_worker=int(threads_per_worker),
        max_in_flight=int(max_in_flight),
    )
    compile_seconds = time.perf_counter() - started
    payload = {
        "schema": COMPILED_TEACHER_PAYLOAD_SCHEMA,
        "authority_hash": authority_hash,
        "train": [asdict(x) for x in train_evidence],
        "validation": [asdict(x) for x in val_evidence],
    }
    _write_deterministic_gzip_json(payload_path, payload)
    payload_sha = sha256_file(payload_path)
    manifest = {
        "schema": COMPILED_TEACHER_CACHE_SCHEMA,
        "authority_hash": authority_hash,
        "identity": identity,
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "train_count": len(train_evidence),
        "validation_count": len(val_evidence),
        "scientific_semantics_changed": False,
        "scheduler_parameters_in_scientific_identity": False,
    }
    # Manifest is the publication seal.  Orphan payloads are never authority.
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, manifest_path)
    receipt = {
        "schema": COMPILED_TEACHER_RECEIPT_SCHEMA,
        "mode": "COMPILED_AND_PUBLISHED",
        "authority_hash": authority_hash,
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "train_count": len(train_evidence),
        "validation_count": len(val_evidence),
        "compile_seconds": compile_seconds,
        "load_seconds": 0.0,
        "scientific_semantics_changed": False,
        "scheduler_parameters_in_scientific_identity": False,
    }
    return train_evidence, val_evidence, receipt