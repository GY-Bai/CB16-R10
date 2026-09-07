from __future__ import annotations

"""P0 incremental Teacher compilation for R10.2/R2.

This module preserves the frozen R6 Teacher semantics while removing two classes of
pure recomputation:

1) the same compiled Teacher evidence can be reused across replay/campaign runs when
   the immutable source files, Teacher protocols, and compiler implementation hashes
   are identical;
2) within one target parent, normalization and context distances are computed once
   and reused across the complete action/risk grid.

The cache is fail-closed and content addressed. Scheduling parameters (worker count,
threads, queue depth) are deliberately excluded from scientific identity.
"""

import gzip
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

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


COMPILED_TEACHER_CACHE_SCHEMA = "CB16_R102_COMPILED_TEACHER_AUTHORITY_V1"
COMPILED_TEACHER_IDENTITY_SCHEMA = "CB16_R102_COMPILED_TEACHER_IDENTITY_V1"
COMPILED_TEACHER_PAYLOAD_SCHEMA = "CB16_R102_COMPILED_TEACHER_PAYLOAD_V1"
COMPILER_SEMANTICS = "R6_EXACT_DISTRIBUTIONAL_TEACHER__P0_GEOMETRY_REUSE_V1"


class ExactIncrementalTeacherR6(DependenceAwareProbabilisticTeacherR6):
    """Exact R6 Teacher with target-local geometry reuse.

    The base implementation recomputes normalization and feature distances once per
    action/risk branch.  Here those values are computed once per target parent and
    reused across the branch grid.  Candidate selection, weighting, quantiles,
    admission rules, and output dataclasses remain byte-for-byte semantic equivalents.
    """

    def _prepare_geometry(
        self,
        *,
        target_features: tuple[float, ...] | np.ndarray,
        train_deps: Sequence[str],
        index: TeacherIndexR6,
        feature_override: Mapping[str, tuple[float, ...]] | None = None,
    ) -> dict[str, Any]:
        mean, std = self._normalization(
            train_deps=train_deps,
            index=index,
            feature_override=feature_override,
        )
        t = (np.asarray(target_features, dtype=np.float64) - mean) / std
        distance_by_parent: dict[str, float] = {}
        for dep in train_deps:
            for p in index.parents_by_dependence_group[dep]:
                feat = (
                    feature_override[p]
                    if feature_override is not None
                    else index.rows_by_parent[p][0].context_features
                )
                z = (np.asarray(feat, dtype=np.float64) - mean) / std
                distance_by_parent[p] = float(np.sqrt(np.mean((z - t) ** 2)))
        return {
            "mean": mean,
            "std": std,
            "target": t,
            "distance_by_parent": distance_by_parent,
        }

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
            for p in index.parents_by_dependence_group[dep]:
                match = [
                    s for s in index.rows_by_parent[p]
                    if s.direction == direction
                    and abs(s.requested_risk - risk) <= 1e-12
                ]
                if not match:
                    continue
                if len(match) != 1:
                    raise RuntimeError("DUPLICATE_ACTION_BRANCH_WITHIN_PARENT")
                dist = float(distance_by_parent[p])
                row = (dist, p, float(match[0].realized_utility))
                if best is None or row[:2] < best[:2]:
                    best = row
            if best is not None:
                candidates.append((best[0], dep, best[1], best[2]))

        if not candidates:
            return None
        candidates.sort(key=lambda x: (x[0], x[1], x[2]))
        selected = candidates[: min(self.config.k_dependence_groups, len(candidates))]
        d = np.asarray([x[0] for x in selected], dtype=np.float64)
        y = np.asarray([x[3] for x in selected], dtype=np.float64)
        deps = [x[1] for x in selected]
        w = np.exp(-0.5 * (d / self.config.distance_temperature) ** 2) + 1e-12
        w /= w.sum()
        q = weighted_quantile(
            y,
            w,
            np.asarray(self.config.quantile_levels, dtype=np.float64),
        )
        mu = float(np.sum(w * y))
        var = float(np.sum(w * (y - mu) ** 2))
        eff = float(1.0 / np.sum(w ** 2))
        return DependenceAwarePredictiveLawR6(
            direction=direction,
            requested_risk=float(risk),
            mean_utility=mu,
            std_utility=math.sqrt(max(0.0, var)),
            quantile_levels=self.config.quantile_levels,
            quantiles=tuple(float(x) for x in q),
            effective_dependence_n=eff,
            unique_dependence_groups=len(deps),
            nearest_distance=float(d[0]),
            max_distance_used=float(d[-1]),
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
            geometry = self._prepare_geometry(
                target_features=target_features,
                train_deps=train_deps,
                index=index,
                feature_override=feature_override,
            )
            distance_by_parent = geometry["distance_by_parent"]
            for d, r in grid:
                law = self._predictive_law_from_geometry(
                    train_deps=train_deps,
                    index=index,
                    direction=d,
                    risk=r,
                    distance_by_parent=distance_by_parent,
                )
                if law is not None:
                    laws.append(law)

        best_by_direction = {}
        for d in (-1, 0, 1):
            x = [l for l in laws if l.direction == d]
            if x:
                best_by_direction[d] = max(
                    x,
                    key=lambda l: (l.mean_utility, -l.requested_risk),
                )

        reasons = []
        if len(train_deps) < self.config.min_train_dependence_groups:
            reasons.append("INSUFFICIENT_TRAIN_DEPENDENCE_GROUPS")
        if set(best_by_direction) != {-1, 0, 1}:
            reasons.append("ACTION_GRID_SUPPORT_INCOMPLETE")
        min_eff = min(
            (l.effective_dependence_n for l in best_by_direction.values()),
            default=0.0,
        )
        max_nearest = max(
            (l.nearest_distance for l in best_by_direction.values()),
            default=float("inf"),
        )
        if min_eff < self.config.min_effective_dependence_n:
            reasons.append("INSUFFICIENT_EFFECTIVE_DEPENDENCE_SUPPORT")
        if max_nearest > self.config.max_nearest_distance:
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
            minimum_action_effective_dependence_n=float(min_eff),
            maximum_action_nearest_distance=float(max_nearest),
            reasons=tuple(reasons),
            protocol_hash=self.config.content_hash,
        )
        dep = index.parent_dependence_group[target_parent]
        return DependenceAwareTeacherEvidenceR6(
            evidence_id=f"R6E:{target_parent}:{self.config.content_hash[:12]}",
            parent_id=target_parent,
            student_context_object_id=rows[0].student_context_object_id,
            target_dependence_group_id=dep,
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
    a = obj["admission"]
    admission = EvidenceAdmissionReceiptR6(
        status=str(a["status"]),
        lane=str(a["lane"]),
        unique_train_dependence_groups=int(a["unique_train_dependence_groups"]),
        minimum_action_effective_dependence_n=float(a["minimum_action_effective_dependence_n"]),
        maximum_action_nearest_distance=float(a["maximum_action_nearest_distance"]),
        reasons=tuple(str(x) for x in a["reasons"]),
        protocol_hash=str(a["protocol_hash"]),
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
    fd, tmp_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=4, mtime=0) as gz:
                gz.write(canonical_json_bytes(obj))
            raw.flush()
            os.fsync(raw.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rb") as f:
        return json.loads(f.read())


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
    core_path = Path(__import__("cb16_local_opt.probabilistic_teacher_r6", fromlist=["x"]).__file__).resolve()
    this_path = Path(__file__).resolve()
    return {
        "schema": COMPILED_TEACHER_IDENTITY_SCHEMA,
        "source_parents_sha256": parents_sha,
        "source_branches_sha256": branches_sha,
        "train_teacher_protocol_hash": train_config.content_hash,
        "validation_teacher_protocol_hash": val_config.content_hash,
        "teacher_core_file_sha256": sha256_file(core_path),
        "incremental_compiler_file_sha256": sha256_file(this_path),
        "compiler_semantics": COMPILER_SEMANTICS,
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
    train_parent_ids = sorted(p.parent_id for p in parents.values() if p.split == "TRAIN")
    val_parent_ids = sorted(p.parent_id for p in parents.values() if p.split == "VALIDATION")
    train_groups = {p.dependence_group_id for p in parents.values() if p.split == "TRAIN"}
    if int(workers) <= 1:
        train_teacher = ExactIncrementalTeacherR6(train_config)
        val_teacher = ExactIncrementalTeacherR6(val_config)
        # TeacherIndexR6 depends only on samples, not Teacher config.  Build once.
        index = train_teacher.index(samples)
        train_e = [
            train_teacher.compile_one(
                target_parent=p,
                index=index,
                eligible_train_dependence_groups=train_groups,
            )
            for p in train_parent_ids
            if p in index.rows_by_parent
        ]
        val_e = [
            val_teacher.compile_one(
                target_parent=p,
                index=index,
                eligible_train_dependence_groups=train_groups,
            )
            for p in val_parent_ids
            if p in index.rows_by_parent
        ]
        return train_e, val_e

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
) -> tuple[list[DependenceAwareTeacherEvidenceR6], list[DependenceAwareTeacherEvidenceR6], dict[str, Any]]:
    if workers is None or threads_per_worker is None or max_in_flight is None:
        from .r102_runtime_authority import load_r102_runtime_parallelism

        rp = load_r102_runtime_parallelism(Path(__file__).resolve().parents[1], live_environment_check=False)
        workers = rp.teacher_workers if workers is None else workers
        threads_per_worker = rp.teacher_threads_per_worker if threads_per_worker is None else threads_per_worker
        max_in_flight = rp.h72_max_in_flight if max_in_flight is None else max_in_flight

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
        if manifest.get("authority_hash") != authority_hash or manifest.get("identity") != identity:
            raise RuntimeError("R102_COMPILED_TEACHER_IDENTITY_CONFLICT")
        observed = sha256_file(payload_path)
        if observed != manifest.get("payload_sha256"):
            raise RuntimeError("R102_COMPILED_TEACHER_PAYLOAD_HASH_MISMATCH")
        started = time.perf_counter()
        payload = _load_gzip_json(payload_path)
        if payload.get("schema") != COMPILED_TEACHER_PAYLOAD_SCHEMA:
            raise RuntimeError("R102_COMPILED_TEACHER_PAYLOAD_SCHEMA_MISMATCH")
        if payload.get("authority_hash") != authority_hash:
            raise RuntimeError("R102_COMPILED_TEACHER_PAYLOAD_IDENTITY_MISMATCH")
        train_e = [_evidence_from_obj(x) for x in payload["train"]]
        val_e = [_evidence_from_obj(x) for x in payload["validation"]]
        receipt = {
            "schema": "CB16_R102_COMPILED_TEACHER_RECEIPT_V1",
            "mode": "REUSED_VERIFIED_AUTHORITY",
            "authority_hash": authority_hash,
            "payload_path": str(payload_path),
            "payload_sha256": observed,
            "train_count": len(train_e),
            "validation_count": len(val_e),
            "compile_seconds": 0.0,
            "load_seconds": time.perf_counter() - started,
            "scientific_semantics_changed": False,
            "scheduler_parameters_in_scientific_identity": False,
        }
        return train_e, val_e, receipt

    started = time.perf_counter()
    train_e, val_e = _compile_exact(
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
        "train": [asdict(x) for x in train_e],
        "validation": [asdict(x) for x in val_e],
    }
    _write_deterministic_gzip_json(payload_path, payload)
    payload_sha = sha256_file(payload_path)
    manifest = {
        "schema": COMPILED_TEACHER_CACHE_SCHEMA,
        "authority_hash": authority_hash,
        "identity": identity,
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "train_count": len(train_e),
        "validation_count": len(val_e),
        "scientific_semantics_changed": False,
        "scheduler_parameters_in_scientific_identity": False,
    }
    # Manifest is the publication seal; payload without a manifest is never authority.
    tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, manifest_path)
    receipt = {
        "schema": "CB16_R102_COMPILED_TEACHER_RECEIPT_V1",
        "mode": "COMPILED_AND_PUBLISHED",
        "authority_hash": authority_hash,
        "payload_path": str(payload_path),
        "payload_sha256": payload_sha,
        "train_count": len(train_e),
        "validation_count": len(val_e),
        "compile_seconds": compile_seconds,
        "load_seconds": 0.0,
        "scientific_semantics_changed": False,
        "scheduler_parameters_in_scientific_identity": False,
    }
    return train_e, val_e, receipt
