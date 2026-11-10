"""
Validated dependency construction for AON diagrams.

Separates detected arrows from valid dependency edges through strict
geometric validation, deduplication, and confidence scoring.

Pipeline:
  DetectedArrow candidates
  → geometric validation (boundary, direction, consistency)
  → arrow deduplication (collinear, overlapping, same-region)
  → confidence scoring with evidence
  → HIGH/MEDIUM/LOW classification
  → validated dependencies (HIGH only) + review candidates (MEDIUM)
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.models import CandidateNode, DetectedArrow, ShapeType

logger = logging.getLogger(__name__)


class DependencyConfidence(Enum):
    """Confidence level for validated dependencies."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    REJECTED = "rejected"


@dataclass
class DependencyEvidence:
    """Individual evidence factors for a dependency candidate."""
    arrowhead_confidence: float = 0.0
    source_boundary_intersection: float = 0.0
    target_boundary_intersection: float = 0.0
    direction_consistency: float = 0.0
    distance_from_source_boundary: float = 0.0
    distance_from_target_boundary: float = 0.0
    angular_consistency: float = 0.0
    line_continuity: float = 0.0
    shaft_visual_support: float = 0.0
    visual_route_support: float = 0.0
    node_interior_crossing_penalty: float = 0.0
    duplicate_arrow_penalty: float = 0.0
    raw_evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidatedDependency:
    """A validated dependency with full evidence provenance."""
    source_id: str
    target_id: str
    arrow_id: str
    confidence_level: DependencyConfidence = DependencyConfidence.LOW
    confidence_score: float = 0.0
    evidence: DependencyEvidence = field(default_factory=DependencyEvidence)
    accepted: bool = False
    review_required: bool = False
    rejection_reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DependencyValidationReport:
    """Report of the dependency validation process."""
    raw_arrow_count: int = 0
    deduplicated_arrow_count: int = 0
    accepted_count: int = 0
    review_count: int = 0
    rejected_count: int = 0
    self_loop_count: int = 0
    duplicate_edge_count: int = 0
    validated_dependencies: List[ValidatedDependency] = field(default_factory=list)
    rejected_candidates: List[ValidatedDependency] = field(default_factory=list)
    review_candidates: List[ValidatedDependency] = field(default_factory=list)
    debug_entries: List[Dict[str, Any]] = field(default_factory=list)
    # Aggregated per-gate pass/fail counts over every evaluated candidate
    # (diagnostics only; the decision logic itself lives in
    # ValidatedDependencyBuilder._can_auto_accept_visual_review).
    gate_digest: Dict[str, Any] = field(default_factory=dict)


class ValidatedDependencyBuilder:
    """
    Builds validated dependencies from detected arrows using strict geometry.

    Unlike the basic GeometryAssociator, this builder:
    - Requires boundary intersection (not just proximity)
    - Requires directional consistency
    - Deduplicates arrows before dependency construction
    - Classifies confidence into HIGH/MEDIUM/LOW
    - Provides full evidence for each decision
    """

    def __init__(
        self,
        high_threshold: float = 0.55,
        medium_threshold: float = 0.35,
        min_boundary_distance: float = 80.0,
        max_direction_tolerance_deg: float = 40.0,
        require_boundary_intersection: bool = True,
    ):
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold
        self.min_boundary_distance = min_boundary_distance
        self.max_direction_tolerance_deg = max_direction_tolerance_deg
        self.require_boundary_intersection = require_boundary_intersection

    def build_validated_dependencies(
        self,
        arrows: List[DetectedArrow],
        rectangle_candidates: List[CandidateNode],
    ) -> DependencyValidationReport:
        """
        Build validated dependencies from detected arrows.

        Args:
            arrows: Detected arrows from arrow detection.
            rectangle_candidates: Rectangle candidate nodes.

        Returns:
            DependencyValidationReport with all findings.
        """
        report = DependencyValidationReport()
        report.raw_arrow_count = len(arrows)

        # Step 1: Deduplicate arrows
        deduped = self._deduplicate_arrows(arrows)
        report.deduplicated_arrow_count = len(deduped)

        # Step 2: Validate each deduplicated arrow
        candidates: List[ValidatedDependency] = []
        for arrow in deduped:
            dep = self._validate_arrow(arrow, rectangle_candidates)
            candidates.append(dep)

        # Step 3: Classify and deduplicate edges
        self._classify_candidates(candidates, report)
        self._deduplicate_edges(report)

        # Step 4: Build debug output
        self._build_debug_entries(candidates, report)

        logger.info(
            "Dependency validation: %d raw → %d deduped → %d accepted, "
            "%d review, %d rejected",
            report.raw_arrow_count,
            report.deduplicated_arrow_count,
            report.accepted_count,
            report.review_count,
            report.rejected_count,
        )

        return report

    # =========================================================================
    # Arrow Deduplication
    # =========================================================================

    def _deduplicate_arrows(
        self, arrows: List[DetectedArrow]
    ) -> List[DetectedArrow]:
        """
        Deduplicate arrows that represent the same visual arrow.

        Multiple HoughLinesP segments may produce overlapping arrows.
        This collapses them based on proximity, direction, and overlap.
        """
        if len(arrows) <= 1:
            return list(arrows)

        used = [False] * len(arrows)
        deduped: List[DetectedArrow] = []

        for i in range(len(arrows)):
            if used[i]:
                continue
            group = [arrows[i]]
            used[i] = True

            for j in range(i + 1, len(arrows)):
                if used[j]:
                    continue
                if self._should_dedup_arrows(arrows[i], arrows[j]):
                    group.append(arrows[j])
                    used[j] = True

            # Keep the arrow with highest confidence from each group
            best = max(group, key=lambda a: a.confidence)
            deduped.append(best)

        return deduped

    def _should_dedup_arrows(
        self, a1: DetectedArrow, a2: DetectedArrow
    ) -> bool:
        """Determine if two arrows represent the same visual arrow."""
        # Direction similarity
        dx1, dy1 = a1.direction_vector
        dx2, dy2 = a2.direction_vector
        len1 = math.sqrt(dx1**2 + dy1**2)
        len2 = math.sqrt(dx2**2 + dy2**2)
        if len1 < 1e-6 or len2 < 1e-6:
            return False

        cos_angle = (dx1*dx2 + dy1*dy2) / (len1 * len2)
        if abs(cos_angle) < 0.7:  # ~45 degrees
            return False

        # Endpoint proximity
        min_dist = min(
            a1.start.distance_to(a2.start),
            a1.start.distance_to(a2.end),
            a1.end.distance_to(a2.start),
            a1.end.distance_to(a2.end),
        )
        if min_dist > 40:
            return False

        # Midpoint proximity
        mid1 = a1.midpoint
        mid2 = a2.midpoint
        mid_dist = mid1.distance_to(mid2)
        if mid_dist > 50:
            return False

        # Length similarity
        length_ratio = min(a1.length, a2.length) / max(a1.length, a2.length) if max(a1.length, a2.length) > 0 else 0
        if length_ratio < 0.3:
            return False

        return True

    # =========================================================================
    # Arrow Validation
    # =========================================================================

    def _validate_arrow(
        self,
        arrow: DetectedArrow,
        rectangle_candidates: List[CandidateNode],
    ) -> ValidatedDependency:
        """Validate a single arrow as a potential dependency.

        Pipeline: Node Pairing → Direction Resolution → ValidatedDependency.
        """
        from pert_analyzer.cv.direction_resolution import (
            ArrowDirectionResolver,
            DirectionStatus,
        )
        from pert_analyzer.cv.node_pairing import PairingStatus, NodePairingResolver

        dep = ValidatedDependency(
            source_id="",
            target_id="",
            arrow_id=arrow.arrow_id,
        )
        dep.metadata["arrow_confidence"] = arrow.confidence
        dep.metadata["arrow_length"] = arrow.length
        dep.metadata["arrow_start"] = (arrow.start.x, arrow.start.y)
        dep.metadata["arrow_end"] = (arrow.end.x, arrow.end.y)

        # Stage 1: Node Pairing — which two nodes does this arrow connect?
        pairing_resolver = NodePairingResolver(
            max_endpoint_distance=120.0,
            confirmed_margin=0.15,
            min_boundary_distance=self.min_boundary_distance,
        )
        pair_result = pairing_resolver.resolve_pairing(arrow, rectangle_candidates)

        dep.metadata["pairing_status"] = pair_result.status.value
        dep.metadata["pairing_confidence"] = pair_result.confidence
        dep.metadata["pairing_alternatives"] = [
            {
                "node_a": alt.node_a_id,
                "node_b": alt.node_b_id,
                "score": alt.score,
                "evidence": alt.evidence.to_dict(),
            }
            for alt in pair_result.alternatives
        ]

        if pair_result.status == PairingStatus.PAIR_REJECTED:
            dep.rejection_reasons.extend(pair_result.rejection_reasons)
            dep.metadata["pairing_rejection"] = pair_result.rejection_reasons
            return dep

        selected_pair = pair_result.selected
        if not selected_pair:
            dep.rejection_reasons.append("no_selected_pair")
            return dep
        dep.metadata["crossing_penalty"] = selected_pair.evidence.crossing_penalty

        # Stage 2: Direction Resolution — which direction for this pair?
        direction_resolver = ArrowDirectionResolver(
            confirmed_margin=0.20,
            min_boundary_distance=self.min_boundary_distance,
            max_direction_tolerance_deg=self.max_direction_tolerance_deg,
        )
        dir_result = direction_resolver.resolve_direction_for_pair(
            arrow, selected_pair.node_a_id, selected_pair.node_b_id,
            rectangle_candidates,
        )

        dep.metadata["direction_status"] = dir_result.status.value
        dep.metadata["direction_margin"] = dir_result.margin
        dep.metadata["direction_alternatives"] = dir_result.alternatives

        if dir_result.status == DirectionStatus.REJECTED:
            dep.rejection_reasons.extend(dir_result.rejection_reasons)
            dep.metadata["direction_rejection"] = dir_result.rejection_reasons
            return dep

        dep.source_id = dir_result.source_id
        dep.target_id = dir_result.target_id

        # Build evidence combining pairing + direction
        dep.evidence = DependencyEvidence(
            arrowhead_confidence=arrow.arrowhead_confidence,
            source_boundary_intersection=dir_result.evidence.get("source_boundary_contact", 0),
            target_boundary_intersection=dir_result.evidence.get("target_boundary_contact", 0),
            direction_consistency=dir_result.evidence.get("direction_alignment", 0),
            angular_consistency=dir_result.evidence.get("angular_consistency", 0),
            shaft_visual_support=self._shaft_visual_support(arrow),
            visual_route_support=self._visual_route_support(
                arrow, selected_pair.evidence
            ),
            raw_evidence={
                "pairing": {
                    "status": pair_result.status.value,
                    "confidence": pair_result.confidence,
                    "selected_pair": {
                        "node_a": selected_pair.node_a_id,
                        "node_b": selected_pair.node_b_id,
                        "score": selected_pair.score,
                        "evidence": selected_pair.evidence.to_dict(),
                    },
                },
                "direction": {
                    "status": dir_result.status.value,
                    "margin": dir_result.margin,
                    "evidence": dir_result.evidence,
                    "orientation_a_to_b": dir_result.orientation_a_to_b.score if dir_result.orientation_a_to_b else 0,
                    "orientation_b_to_a": dir_result.orientation_b_to_a.score if dir_result.orientation_b_to_a else 0,
                },
                "visual": {
                    "shaft_support": self._shaft_visual_support(arrow),
                    "route_support": self._visual_route_support(
                        arrow, selected_pair.evidence
                    ),
                    "detector_evidence": arrow.evidence.get("shaft_continuity", {}),
                    "pairing_evidence": selected_pair.evidence.to_dict(),
                },
            },
        )

        # Score: combine pairing confidence + direction confidence
        pair_weight = 0.5
        dir_weight = 0.5
        geometry_score = (
            pair_weight * pair_result.confidence +
            dir_weight * dir_result.confidence
        )
        # A real detected shaft is useful corroboration, but never substitutes
        # for node-boundary and direction evidence.  This is deliberately a
        # bounded bonus rather than a lower threshold or an unconditional
        # acceptance path: weak/noisy geometry remains review or rejected.
        dep.confidence_score = min(
            1.0,
            geometry_score + 0.08 * dep.evidence.shaft_visual_support,
        )

        # A shaft that crosses another detected node is visual evidence of a
        # routed bus, branch, or incorrect endpoint pair—not a clean direct
        # dependency. Keep it reviewable even when aggregate confidence is
        # high; do not silently accept a geometrically contradicted edge.
        if selected_pair.evidence.crossing_penalty >= 0.30:
            dep.review_required = True
            dep.metadata["visual_confirmation_blocked"] = "intervening_node_crossing"

        # REVIEW if either stage says so. A review candidate may be promoted
        # only when independent pixel-derived shaft evidence corroborates the
        # selected boundary route and direction.
        if (pair_result.status == PairingStatus.PAIR_REVIEW_REQUIRED or
                dir_result.status == DirectionStatus.REVIEW_REQUIRED):
            dep.review_required = True
            if self._can_auto_accept_visual_review(dep):
                dep.review_required = False
                dep.metadata["visual_confirmation"] = {
                    "reason": "shaft_boundary_direction_angular_conjunction",
                    "route_support": dep.evidence.visual_route_support,
                }

        return dep

    @staticmethod
    def _shaft_visual_support(arrow: DetectedArrow) -> float:
        """Return conservative support from the detector's visual shaft evidence.

        ``line_confidence`` and ``shaft_continuity`` are produced from image
        pixels by :mod:`arrow_detection`; they are not semantic edge guesses.
        The value is intentionally capped and used only as a small corroborating
        bonus in ``_validate_arrow``.
        """
        evidence = arrow.evidence or {}
        breakdown = evidence.get("confidence_breakdown", {})
        line_conf = float(
            breakdown.get("line_confidence", evidence.get("line_confidence", 0.0))
            or 0.0
        )
        continuity = evidence.get("shaft_continuity", {}) or {}
        length = float(continuity.get("length", arrow.length) or arrow.length or 0.0)
        length_support = min(1.0, max(0.0, length / 120.0))
        segment_support = min(
            1.0,
            max(0.0, float(len(arrow.source_segment_ids or [])) / 3.0),
        )
        return max(
            0.0,
            min(1.0, 0.60 * line_conf + 0.25 * length_support + 0.15 * segment_support),
        )

    @classmethod
    def _visual_route_support(cls, arrow: DetectedArrow, pairing_evidence: Any) -> float:
        """Combine pixel shaft evidence with the selected geometric route."""
        pairing = pairing_evidence.to_dict() if hasattr(pairing_evidence, "to_dict") else {}
        boundary = min(
            float(pairing.get("boundary_contact_a", 0.0) or 0.0),
            float(pairing.get("boundary_contact_b", 0.0) or 0.0),
        )
        continuity = float(pairing.get("line_continuity", 0.0) or 0.0)
        arrowhead = min(1.0, max(0.0, float(arrow.arrowhead_confidence or 0.0)))
        shaft = cls._shaft_visual_support(arrow)
        return max(
            0.0,
            min(1.0, 0.40 * shaft + 0.25 * continuity + 0.20 * boundary + 0.15 * arrowhead),
        )

    # Promotion-lift thresholds for review candidates. Do not change these
    # values silently: they are the measured six-gate conjunction that
    # decides when a geometrically corroborated review candidate is
    # promoted to an accepted dependency. Diagnostics (gate digest) read
    # these same constants, so debug output can never contradict them.
    PROMOTION_MIN_CONFIDENCE = 0.68
    PROMOTION_MIN_VISUAL_ROUTE_SUPPORT = 0.58
    PROMOTION_MIN_ARROWHEAD = 0.50
    PROMOTION_MIN_BOUNDARY_INTERSECTION = 0.40
    PROMOTION_MAX_CROSSING_PENALTY = 0.30
    PROMOTION_MIN_DIRECTION_CONSISTENCY = 0.85
    PROMOTION_MIN_ANGULAR_CONSISTENCY = 0.80

    @staticmethod
    def _can_auto_accept_visual_review(dep: ValidatedDependency) -> bool:
        """Require a high-quality visual conjunction before resolving review."""
        evidence = dep.evidence
        return (
            float(dep.metadata.get("crossing_penalty", 0.0) or 0.0)
            < ValidatedDependencyBuilder.PROMOTION_MAX_CROSSING_PENALTY
            and
            dep.confidence_score >= ValidatedDependencyBuilder.PROMOTION_MIN_CONFIDENCE
            and evidence.visual_route_support
            >= ValidatedDependencyBuilder.PROMOTION_MIN_VISUAL_ROUTE_SUPPORT
            and evidence.arrowhead_confidence
            >= ValidatedDependencyBuilder.PROMOTION_MIN_ARROWHEAD
            and min(
                evidence.source_boundary_intersection,
                evidence.target_boundary_intersection,
            )
            >= ValidatedDependencyBuilder.PROMOTION_MIN_BOUNDARY_INTERSECTION
            and evidence.direction_consistency
            >= ValidatedDependencyBuilder.PROMOTION_MIN_DIRECTION_CONSISTENCY
            and evidence.angular_consistency
            >= ValidatedDependencyBuilder.PROMOTION_MIN_ANGULAR_CONSISTENCY
        )

    def _get_strict_direction(
        self, arrow: DetectedArrow
    ) -> Optional[Tuple[float, float]]:
        """Get arrow direction with strict validation."""
        # Try direction_vector first
        dx, dy = arrow.direction_vector
        length = math.sqrt(dx**2 + dy**2)

        # Fallback to arrowhead→start or end→start
        if length < 1e-6:
            if arrow.arrowhead_point:
                dx = arrow.arrowhead_point.x - arrow.start.x
                dy = arrow.arrowhead_point.y - arrow.start.y
            else:
                dx = arrow.end.x - arrow.start.x
                dy = arrow.end.y - arrow.start.y
            length = math.sqrt(dx**2 + dy**2)

        if length < 1e-6:
            return None

        # Normalize
        ndx, ndy = dx / length, dy / length

        # Verify arrowhead is at the expected end
        if arrow.arrowhead_point:
            ah_dx = arrow.arrowhead_point.x - arrow.start.x
            ah_dy = arrow.arrowhead_point.y - arrow.start.y
            ah_len = math.sqrt(ah_dx**2 + ah_dy**2)
            if ah_len > 1e-6:
                ah_ndx, ah_ndy = ah_dx / ah_len, ah_dy / ah_len
                # Arrowhead direction should agree with line direction
                dot = ndx * ah_ndx + ndy * ah_ndy
                if dot < 0.3:  # Arrowhead points backward
                    return None

        return (ndx, ndy)

    def _score_candidates_strict(
        self,
        candidates: List[CandidateNode],
        reference_point: Point,
        ndx: float,
        ndy: float,
        is_target: bool,
        arrow: DetectedArrow,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """Score candidates with strict geometric criteria."""
        scored = []
        for cand in candidates:
            bb = cand.bounding_box
            score, evidence = self._score_candidate_strict(
                cand, reference_point, ndx, ndy, is_target, arrow
            )
            if score >= self.medium_threshold:
                scored.append((cand.node_id, score, evidence))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _score_candidate_strict(
        self,
        candidate: CandidateNode,
        reference_point: Point,
        ndx: float,
        ndy: float,
        is_target: bool,
        arrow: DetectedArrow,
    ) -> Tuple[float, Dict[str, float]]:
        """Score a single candidate with strict criteria."""
        bb = candidate.bounding_box
        evidence: Dict[str, float] = {}

        # 1. Boundary intersection (critical for target)
        boundary_score = self._compute_boundary_score(
            reference_point, bb, ndx, ndy, is_target
        )
        evidence["boundary_intersection"] = boundary_score

        # 2. Direction alignment
        direction_score = self._compute_direction_score(
            reference_point, bb, ndx, ndy, is_target
        )
        evidence["direction_alignment"] = direction_score

        # 3. Distance from boundary
        dist = self._distance_to_boundary(reference_point, bb)
        dist_score = max(0, 1.0 - dist / self.min_boundary_distance)
        evidence["distance_score"] = dist_score

        # 4. Angular consistency
        angular_score = self._compute_angular_score(
            reference_point, bb, ndx, ndy, is_target
        )
        evidence["angular_consistency"] = angular_score

        # 5. Node interior crossing penalty
        crossing_penalty = self._compute_crossing_penalty(
            reference_point, bb, ndx, ndy, candidates=[candidate]
        )
        evidence["crossing_penalty"] = crossing_penalty

        # Combine scores (NO center_proximity or inside_bonus)
        score = (
            0.35 * boundary_score
            + 0.25 * direction_score
            + 0.20 * angular_score
            + 0.15 * dist_score
            - 0.15 * crossing_penalty
        )

        # Apply strict minimum boundary requirement
        if self.require_boundary_intersection and boundary_score < 0.2:
            score *= 0.3  # Heavy penalty for no boundary intersection

        return max(0.0, score), evidence

    def _compute_boundary_score(
        self,
        point: Point,
        bb: BoundingBox,
        ndx: float,
        ndy: float,
        is_target: bool,
    ) -> float:
        """Compute boundary intersection score."""
        # For target: ray from point in direction of arrow
        # For source: ray from point OPPOSITE to arrow direction
        if is_target:
            rdx, rdy = ndx, ndy
        else:
            rdx, rdy = -ndx, -ndy

        intersections = self._ray_rectangle_intersections(point, rdx, rdy, bb)
        # Accept t >= 0 (including t=0 for points on boundary)
        valid_intersections = [(x, y, e, t) for x, y, e, t in intersections if t >= 0]

        if valid_intersections:
            # Best intersection is closest to origin
            best = min(valid_intersections, key=lambda ix: ix[3])
            dist = math.sqrt((best[0] - point.x)**2 + (best[1] - point.y)**2)
            if dist <= self.min_boundary_distance:
                # Bonus for being ON the boundary (t very small)
                if best[3] < 1.0:
                    return 0.95
                return 1.0 - (dist / self.min_boundary_distance) * 0.3
            return max(0, 0.5 - dist / (self.min_boundary_distance * 3))

        # Fallback: distance to boundary
        dist = self._distance_to_boundary(point, bb)
        if dist <= self.min_boundary_distance * 0.5:
            return 0.5
        return 0.0

    def _compute_direction_score(
        self,
        point: Point,
        bb: BoundingBox,
        ndx: float,
        ndy: float,
        is_target: bool,
    ) -> float:
        """Compute direction alignment score."""
        center = bb.center
        to_center_x = center.x - point.x
        to_center_y = center.y - point.y
        to_center_len = math.sqrt(to_center_x**2 + to_center_y**2)
        if to_center_len < 1e-6:
            return 0.0

        tcx = to_center_x / to_center_len
        tcy = to_center_y / to_center_len

        if is_target:
            dot = ndx * tcx + ndy * tcy
        else:
            dot = -(ndx * tcx + ndy * tcy)

        # Strict: require positive dot product
        if dot <= 0:
            return 0.0
        return dot

    def _compute_angular_score(
        self,
        point: Point,
        bb: BoundingBox,
        ndx: float,
        ndy: float,
        is_target: bool,
    ) -> float:
        """Compute angular consistency score."""
        center = bb.center
        angle_to_center = math.atan2(center.y - point.y, center.x - point.x)
        arrow_angle = math.atan2(ndy, ndx)

        if is_target:
            angle_diff = abs(angle_to_center - arrow_angle)
        else:
            angle_diff = abs(angle_to_center - (arrow_angle + math.pi))

        angle_diff = angle_diff % (2 * math.pi)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        tolerance_rad = math.radians(self.max_direction_tolerance_deg)
        if angle_diff > tolerance_rad:
            return 0.0
        return 1.0 - (angle_diff / tolerance_rad) * 0.5

    def _compute_crossing_penalty(
        self,
        point: Point,
        bb: BoundingBox,
        ndx: float,
        ndy: float,
        candidates: List[CandidateNode],
    ) -> float:
        """Penalize arrows that cross through unrelated node interiors."""
        penalty = 0.0
        for cand in candidates:
            if cand.bounding_box == bb:
                continue
            cb = cand.bounding_box
            # Check if ray from point crosses this node's interior
            intersections = self._ray_rectangle_intersections(point, ndx, ndy, cb)
            for x, y, edge, t in intersections:
                if t > 0:
                    penalty += 0.3
                    break
        return min(1.0, penalty)

    def _distance_to_boundary(self, point: Point, bb: BoundingBox) -> float:
        """Distance from point to nearest rectangle boundary."""
        cx = max(bb.x, min(point.x, bb.x + bb.width))
        cy = max(bb.y, min(point.y, bb.y + bb.height))
        dx = point.x - cx
        dy = point.y - cy
        return math.sqrt(dx**2 + dy**2)

    def _ray_rectangle_intersections(
        self,
        origin: Point,
        rdx: float,
        rdy: float,
        bb: BoundingBox,
    ) -> List[Tuple[float, float, str, float]]:
        """Find ray-rectangle intersections."""
        intersections = []
        edges = [
            (bb.x, "left"), (bb.x + bb.width, "right"),
            (bb.y, "top"), (bb.y + bb.height, "bottom"),
        ]
        for edge_val, edge_name in edges:
            if edge_name in ("left", "right"):
                if abs(rdx) < 1e-10:
                    continue
                t = (edge_val - origin.x) / rdx
                if t < 0:
                    continue
                y = origin.y + t * rdy
                if bb.y - 1 <= y <= bb.y + bb.height + 1:
                    intersections.append((edge_val, y, edge_name, t))
            else:
                if abs(rdy) < 1e-10:
                    continue
                t = (edge_val - origin.y) / rdy
                if t < 0:
                    continue
                x = origin.x + t * rdx
                if bb.x - 1 <= x <= bb.x + bb.width + 1:
                    intersections.append((x, edge_val, edge_name, t))
        return intersections

    def _check_direction_consistency(
        self,
        source_point: Point,
        target_point: Point,
        source_id: str,
        target_id: str,
        candidates: List[CandidateNode],
        ndx: float,
        ndy: float,
    ) -> bool:
        """Check that source→target direction is consistent with arrow direction."""
        source_cand = next((c for c in candidates if c.node_id == source_id), None)
        target_cand = next((c for c in candidates if c.node_id == target_id), None)
        if not source_cand or not target_cand:
            return False

        src_center = source_cand.position
        tgt_center = target_cand.position
        dx = tgt_center.x - src_center.x
        dy = tgt_center.y - src_center.y
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return False

        dot = (dx/length) * ndx + (dy/length) * ndy
        return dot > 0.2  # Source→Target should roughly agree with arrow direction

    def _compute_evidence(
        self,
        arrow: DetectedArrow,
        source_point: Point,
        target_point: Point,
        best_source: Tuple[str, float, Dict[str, float]],
        best_target: Tuple[str, float, Dict[str, float]],
        candidates: List[CandidateNode],
        ndx: float,
        ndy: float,
    ) -> DependencyEvidence:
        """Compute full evidence for a dependency candidate."""
        evidence = DependencyEvidence()
        evidence.arrowhead_confidence = arrow.arrowhead_confidence
        evidence.source_boundary_intersection = best_source[2].get("boundary_intersection", 0)
        evidence.target_boundary_intersection = best_target[2].get("boundary_intersection", 0)
        evidence.direction_consistency = best_source[2].get("direction_alignment", 0)
        evidence.distance_from_source_boundary = best_source[2].get("distance_score", 0)
        evidence.distance_from_target_boundary = best_target[2].get("distance_score", 0)
        evidence.angular_consistency = best_source[2].get("angular_consistency", 0)
        evidence.raw_evidence = {
            "source": best_source[2],
            "target": best_target[2],
        }
        return evidence

    def _compute_total_score(self, evidence: DependencyEvidence) -> float:
        """Compute total confidence score from evidence."""
        score = (
            0.20 * evidence.arrowhead_confidence
            + 0.25 * evidence.source_boundary_intersection
            + 0.25 * evidence.target_boundary_intersection
            + 0.15 * evidence.direction_consistency
            + 0.10 * evidence.angular_consistency
            - 0.15 * evidence.node_interior_crossing_penalty
        )
        return max(0.0, min(1.0, score))

    # =========================================================================
    # Classification
    # =========================================================================

    def _classify_candidates(
        self,
        candidates: List[ValidatedDependency],
        report: DependencyValidationReport,
    ) -> None:
        """Classify candidates into HIGH/MEDIUM/LOW/REJECTED."""
        for dep in candidates:
            # Rejected if has rejection reasons
            if dep.rejection_reasons:
                dep.confidence_level = DependencyConfidence.REJECTED
                dep.accepted = False
                report.rejected_candidates.append(dep)
                report.rejected_count += 1
                continue

            # If resolver flagged as review-required, honor that
            if dep.review_required:
                dep.confidence_level = DependencyConfidence.MEDIUM
                dep.accepted = False
                report.review_candidates.append(dep)
                report.review_count += 1
                continue

            # Classify by score
            if dep.confidence_score >= self.high_threshold:
                dep.confidence_level = DependencyConfidence.HIGH
                dep.accepted = True
                report.validated_dependencies.append(dep)
                report.accepted_count += 1
            elif dep.confidence_score >= self.medium_threshold:
                dep.confidence_level = DependencyConfidence.MEDIUM
                dep.review_required = True
                report.review_candidates.append(dep)
                report.review_count += 1
            else:
                dep.confidence_level = DependencyConfidence.LOW
                dep.accepted = False
                report.rejected_candidates.append(dep)
                report.rejected_count += 1

    def _deduplicate_edges(self, report: DependencyValidationReport) -> None:
        """Remove duplicate source→target edges."""
        seen_edges: Set[Tuple[str, str]] = set()
        unique_deps: List[ValidatedDependency] = []

        for dep in report.validated_dependencies:
            edge = (dep.source_id, dep.target_id)
            if edge not in seen_edges:
                seen_edges.add(edge)
                unique_deps.append(dep)
            else:
                report.duplicate_edge_count += 1

        report.validated_dependencies = unique_deps
        report.accepted_count = len(unique_deps)

    # =========================================================================
    # Debug Output
    # =========================================================================

    def _build_debug_entries(
        self,
        candidates: List[ValidatedDependency],
        report: DependencyValidationReport,
    ) -> None:
        """Build machine-readable debug entries."""
        for dep in candidates:
            entry = {
                "arrow_id": dep.arrow_id,
                "source_node": dep.source_id,
                "target_node": dep.target_id,
                "accepted": dep.accepted,
                "confidence": dep.confidence_score,
                "confidence_level": dep.confidence_level.value,
                "reasons": dep.rejection_reasons,
                "evidence": {
                    "arrowhead_confidence": dep.evidence.arrowhead_confidence,
                    "source_boundary": dep.evidence.source_boundary_intersection,
                    "target_boundary": dep.evidence.target_boundary_intersection,
                    "direction": dep.evidence.direction_consistency,
                    "angular": dep.evidence.angular_consistency,
                    "shaft_visual_support": dep.evidence.shaft_visual_support,
                    "visual_route_support": dep.evidence.visual_route_support,
                },
            }
            gates = self.evaluate_promotion_gates(dep)
            if gates is not None:
                entry["promotion_gates"] = gates
            report.debug_entries.append(entry)

        report.gate_digest = self.compute_gate_digest(candidates)

    @classmethod
    def evaluate_promotion_gates(
        cls, dep: ValidatedDependency
    ) -> Dict[str, Any]:
        """Return per-gate pass/fail values for one candidate.

        Pure diagnostics: the same six values the promotion conjunction in
        ``_can_auto_accept_visual_review`` inspects, compared against the
        same class constants. Candidates built by this builder always
        carry computed evidence, so this always evaluates (a paper-zero
        evidence object simply fails every gate).
        """
        evidence = dep.evidence
        crossing = float(dep.metadata.get("crossing_penalty", 0.0) or 0.0)
        gate_values = {
            "crossing_penalty": crossing,
            "confidence": float(dep.confidence_score),
            "visual_route_support": float(evidence.visual_route_support),
            "arrowhead": float(evidence.arrowhead_confidence),
            "boundary_min": float(min(
                evidence.source_boundary_intersection or 0.0,
                evidence.target_boundary_intersection or 0.0,
            )),
            "direction": float(evidence.direction_consistency),
            "angular": float(evidence.angular_consistency),
        }
        thresholds = {
            "crossing_penalty": ("<", cls.PROMOTION_MAX_CROSSING_PENALTY),
            "confidence": (">=", cls.PROMOTION_MIN_CONFIDENCE),
            "visual_route_support": (">=", cls.PROMOTION_MIN_VISUAL_ROUTE_SUPPORT),
            "arrowhead": (">=", cls.PROMOTION_MIN_ARROWHEAD),
            "boundary_min": (">=", cls.PROMOTION_MIN_BOUNDARY_INTERSECTION),
            "direction": (">=", cls.PROMOTION_MIN_DIRECTION_CONSISTENCY),
            "angular": (">=", cls.PROMOTION_MIN_ANGULAR_CONSISTENCY),
        }
        passed: Dict[str, bool] = {}
        for name, value in gate_values.items():
            op, threshold = thresholds[name]
            passed[name] = (value < threshold) if op == "<" else (value >= threshold)
        return {
            "values": gate_values,
            "passed": passed,
            "all_passed": all(passed.values()),
        }

    @classmethod
    def compute_gate_digest(
        cls,
        candidates: List[ValidatedDependency],
    ) -> Dict[str, Any]:
        """Aggregate per-gate pass counts and the first-failing-gate histogram."""
        gates = [
            cls.evaluate_promotion_gates(dep)
            for dep in candidates
        ]
        gates = [g for g in gates if g is not None]
        digest: Dict[str, Any] = {
            "candidates_evaluated": len(gates),
            "all_gates_passed": sum(1 for g in gates if g["all_passed"]),
            "per_gate": {},
            "first_failing_gate": {},
        }
        order = ["crossing_penalty", "confidence", "visual_route_support",
                 "arrowhead", "boundary_min", "direction", "angular"]
        for name in order:
            digest["per_gate"][name] = {
                "passed": sum(1 for g in gates if g["passed"][name]),
                "failed": sum(1 for g in gates if not g["passed"][name]),
            }
        for g in gates:
            if g["all_passed"]:
                continue
            blocker = next(
                (name for name in order if not g["passed"][name]),
                "unknown",
            )
            digest["first_failing_gate"][blocker] = (
                digest["first_failing_gate"].get(blocker, 0) + 1
            )
        return digest
