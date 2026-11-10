"""
Semantic diagram reconstruction engine.

Bridges CV/OCR detection evidence with the semantic GraphModel.
Consumes outputs from Shape Detection, Arrow Detection, and OCR/Text
Association — never re-runs CV operations.

Produces reconstructed diagram elements with confidence scores,
evidence provenance, and ambiguity tracking. Supports both AON and
AOA representations natively.

Usage:
    engine = ReconstructionEngine()
    diagram = engine.reconstruct_aon(shape_result, arrow_result, ocr_result, association_result)
    graph_model = diagram.to_graph_model()
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from pert_analyzer.core.models import DiagramType, GraphModel
from pert_analyzer.cv.exceptions import ReconstructionError
from pert_analyzer.cv.models import (
    ArrowDetectionResult,
    CandidateNode,
    DiagramClassificationResult,
    DetectedArrow,
    ShapeDetectionResult,
    ShapeType,
)
from pert_analyzer.cv.ocr_models import (
    AssociationTargetType,
    OCRProcessingResult,
    OCRTextRegion,
    TextAssociation,
    TextAssociationResult,
    TextType,
)
from pert_analyzer.cv.reconstruction_models import (
    ActivityStatus,
    AmbiguityIssue,
    AmbiguityType,
    EvidenceTrace,
    IDSource,
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
    ReconstructedEvent,
    ValidationResult,
    ValidationSeverity,
)
from pert_analyzer.graph.builder import GraphBuilder

logger = logging.getLogger(__name__)


class ReconstructionEngine:
    """
    Reconstructs semantic diagram structure from CV/OCR evidence.

    This engine is the bridge between low-level visual detection and
    high-level graph modeling. It:
    - Consumes detection results (Phases 4, 5, 6)
    - Never calls CV operations directly
    - Produces semantic candidates with confidence and ambiguity
    - Supports both AON and AOA representations
    """

    def __init__(self, confidence_threshold: float = 0.1):
        """
        Initialize the reconstruction engine.

        Args:
            confidence_threshold: Minimum confidence for including
                a reconstructed element.
        """
        self.confidence_threshold = confidence_threshold

    # =========================================================================
    # Public API
    # =========================================================================

    def reconstruct_aon(
        self,
        shape_result: ShapeDetectionResult,
        arrow_result: Optional[ArrowDetectionResult] = None,
        ocr_result: Optional[OCRProcessingResult] = None,
        association_result: Optional[TextAssociationResult] = None,
        region_ocr_results: Optional[List[Any]] = None,
    ) -> ReconstructedDiagram:
        """
        Reconstruct a diagram in Activity-on-Node representation.

        AON mapping:
        - Rectangles/Squares → Activity nodes
        - Circles/Ellipses → Event nodes (if present)
        - Arrows → Dependencies between nodes
        - Text inside shapes → Activity ID, label, duration
        - Text near arrows → Dependency labels (usually ignored)

        Args:
            shape_result: Results from shape detection.
            arrow_result: Results from arrow detection.
            ocr_result: Results from OCR text extraction.
            association_result: Results from text-to-shape association.

        Returns:
            ReconstructedDiagram with all semantic elements.
        """
        diagram = ReconstructedDiagram(diagram_type="AON")
        text_map = self._build_text_region_map(ocr_result)
        shape_map = {s.shape_id: s for s in shape_result.detected_shapes}
        candidate_map = {c.node_id: c for c in shape_result.candidate_nodes}
        arrow_map = {a.arrow_id: a for a in (arrow_result.arrows if arrow_result else [])}

        # 1. Reconstruct activities from rectangles
        self._reconstruct_aon_activities(
            diagram, shape_result, candidate_map, text_map, association_result, region_ocr_results
        )

        # 2. Reconstruct events from circles (optional in AON)
        self._reconstruct_aon_events(
            diagram, shape_result, candidate_map, text_map
        )

        # 3. Reconstruct dependencies from arrows
        if arrow_result:
            self._reconstruct_aon_dependencies(
                diagram, arrow_result, arrow_map, candidate_map
            )

        # 3.5. Disambiguate duplicate activity IDs
        self._disambiguate_duplicate_ids(diagram)

        # 3.6. Graph-consistency validation and topology-based correction
        from pert_analyzer.cv.topology_validation import TopologyValidator
        from pert_analyzer.cv.semantic_resolution import SemanticResolver
        topology_validator = TopologyValidator()
        topology_report = topology_validator.validate(diagram)
        diagram.metadata["topology_report"] = topology_report

        # Apply graph-consistency correction to ambiguous IDs
        resolution_result = diagram.metadata.get("semantic_resolution")
        if resolution_result:
            resolver = SemanticResolver()
            resolver.resolve_with_topology(
                resolution_result, diagram.activities, diagram.dependencies
            )
            # Update activities with topology-corrected IDs
            for cr in resolution_result.id_resolutions.values():
                if cr.resolved_id and cr.source.value == "graph_topology":
                    for act in diagram.activities:
                        if act.source_node_id == cr.node_id:
                            # Preserve raw in label for traceability
                            if act.label != cr.resolved_id:
                                act.metadata["raw_id_before_topology"] = act.activity_id
                                act.activity_id = cr.resolved_id
                                act.label = cr.resolved_id
                                act.id_source = IDSource.REGION_OCR
                                act.status = ActivityStatus.CONFIRMED
                                act.evidence.append(EvidenceTrace(
                                    source_phase="topology_correction",
                                    source_ids=[cr.node_id],
                                    confidence_contribution=cr.confidence,
                                    description=f"Topology correction: raw='{cr.raw_id}' → '{cr.resolved_id}'",
                                    metadata={"resolution_source": cr.source.value},
                                ))
                            break

        # 3.7. Finalize dependency endpoints against the FINAL activity IDs.
        #      Topology correction (3.6) may rename activities after the
        #      dependency list was built, leaving stale endpoints (self-loops
        #      and wrong edges). Rebuild deps from the geometric report so
        #      every endpoint maps to the current canonical ID.
        self._finalize_aon_dependencies(diagram)

        # 4. Detect ambiguities
        self._detect_ambiguities(diagram)

        # 5. Validate
        diagram.validation = self.validate(diagram)

        # 6. Compute overall confidence
        diagram.overall_confidence = self._compute_overall_confidence(diagram)

        logger.info(
            "AON reconstruction: %d activities, %d events, %d deps, "
            "%d ambiguities, confidence=%.3f",
            diagram.activity_count,
            diagram.event_count,
            diagram.dependency_count,
            diagram.ambiguity_count,
            diagram.overall_confidence,
        )

        return diagram

    def reconcile(
        self,
        diagram: ReconstructedDiagram,
        reference_activities: Optional[Dict[str, Any]] = None,
    ) -> "ReconciliationResult":
        """
        Run graph-based reconciliation on a reconstructed diagram.

        This is the bridge between Phase 7 (reconstruction) and
        Phase 9 (CPM). It:
        - Analyzes graph connectivity
        - Identifies START/FINISH
        - Filters false positives
        - Reconciles IDs and durations
        - Verifies graph consistency
        - Runs CPM if possible

        Args:
            diagram: Reconstructed diagram from Phase 7.
            reference_activities: Optional reference for evaluation only.

        Returns:
            ReconciliationResult with full diagnostic info.
        """
        from pert_analyzer.cv.reconciliation import ReconciliationEngine

        reconciler = ReconciliationEngine()
        return reconciler.reconcile(diagram, reference_activities)

    def reconstruct_aoa(
        self,
        shape_result: ShapeDetectionResult,
        arrow_result: Optional[ArrowDetectionResult] = None,
        ocr_result: Optional[OCRProcessingResult] = None,
        association_result: Optional[TextAssociationResult] = None,
    ) -> ReconstructedDiagram:
        """
        Reconstruct a diagram in Activity-on-Arrow representation.

        AOA mapping:
        - Circles/Ellipses → Event nodes
        - Arrows → Activities (with duration)
        - Rectangles (if any) → Labels or annotations
        - Text inside circles → Event ID/label
        - Text near arrows → Activity ID, label, duration

        Args:
            shape_result: Results from shape detection.
            arrow_result: Results from arrow detection.
            ocr_result: Results from OCR text extraction.
            association_result: Results from text-to-shape association.

        Returns:
            ReconstructedDiagram with all semantic elements.
        """
        diagram = ReconstructedDiagram(diagram_type="AOA")
        text_map = self._build_text_region_map(ocr_result)
        candidate_map = {c.node_id: c for c in shape_result.candidate_nodes}
        arrow_map = {a.arrow_id: a for a in (arrow_result.arrows if arrow_result else [])}

        # 1. Reconstruct events from circles
        self._reconstruct_aoa_events(
            diagram, shape_result, candidate_map, text_map
        )

        # 2. Reconstruct activities from arrows
        if arrow_result:
            self._reconstruct_aoa_activities(
                diagram, arrow_result, arrow_map, candidate_map,
                text_map, association_result
            )

        # 2.5. Disambiguate duplicate activity IDs (multiple arrows can
        #      resolve to the same OCR label, e.g. several "A"s). Duplicate
        #      IDs abort graph construction, so canonicalize before building
        #      dependencies.
        self._disambiguate_duplicate_ids(diagram)

        # 3. Reconstruct dependencies (AOA: activities chained through events)
        if arrow_result:
            self._reconstruct_aoa_dependencies(
                diagram, arrow_result, arrow_map
            )

        # 4. Detect ambiguities
        self._detect_ambiguities(diagram)

        # 5. Validate
        diagram.validation = self.validate(diagram)

        # 6. Compute overall confidence
        diagram.overall_confidence = self._compute_overall_confidence(diagram)

        logger.info(
            "AOA reconstruction: %d activities, %d events, %d deps, "
            "%d ambiguities, confidence=%.3f",
            diagram.activity_count,
            diagram.event_count,
            diagram.dependency_count,
            diagram.ambiguity_count,
            diagram.overall_confidence,
        )

        return diagram

    # =========================================================================
    # Validation
    # =========================================================================

    def validate(self, diagram: ReconstructedDiagram) -> ValidationResult:
        """
        Validate a reconstructed diagram.

        Checks for:
        - Missing durations on non-dummy activities
        - Isolated nodes (no dependencies)
        - Orphaned arrows (not mapped to dependencies)
        - Duplicate activity IDs
        - Duplicate event IDs
        - Empty diagram

        Args:
            diagram: The reconstructed diagram to validate.

        Returns:
            ValidationResult with errors, warnings, and info.
        """
        result = ValidationResult()

        # Empty diagram check
        if diagram.activity_count == 0 and diagram.event_count == 0:
            result.add_issue(AmbiguityIssue(
                ambiguity_type=AmbiguityType.MISSING_DURATION,
                description="Diagram has no activities or events",
                severity=ValidationSeverity.ERROR,
            ))
            return result

        # Duplicate activity IDs
        seen_act_ids: Dict[str, int] = {}
        for act in diagram.activities:
            count = seen_act_ids.get(act.activity_id, 0) + 1
            seen_act_ids[act.activity_id] = count
        for aid, count in seen_act_ids.items():
            if count > 1:
                result.add_issue(AmbiguityIssue(
                    ambiguity_type=AmbiguityType.DUPLICATE_LABEL,
                    description=f"Duplicate activity ID '{aid}' ({count} occurrences)",
                    involved_ids=[aid],
                    severity=ValidationSeverity.ERROR,
                ))

        # Duplicate event IDs
        seen_evt_ids: Dict[str, int] = {}
        for evt in diagram.events:
            count = seen_evt_ids.get(evt.event_id, 0) + 1
            seen_evt_ids[evt.event_id] = count
        for eid, count in seen_evt_ids.items():
            if count > 1:
                result.add_issue(AmbiguityIssue(
                    ambiguity_type=AmbiguityType.DUPLICATE_LABEL,
                    description=f"Duplicate event ID '{eid}' ({count} occurrences)",
                    involved_ids=[eid],
                    severity=ValidationSeverity.ERROR,
                ))

        # Missing durations on non-dummy activities
        for act in diagram.activities:
            if not act.is_dummy and act.duration <= 0:
                result.add_issue(AmbiguityIssue(
                    ambiguity_type=AmbiguityType.MISSING_DURATION,
                    description=f"Activity '{act.activity_id}' has no duration",
                    involved_ids=[act.activity_id],
                    severity=ValidationSeverity.WARNING,
                ))

        # Isolated nodes (activities with no dependencies)
        all_element_ids = set()
        for act in diagram.activities:
            all_element_ids.add(act.activity_id)
        for evt in diagram.events:
            all_element_ids.add(evt.event_id)

        connected_ids = set()
        for dep in diagram.dependencies:
            connected_ids.add(dep.source_id)
            connected_ids.add(dep.target_id)

        isolated = all_element_ids - connected_ids
        for node_id in isolated:
            result.add_issue(AmbiguityIssue(
                ambiguity_type=AmbiguityType.ISOLATED_NODE,
                description=f"Node '{node_id}' has no dependencies",
                involved_ids=[node_id],
                severity=ValidationSeverity.WARNING,
            ))

        # Low confidence warnings
        for act in diagram.activities:
            if act.confidence < 0.3:
                result.add_issue(AmbiguityIssue(
                    ambiguity_type=AmbiguityType.SHAPE_ROLE_UNKNOWN,
                    description=(
                        f"Activity '{act.activity_id}' has low confidence "
                        f"({act.confidence:.3f})"
                    ),
                    involved_ids=[act.activity_id],
                    severity=ValidationSeverity.INFO,
                ))

        logger.debug(
            "Validation: valid=%s, errors=%d, warnings=%d, info=%d",
            result.is_valid,
            result.error_count,
            result.warning_count,
            len(result.info),
        )

        return result

    # =========================================================================
    # Private: AON Reconstruction
    # =========================================================================

    def _reconstruct_aon_activities(
        self,
        diagram: ReconstructedDiagram,
        shape_result: ShapeDetectionResult,
        candidate_map: Dict[str, CandidateNode],
        text_map: Dict[str, OCRTextRegion],
        association_result: Optional[TextAssociationResult],
        region_ocr_results: Optional[List[Any]] = None,
    ) -> None:
        """Reconstruct activities from rectangular shapes in AON.

        Every valid rectangle produces exactly one activity candidate,
        even when OCR fails. Deterministic INFERRED IDs are assigned
        using spatial ordering (top-to-bottom, left-to-right).
        """
        # Collect rectangle candidates and sort spatially
        rectangle_candidates = [
            c for c in shape_result.candidate_nodes
            if c.shape_type in (ShapeType.RECTANGLE, ShapeType.SQUARE)
        ]
        rectangle_candidates.sort(
            key=lambda c: (round(c.position.y / 40.0), c.position.x)
        )

        # Build region_ocr lookup by node_id (rectangles only — polygons are arrow labels)
        region_ocr_map: Dict[str, Any] = {}
        if region_ocr_results:
            rectangle_ids = {c.node_id for c in rectangle_candidates}
            for nr in region_ocr_results:
                if nr.node_id in rectangle_ids:
                    region_ocr_map[nr.node_id] = nr

        # Run semantic resolution on node-local OCR results
        from pert_analyzer.cv.semantic_resolution import SemanticResolver
        resolver = SemanticResolver()
        rectangle_ocr_results = [region_ocr_map[c.node_id] for c in rectangle_candidates if c.node_id in region_ocr_map]
        resolution_result = resolver.resolve_all(rectangle_ocr_results)
        diagram.metadata["semantic_resolution"] = resolution_result

        # Apply resolved IDs and durations back to NodeOCRResult objects
        for cr in resolution_result.id_resolutions.values():
            if cr.node_id in region_ocr_map:
                nr = region_ocr_map[cr.node_id]
                if cr.resolved_id and cr.resolved_id != nr.best_activity_id:
                    # Preserve raw in alternatives for traceability
                    if nr.best_activity_id:
                        nr.activity_id_alternatives.insert(0, (nr.best_activity_id, nr.best_activity_id_confidence))
                    nr.best_activity_id = cr.resolved_id
                    nr.best_activity_id_confidence = cr.confidence
                    logger.debug(
                        "Resolved ID: node=%s raw=%s → resolved=%s (source=%s)",
                        cr.node_id, cr.raw_id, cr.resolved_id, cr.source.value,
                    )

        for dr in resolution_result.duration_resolutions.values():
            if dr.node_id in region_ocr_map and dr.resolved_value is not None:
                nr = region_ocr_map[dr.node_id]
                if dr.resolved_value != (nr.best_numeric[0] if nr.best_numeric else None):
                    # Preserve raw numeric in alternatives
                    if nr.best_numeric:
                        nr.numeric_candidates.insert(0, nr.best_numeric)
                    nr.best_numeric = (dr.resolved_value, dr.raw_text or str(int(dr.resolved_value)), dr.confidence)
                    logger.debug(
                        "Resolved duration: node=%s raw=%s → resolved=%s (source=%s)",
                        dr.node_id, dr.raw_value, dr.resolved_value, dr.source.value,
                    )

        inferred_counter = 0
        for candidate in rectangle_candidates:
            label = ""
            duration = 0.0
            text_region_ids: List[str] = []
            evidence_traces: List[EvidenceTrace] = []
            id_source = IDSource.SPATIAL_INFERENCE
            status = ActivityStatus.INFERRED

            # Shape evidence
            evidence_traces.append(EvidenceTrace(
                source_phase="shape_detection",
                source_ids=[candidate.source_shape_id],
                confidence_contribution=candidate.confidence,
                description=f"Rectangle shape detected (confidence={candidate.confidence:.3f})",
            ))

            # PRIORITY 1: Region OCR (node-local, authoritative for AON)
            if candidate.node_id in region_ocr_map:
                nr = region_ocr_map[candidate.node_id]

                # ID from region OCR sub-crop
                if nr.best_activity_id:
                    label = nr.best_activity_id
                    id_source = IDSource.REGION_OCR
                    status = ActivityStatus.CONFIRMED
                    evidence_traces.append(EvidenceTrace(
                        source_phase="region_ocr_id",
                        source_ids=[candidate.node_id],
                        confidence_contribution=nr.best_activity_id_confidence,
                        description=f"Region OCR ID: '{nr.best_activity_id}' (conf={nr.best_activity_id_confidence:.3f})",
                    ))

                # Duration from region OCR sub-crop
                if nr.best_numeric:
                    duration = nr.best_numeric[0]
                    evidence_traces.append(EvidenceTrace(
                        source_phase="region_ocr_duration",
                        source_ids=[candidate.node_id],
                        confidence_contribution=nr.best_numeric[2],
                        description=f"Region OCR duration: {nr.best_numeric[0]} (conf={nr.best_numeric[2]:.3f})",
                    ))

                # Semantic resolution evidence (if resolver changed the ID or duration)
                if candidate.node_id in resolution_result.id_resolutions:
                    cr = resolution_result.id_resolutions[candidate.node_id]
                    if cr.source.value not in ("ocr_high_confidence",):
                        evidence_traces.append(EvidenceTrace(
                            source_phase="semantic_resolution",
                            source_ids=[candidate.node_id],
                            confidence_contribution=cr.confidence,
                            description=(
                                f"Semantic resolution: raw='{cr.raw_id}' → resolved='{cr.resolved_id}' "
                                f"(source={cr.source.value}, status={cr.status.value})"
                            ),
                            metadata={"resolution_source": cr.source.value},
                        ))
                        # Update source/status based on resolution
                        if cr.source.value == "contextual_resolution":
                            id_source = IDSource.REGION_OCR
                            status = ActivityStatus.CONFIRMED
                        elif cr.source.value == "ocr_alternative":
                            id_source = IDSource.REGION_OCR
                            status = ActivityStatus.CONFIRMED

            # PRIORITY 2: Spatial association (full-image OCR → shape mapping)
            # Only use if region OCR did NOT provide an ID
            if not label and association_result:
                _, assoc_label, assoc_duration, assoc_evidence = (
                    self._find_text_for_shape(
                        candidate, association_result, text_map
                    )
                )
                if assoc_label:
                    # Validate: full-image OCR text must be plausible as activity ID
                    plausible = self._is_plausible_activity_id(assoc_label)
                    if plausible:
                        label = assoc_label
                        if id_source == IDSource.SPATIAL_INFERENCE:
                            id_source = IDSource.OCR_HIGH_CONFIDENCE
                        evidence_traces.extend(assoc_evidence)
                    else:
                        evidence_traces.append(EvidenceTrace(
                            source_phase="full_image_ocr_rejected",
                            source_ids=[],
                            confidence_contribution=0.0,
                            description=f"Full-image OCR '{assoc_label}' rejected (not plausible activity ID)",
                        ))
                if assoc_duration and duration <= 0:
                    duration = assoc_duration
                    evidence_traces.extend(assoc_evidence)

            # 3. Try to extract activity ID from label
            extracted_id = None
            if label:
                extracted_id = self._extract_activity_id(label)
                if extracted_id:
                    if id_source == IDSource.SPATIAL_INFERENCE:
                        id_source = IDSource.OCR_HIGH_CONFIDENCE
                    status = ActivityStatus.CONFIRMED
                    label = label

            # 4. Deterministic INFERRED_XXX ID
            inferred_counter += 1
            if extracted_id:
                act_id = extracted_id
            else:
                act_id = f"INFERRED_{inferred_counter:03d}"

            # 5. Determine status based on what we have
            if extracted_id:
                if duration > 0:
                    status = ActivityStatus.CONFIRMED
                else:
                    status = ActivityStatus.REVIEW_REQUIRED
                    id_source = IDSource.OCR_HIGH_CONFIDENCE
            else:
                # No OCR ID at all
                status = ActivityStatus.INFERRED
                id_source = IDSource.SPATIAL_INFERENCE

            # 6. Build review issues for missing data
            warnings: List[str] = []
            if duration <= 0:
                warnings.append("No duration detected")
            if not extracted_id:
                warnings.append("Activity ID inferred from spatial position (no OCR)")
            if status == ActivityStatus.REVIEW_REQUIRED:
                warnings.append("Requires review: incomplete OCR data")

            act = ReconstructedActivity(
                activity_id=act_id,
                label=label or act_id,
                duration=duration,
                confidence=self._aggregate_confidence([
                    candidate.confidence,
                    *[e.confidence_contribution for e in evidence_traces],
                ]),
                evidence=evidence_traces,
                source_shape_id=candidate.source_shape_id,
                source_node_id=candidate.node_id,
                source_text_region_ids=text_region_ids,
                position=(candidate.position.x, candidate.position.y),
                bounding_box=candidate.bounding_box,
                warnings=warnings,
                status=status,
                id_source=id_source,
                needs_review=(status != ActivityStatus.CONFIRMED),
                # Dual identity: geometric ID is always stable; semantic ID
                # may be None when OCR fails.
                geometric_node_id=candidate.node_id,
                semantic_activity_id=(
                    extracted_id if extracted_id else None
                ),
                semantic_status=(
                    ActivityStatus.CONFIRMED
                    if extracted_id
                    else ActivityStatus.REVIEW_REQUIRED
                ),
            )

            diagram.activities.append(act)

    def _reconstruct_aon_events(
        self,
        diagram: ReconstructedDiagram,
        shape_result: ShapeDetectionResult,
        candidate_map: Dict[str, CandidateNode],
        text_map: Dict[str, OCRTextRegion],
    ) -> None:
        """Reconstruct events from circular shapes in AON (optional)."""
        event_counter = 0
        for candidate in shape_result.candidate_nodes:
            if candidate.shape_type not in (
                ShapeType.CIRCLE, ShapeType.ELLIPSE
            ):
                continue

            event_counter += 1
            evt_id = f"E{event_counter}"
            label = ""

            # Check if text was associated with this shape
            for region in text_map.values():
                if candidate.bounding_box.contains_point(
                    region.bounding_box.center
                ):
                    label = region.raw_text
                    break

            evt = ReconstructedEvent(
                event_id=evt_id,
                label=label or evt_id,
                confidence=candidate.confidence,
                evidence=[EvidenceTrace(
                    source_phase="shape_detection",
                    source_ids=[candidate.source_shape_id],
                    confidence_contribution=candidate.confidence,
                    description=f"Circle shape detected (confidence={candidate.confidence:.3f})",
                )],
                source_shape_id=candidate.source_shape_id,
                position=(candidate.position.x, candidate.position.y),
                bounding_box=candidate.bounding_box,
            )

            diagram.events.append(evt)

    def _reconstruct_aon_dependencies(
        self,
        diagram: ReconstructedDiagram,
        arrow_result: ArrowDetectionResult,
        arrow_map: Dict[str, DetectedArrow],
        candidate_map: Dict[str, CandidateNode],
    ) -> None:
        """Reconstruct dependencies from detected arrows in AON using strict geometry.

        Uses ValidatedDependencyBuilder for strict geometric validation,
        arrow deduplication, and confidence-based classification.
        """
        from pert_analyzer.cv.validated_dependency import ValidatedDependencyBuilder

        # Build node_id -> activity_id mapping
        node_to_activity: Dict[str, str] = {}
        for act in diagram.activities:
            if act.source_node_id:
                node_to_activity[act.source_node_id] = act.activity_id

        # Filter to rectangle candidates only
        rectangle_candidates = [
            c for c in candidate_map.values()
            if c.shape_type in (ShapeType.RECTANGLE, ShapeType.SQUARE)
        ]

        # Use strict validated dependency builder
        builder = ValidatedDependencyBuilder(
            high_threshold=0.55,
            medium_threshold=0.35,
            min_boundary_distance=80.0,
            max_direction_tolerance_deg=40.0,
            require_boundary_intersection=True,
        )

        validation_report = builder.build_validated_dependencies(
            arrow_result.arrows, rectangle_candidates
        )

        # Store validation report in diagram metadata
        diagram.metadata["dependency_validation"] = validation_report

        # Build dependencies from accepted (HIGH confidence) validations
        for vdep in validation_report.validated_dependencies:
            source_id = node_to_activity.get(vdep.source_id, "")
            target_id = node_to_activity.get(vdep.target_id, "")

            if not source_id or not target_id:
                continue

            evidence_traces = [EvidenceTrace(
                source_phase="validated_dependency",
                source_ids=[vdep.arrow_id],
                confidence_contribution=vdep.confidence_score,
                description=(
                    f"Validated arrow: {source_id}→{target_id} "
                    f"(score={vdep.confidence_score:.3f}, "
                    f"level={vdep.confidence_level.value})"
                ),
                metadata={
                    "source_boundary": vdep.evidence.source_boundary_intersection,
                    "target_boundary": vdep.evidence.target_boundary_intersection,
                    "direction": vdep.evidence.direction_consistency,
                    "angular": vdep.evidence.angular_consistency,
                },
            )]

            dep = ReconstructedDependency(
                source_id=source_id,
                target_id=target_id,
                confidence=vdep.confidence_score,
                evidence=evidence_traces,
                source_arrow_id=vdep.arrow_id,
                metadata={
                    **vdep.metadata,
                    **(vdep.evidence.raw_evidence if hasattr(vdep.evidence, 'raw_evidence') else {}),
                    "confidence_level": vdep.confidence_level.value,
                    "validation_score": vdep.confidence_score,
                    "source_boundary": vdep.evidence.source_boundary_intersection,
                    "target_boundary": vdep.evidence.target_boundary_intersection,
                },
            )
            diagram.dependencies.append(dep)

        # Log review candidates as ambiguities
        for vdep in validation_report.review_candidates:
            source_id = node_to_activity.get(vdep.source_id, vdep.source_id)
            target_id = node_to_activity.get(vdep.target_id, vdep.target_id)
            diagram.ambiguities.append(AmbiguityIssue(
                ambiguity_type=AmbiguityType.ARROW_AMBIGUOUS,
                description=(
                    f"Review candidate: {source_id}→{target_id} "
                    f"(score={vdep.confidence_score:.3f})"
                ),
                involved_ids=[vdep.arrow_id],
                severity=ValidationSeverity.INFO,
            ))

        # Log rejected candidates
        for vdep in validation_report.rejected_candidates:
            reasons = ", ".join(vdep.rejection_reasons) if vdep.rejection_reasons else "low_score"
            diagram.ambiguities.append(AmbiguityIssue(
                ambiguity_type=AmbiguityType.ARROW_NO_SOURCE,
                description=f"Rejected arrow: {reasons}",
                involved_ids=[vdep.arrow_id],
                severity=ValidationSeverity.INFO,
            ))

        logger.info(
            "Validated dependencies: %d accepted, %d review, %d rejected (from %d raw arrows)",
            validation_report.accepted_count,
            validation_report.review_count,
            validation_report.rejected_count,
            validation_report.raw_arrow_count,
        )

    def _finalize_aon_dependencies(self, diagram: ReconstructedDiagram) -> None:
        """Rebuild AON dependencies against the FINAL activity IDs.

        Dependency endpoints are re-mapped from the strictly-validated
        geometric report (node IDs) to the current canonical activity IDs.
        This undoes stale labels left behind by post-build steps:

        - Dependencies built before ``_disambiguate_duplicate_ids`` /
          topology correction reference pre-canonical IDs; both endpoints
          of a single arrow can collapse to the same ID (a self-loop) or
          resolve to the wrong activities.
        - Contradictory accepted edges (arrow A→B and arrow B→A between the
          same two geometric nodes) cannot both be true; both are routed to
          human review instead of being auto-accepted.

        The GraphModel / graph validation are untouched: this only re-anchors
        detection evidence to its final semantic labels.
        """
        metadata = diagram.metadata or {}
        report = metadata.get("dependency_validation")
        if report is None:
            return

        node_to_activity: Dict[str, str] = {}
        for act in diagram.activities:
            if act.source_node_id:
                node_to_activity[act.source_node_id] = act.activity_id

        accepted = list(getattr(report, "validated_dependencies", []) or [])

        # Detect reciprocal conflicts: both the (s,t) and (t,s) directions
        # are accepted between the same two geometric nodes.
        directed: Dict[Tuple[str, str], Any] = {}
        for vdep in accepted:
            directed[(vdep.source_id, vdep.target_id)] = vdep

        conflicted: List[Any] = []
        clean: List[Any] = []
        for vdep in accepted:
            reverse = (vdep.target_id, vdep.source_id)
            if reverse in directed and reverse != (vdep.source_id, vdep.target_id):
                conflicted.append(vdep)
            else:
                clean.append(vdep)

        # Route conflicted evidence to human review (never silently drop).
        review_candidates = list(getattr(report, "review_candidates", []) or [])
        seen_review = {id(v) for v in review_candidates}
        for vdep in conflicted:
            if id(vdep) in seen_review:
                continue
            seen_review.add(id(vdep))
            review_candidates.append(vdep)
            source_label = node_to_activity.get(vdep.source_id, vdep.source_id)
            target_label = node_to_activity.get(vdep.target_id, vdep.target_id)
            diagram.ambiguities.append(AmbiguityIssue(
                ambiguity_type=AmbiguityType.ARROW_AMBIGUOUS,
                description=(
                    f"Conflicting arrows brace both {source_label}→{target_label} "
                    f"and {target_label}→{source_label}; requires review"
                ),
                involved_ids=[getattr(vdep, "arrow_id", "")],
                severity=ValidationSeverity.INFO,
            ))
        report.review_candidates = review_candidates

        # Rebuild the dependency list from the clean accepted arrows using
        # FINAL activity IDs.
        diagram.dependencies = []
        for vdep in clean:
            source_id = node_to_activity.get(vdep.source_id, "")
            target_id = node_to_activity.get(vdep.target_id, "")
            if not source_id or not target_id:
                continue
            if source_id == target_id:
                continue
            evidence_traces = [EvidenceTrace(
                source_phase="validated_dependency",
                source_ids=[getattr(vdep, "arrow_id", "")],
                confidence_contribution=vdep.confidence_score,
                description=(
                    f"Validated arrow: {source_id}→{target_id} "
                    f"(score={vdep.confidence_score:.3f}, "
                    f"level={getattr(vdep, 'confidence_level', '')})"
                ),
                metadata={
                    "source_boundary": getattr(
                        getattr(vdep, "evidence", None),
                        "source_boundary_intersection", None,
                    ),
                    "target_boundary": getattr(
                        getattr(vdep, "evidence", None),
                        "target_boundary_intersection", None,
                    ),
                    "direction": getattr(
                        getattr(vdep, "evidence", None), "direction_consistency", None,
                    ),
                    "angular": getattr(
                        getattr(vdep, "evidence", None), "angular_consistency", None,
                    ),
                },
            )]
            diagram.dependencies.append(ReconstructedDependency(
                source_id=source_id,
                target_id=target_id,
                confidence=vdep.confidence_score,
                evidence=evidence_traces,
                source_arrow_id=getattr(vdep, "arrow_id", ""),
                metadata={
                    **getattr(vdep, "metadata", {}),
                    "confidence_level": getattr(vdep, "confidence_level", ""),
                    "validation_score": vdep.confidence_score,
                    "source_boundary": getattr(
                        getattr(vdep, "evidence", None),
                        "source_boundary_intersection", None,
                    ),
                    "target_boundary": getattr(
                        getattr(vdep, "evidence", None),
                        "target_boundary_intersection", None,
                    ),
                },
            ))

        if conflicted:
            logger.info(
                "Dependency finalize: %d conflicted reciprocal arrow pair(s) "
                "routed to review; %d deps re-anchored to final IDs",
                len(conflicted), len(diagram.dependencies),
            )

    # =========================================================================
    # Private: AOA Reconstruction
    # =========================================================================

    def _reconstruct_aoa_events(
        self,
        diagram: ReconstructedDiagram,
        shape_result: ShapeDetectionResult,
        candidate_map: Dict[str, CandidateNode],
        text_map: Dict[str, OCRTextRegion],
    ) -> None:
        """Reconstruct event nodes from circles in AOA."""
        event_counter = 0
        for candidate in shape_result.candidate_nodes:
            if candidate.shape_type not in (
                ShapeType.CIRCLE, ShapeType.ELLIPSE
            ):
                continue

            event_counter += 1
            evt_id = str(event_counter)
            label = ""

            # Find associated text
            for region in text_map.values():
                if candidate.bounding_box.contains_point(
                    region.bounding_box.center
                ):
                    label = region.raw_text
                    extracted_id = self._extract_activity_id(label)
                    if extracted_id:
                        evt_id = extracted_id
                    break

            evt = ReconstructedEvent(
                event_id=evt_id,
                label=label or evt_id,
                confidence=candidate.confidence,
                evidence=[EvidenceTrace(
                    source_phase="shape_detection",
                    source_ids=[candidate.source_shape_id],
                    confidence_contribution=candidate.confidence,
                    description=f"Circle/event detected (confidence={candidate.confidence:.3f})",
                )],
                source_shape_id=candidate.source_shape_id,
                position=(candidate.position.x, candidate.position.y),
                bounding_box=candidate.bounding_box,
            )

            diagram.events.append(evt)

    def _reconstruct_aoa_activities(
        self,
        diagram: ReconstructedDiagram,
        arrow_result: ArrowDetectionResult,
        arrow_map: Dict[str, DetectedArrow],
        candidate_map: Dict[str, CandidateNode],
        text_map: Dict[str, OCRTextRegion],
        association_result: Optional[TextAssociationResult],
    ) -> None:
        """Reconstruct activities from arrows in AOA."""
        act_counter = 0
        for arrow in arrow_result.arrows:
            act_counter += 1
            act_id = f"A{act_counter}"
            label = ""
            duration = 0.0
            text_region_ids: List[str] = []
            evidence_traces: List[EvidenceTrace] = []

            # Arrow evidence
            evidence_traces.append(EvidenceTrace(
                source_phase="arrow_detection",
                source_ids=[arrow.arrow_id],
                confidence_contribution=arrow.confidence,
                description=f"Arrow/activity detected (confidence={arrow.confidence:.3f})",
            ))

            # Find text associated with this arrow
            if association_result:
                text_region_ids, label, duration, text_evidence = (
                    self._find_text_for_arrow(
                        arrow, association_result, text_map
                    )
                )
                evidence_traces.extend(text_evidence)

            if label:
                extracted_id = self._extract_activity_id(label)
                if extracted_id:
                    act_id = extracted_id

            missing_arrowhead = arrow.arrowhead_point is None
            if missing_arrowhead:
                evidence_traces.append(EvidenceTrace(
                    source_phase="arrow_detection",
                    source_ids=[arrow.arrow_id],
                    confidence_contribution=0.0,
                    description=(
                        "Arrowhead not observed; direction is geometry/layout-derived "
                        "and requires human review"
                    ),
                    metadata={
                        "recovered_route": bool(
                            arrow.evidence.get("aoa_validation", {}).get(
                                "recovered_route", False
                            )
                        ),
                        "direction_source": arrow.metadata.get(
                            "direction_source", "unavailable"
                        ),
                    },
                ))

            act = ReconstructedActivity(
                activity_id=act_id,
                label=label or act_id,
                duration=duration,
                confidence=self._aggregate_confidence([
                    arrow.confidence,
                    *[e.confidence_contribution for e in evidence_traces],
                ]),
                evidence=evidence_traces,
                source_arrow_id=arrow.arrow_id,
                source_text_region_ids=text_region_ids,
                position=(arrow.midpoint.x, arrow.midpoint.y),
                warnings=([] if duration > 0 else ["No duration detected"])
                + (["Arrowhead not observed; direction requires review"]
                   if missing_arrowhead else []),
                metadata={
                    "missing_arrowhead": missing_arrowhead,
                    "recovered_route": bool(
                        arrow.evidence.get("aoa_validation", {}).get(
                            "recovered_route", False
                        )
                    ),
                },
                needs_review=missing_arrowhead,
                # AOA activities are arrows; geometric identity is the arrow
                geometric_node_id=arrow.arrow_id,
                semantic_activity_id=(
                    extracted_id if label and extracted_id else None
                ),
                semantic_status=(
                    ActivityStatus.CONFIRMED
                    if label and extracted_id
                    else ActivityStatus.REVIEW_REQUIRED
                ),
            )

            diagram.activities.append(act)

    def _reconstruct_aoa_dependencies(
        self,
        diagram: ReconstructedDiagram,
        arrow_result: ArrowDetectionResult,
        arrow_map: Dict[str, DetectedArrow],
    ) -> None:
        """Reconstruct dependencies between activities in AOA.

        AOA arrows ARE the activities; precedence is defined by shared
        event nodes. For each activity that ends at event E and each
        activity that starts at event E, we add ``pred → succ``.

        Endpoints reference FINAL (disambiguated) activity IDs so the
        downstream AON-style graph can resolve them. Without this
        translation, event-ID endpoints could never map to activities
        and the graph would silently drop every edge.
        """
        event_of_start: Dict[str, Optional[str]] = {}
        event_of_end: Dict[str, Optional[str]] = {}
        arrow_of: Dict[str, str] = {}
        confidence_of: Dict[str, float] = {}

        for act in diagram.activities:
            if act.source_arrow_id not in arrow_map:
                continue
            arrow = arrow_map[act.source_arrow_id]
            start_event = self._find_nearest_event(arrow.start, diagram.events)
            end_event = self._find_nearest_event(arrow.end, diagram.events)
            event_of_start[act.activity_id] = start_event.event_id if start_event else None
            event_of_end[act.activity_id] = end_event.event_id if end_event else None
            arrow_of[act.activity_id] = act.source_arrow_id
            confidence_of[act.activity_id] = arrow.confidence

        outgoing: Dict[str, List[str]] = defaultdict(list)  # event -> activities
        incoming: Dict[str, List[str]] = defaultdict(list)  # event -> activities
        for aid, event in event_of_start.items():
            if event:
                outgoing[event].append(aid)
        for aid, event in event_of_end.items():
            if event:
                incoming[event].append(aid)

        seen_edges: set = set()
        edge_candidates: List[Tuple[str, str, str]] = []  # (pred, succ, via_event)
        for event, preds in incoming.items():
            for succ in outgoing.get(event, []):
                for pred in preds:
                    if pred == succ:
                        continue
                    if (pred, succ) in seen_edges:
                        continue
                    seen_edges.add((pred, succ))
                    edge_candidates.append((pred, succ, event))

        # Contradictory precedence: a cycle of length two (both pred→succ and
        # succ→pred appear in the arrow evidence) cannot both be true.
        # Route both directions to human review instead of auto-accepting.
        conflicted_pairs: set = set()
        for pred, succ, _ in edge_candidates:
            if (succ, pred) in seen_edges:
                conflicted_pairs.add((pred, succ))
                conflicted_pairs.add((succ, pred))

        for pred, succ, via_event in edge_candidates:
            if (pred, succ) in conflicted_pairs:
                diagram.ambiguities.append(AmbiguityIssue(
                    ambiguity_type=AmbiguityType.ARROW_AMBIGUOUS,
                    description=(
                        f"Conflicting arrows brace both '{pred}'→'{succ}' and "
                        f"'{succ}'→'{pred}'; requires review"
                    ),
                    involved_ids=[arrow_of.get(pred, ""), arrow_of.get(succ, "")],
                    severity=ValidationSeverity.INFO,
                ))
                continue
            diagram.dependencies.append(ReconstructedDependency(
                source_id=pred,
                target_id=succ,
                confidence=min(
                    confidence_of.get(pred, 1.0),
                    confidence_of.get(succ, 1.0),
                ),
                evidence=[EvidenceTrace(
                    source_phase="arrow_detection",
                    source_ids=[arrow_of.get(pred, ""), arrow_of.get(succ, "")],
                    confidence_contribution=min(
                        confidence_of.get(pred, 1.0),
                        confidence_of.get(succ, 1.0),
                    ),
                    description=f"AOA precedence: '{pred}' precedes '{succ}' via event '{via_event}'",
                )],
                source_arrow_id=arrow_of.get(pred, ""),
            ))

    # =========================================================================
    # Private: Text Resolution
    # =========================================================================

    def _find_text_for_shape(
        self,
        candidate: CandidateNode,
        association_result: TextAssociationResult,
        text_map: Dict[str, OCRTextRegion],
    ) -> Tuple[List[str], str, float, List[EvidenceTrace]]:
        """
        Find text associated with a shape candidate.

        Returns:
            Tuple of (region_ids, label, duration, evidence_traces).
        """
        region_ids: List[str] = []
        label = ""
        duration = 0.0
        evidence_traces: List[EvidenceTrace] = []
        best_numeric_score = -1.0
        best_numeric_region: Optional[OCRTextRegion] = None
        best_numeric_value: Optional[float] = None

        for assoc in association_result.associations:
            if (assoc.target_type == AssociationTargetType.NODE
                    and assoc.candidate_target_id == candidate.node_id):
                region = text_map.get(assoc.text_region_id)
                if region:
                    region_ids.append(region.region_id)

                    if region.text_type == TextType.ACTIVITY_ID_CANDIDATE:
                        label = region.normalized_text
                        evidence_traces.append(EvidenceTrace(
                            source_phase="ocr_association",
                            source_ids=[region.region_id],
                            confidence_contribution=assoc.association_score,
                            description=f"Activity ID text: '{region.raw_text}'",
                        ))
                    elif region.text_type == TextType.NUMERIC_CANDIDATE:
                        num_val = self._parse_numeric(region.normalized_text)
                        # Highest association score wins, not the last
                        # region processed (the list is score-sorted
                        # descending, so "last wins" picked the WORST
                        # candidate when a node has several numbers).
                        if num_val is not None and assoc.association_score > best_numeric_score:
                            best_numeric_score = assoc.association_score
                            best_numeric_region = region
                            best_numeric_value = num_val
                    elif region.text_type == TextType.TEXT_LABEL_CANDIDATE:
                        if not label:
                            label = region.raw_text
                        evidence_traces.append(EvidenceTrace(
                            source_phase="ocr_association",
                            source_ids=[region.region_id],
                            confidence_contribution=assoc.association_score,
                            description=f"Label text: '{region.raw_text}'",
                        ))

        if best_numeric_value is not None and best_numeric_region is not None:
            duration = best_numeric_value
            evidence_traces.append(EvidenceTrace(
                source_phase="ocr_association",
                source_ids=[best_numeric_region.region_id],
                confidence_contribution=best_numeric_score,
                description=f"Duration text: '{best_numeric_region.raw_text}' -> {duration}",
            ))

        return region_ids, label, duration, evidence_traces

    def _find_text_for_arrow(
        self,
        arrow: DetectedArrow,
        association_result: TextAssociationResult,
        text_map: Dict[str, OCRTextRegion],
    ) -> Tuple[List[str], str, float, List[EvidenceTrace]]:
        """
        Find text associated with an arrow.

        Returns:
            Tuple of (region_ids, label, duration, evidence_traces).
        """
        region_ids: List[str] = []
        label = ""
        duration = 0.0
        evidence_traces: List[EvidenceTrace] = []
        best_numeric_score = -1.0
        best_numeric_region: Optional[OCRTextRegion] = None
        best_numeric_value: Optional[float] = None

        for assoc in association_result.associations:
            if (assoc.target_type == AssociationTargetType.ARROW
                    and assoc.candidate_target_id == arrow.arrow_id
                    and assoc.is_best_candidate):
                region = text_map.get(assoc.text_region_id)
                if region:
                    region_ids.append(region.region_id)

                    if region.text_type == TextType.NUMERIC_CANDIDATE:
                        num_val = self._parse_numeric(region.normalized_text)
                        # Highest association score wins, not the last
                        # region processed (see _find_text_for_shape).
                        if num_val is not None and assoc.association_score > best_numeric_score:
                            best_numeric_score = assoc.association_score
                            best_numeric_region = region
                            best_numeric_value = num_val
                    else:
                        if not label:
                            label = region.raw_text
                        evidence_traces.append(EvidenceTrace(
                            source_phase="ocr_association",
                            source_ids=[region.region_id],
                            confidence_contribution=assoc.association_score,
                            description=f"Label text: '{region.raw_text}'",
                        ))

        if best_numeric_value is not None and best_numeric_region is not None:
            duration = best_numeric_value
            evidence_traces.append(EvidenceTrace(
                source_phase="ocr_association",
                source_ids=[best_numeric_region.region_id],
                confidence_contribution=best_numeric_score,
                description=f"Duration text: '{best_numeric_region.raw_text}' -> {duration}",
            ))

        return region_ids, label, duration, evidence_traces

    # =========================================================================
    # Private: Node Resolution (for dependencies)
    # =========================================================================

    def _resolve_arrow_source(
        self,
        arrow: DetectedArrow,
        activities: List[ReconstructedActivity],
        events: List[ReconstructedEvent],
        candidate_map: Dict[str, CandidateNode],
    ) -> Optional[str]:
        """Resolve which node an arrow's source endpoint connects to."""
        # Skip self-loop pre-assigned candidates
        source_id = arrow.evidence.get("node_at_endpoint_1")
        target_id = arrow.evidence.get("node_at_endpoint_2")
        if source_id and source_id != target_id:
            for act in activities:
                if act.source_shape_id and act.source_shape_id == candidate_map.get(
                    source_id, CandidateNode()
                ).source_shape_id:
                    return act.activity_id
            for evt in events:
                if evt.source_shape_id and evt.source_shape_id == candidate_map.get(
                    source_id, CandidateNode()
                ).source_shape_id:
                    return evt.event_id

        # Fallback: find closest element to arrow start
        return self._find_closest_element_id(
            arrow.start, activities, events
        )

    def _resolve_arrow_target(
        self,
        arrow: DetectedArrow,
        activities: List[ReconstructedActivity],
        events: List[ReconstructedEvent],
        candidate_map: Dict[str, CandidateNode],
    ) -> Optional[str]:
        """Resolve which node an arrow's target endpoint connects to."""
        # Skip self-loop pre-assigned candidates
        target_id = arrow.evidence.get("node_at_endpoint_2")
        source_id = arrow.evidence.get("node_at_endpoint_1")
        if target_id and source_id != target_id:
            for act in activities:
                if act.source_shape_id and act.source_shape_id == candidate_map.get(
                    target_id, CandidateNode()
                ).source_shape_id:
                    return act.activity_id
            for evt in events:
                if evt.source_shape_id and evt.source_shape_id == candidate_map.get(
                    target_id, CandidateNode()
                ).source_shape_id:
                    return evt.event_id

        # Use arrowhead point if available (more reliable for target)
        if arrow.arrowhead_point:
            result = self._find_closest_element_id(
                arrow.arrowhead_point, activities, events
            )
            if result:
                return result

        # Fallback: find closest element to arrow end
        return self._find_closest_element_id(
            arrow.end, activities, events
        )

    def _find_closest_element_id(
        self,
        point: Any,
        activities: List[ReconstructedActivity],
        events: List[ReconstructedEvent],
    ) -> Optional[str]:
        """Find the closest reconstructed element to a point."""
        import math

        best_id = None
        best_dist = float("inf")

        for act in activities:
            if act.position:
                dist = math.sqrt(
                    (point.x - act.position[0]) ** 2
                    + (point.y - act.position[1]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_id = act.activity_id

        for evt in events:
            if evt.position:
                dist = math.sqrt(
                    (point.x - evt.position[0]) ** 2
                    + (point.y - evt.position[1]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_id = evt.event_id

        return best_id

    def _find_nearest_event(
        self,
        point: Any,
        events: List[ReconstructedEvent],
    ) -> Optional[ReconstructedEvent]:
        """Find the nearest event node to a point."""
        import math

        best_event = None
        best_dist = float("inf")

        for evt in events:
            if evt.position:
                dist = math.sqrt(
                    (point.x - evt.position[0]) ** 2
                    + (point.y - evt.position[1]) ** 2
                )
                if dist < best_dist:
                    best_dist = dist
                    best_event = evt

        return best_event

    # =========================================================================
    # Private: Duplicate ID Disambiguation
    # =========================================================================

    def _disambiguate_duplicate_ids(self, diagram: ReconstructedDiagram) -> None:
        """Disambiguate duplicate activity IDs by appending spatial suffixes.

        When multiple activities get the same OCR-derived ID (e.g., three "A"s),
        they are renamed to "A_1", "A_2", "A_3" in spatial order (top-to-bottom,
        left-to-right). INFERRED_* IDs are left unchanged.
        """
        id_to_acts: Dict[str, List[ReconstructedActivity]] = {}
        for act in diagram.activities:
            aid = act.activity_id
            if aid not in id_to_acts:
                id_to_acts[aid] = []
            id_to_acts[aid].append(act)

        for aid, acts in id_to_acts.items():
            if len(acts) <= 1 or aid.startswith("INFERRED_"):
                continue
            acts.sort(key=lambda a: (round(a.position[1] / 40.0), a.position[0]))
            for idx, act in enumerate(acts):
                new_id = f"{aid}_{idx + 1}"
                old_id = act.activity_id
                act.activity_id = new_id
                if act.label == old_id:
                    act.label = new_id
                for dep in diagram.dependencies:
                    if dep.source_id == old_id:
                        dep.source_id = new_id
                    if dep.target_id == old_id:
                        dep.target_id = new_id

    # =========================================================================
    # Private: Ambiguity Detection
    # =========================================================================

    def _detect_ambiguities(self, diagram: ReconstructedDiagram) -> None:
        """Detect ambiguities in the reconstructed diagram."""
        # Check for unassociated text regions
        all_text_ids = set()
        for act in diagram.activities:
            all_text_ids.update(act.source_text_region_ids)
        for evt in diagram.events:
            all_text_ids.update(evt.source_text_region_ids)
        for dep in diagram.dependencies:
            all_text_ids.update(dep.source_text_region_ids)

        # Check for duplicate labels
        labels = [a.label for a in diagram.activities if a.label]
        seen_labels: Dict[str, int] = {}
        for lbl in labels:
            seen_labels[lbl] = seen_labels.get(lbl, 0) + 1
        for lbl, count in seen_labels.items():
            if count > 1:
                diagram.ambiguities.append(AmbiguityIssue(
                    ambiguity_type=AmbiguityType.DUPLICATE_LABEL,
                    description=f"Duplicate label '{lbl}' found in {count} activities",
                    involved_ids=[a.activity_id for a in diagram.activities if a.label == lbl],
                    severity=ValidationSeverity.WARNING,
                ))

        # Check for activities without text
        for act in diagram.activities:
            if not act.source_text_region_ids:
                diagram.ambiguities.append(AmbiguityIssue(
                    ambiguity_type=AmbiguityType.SHAPE_NO_TEXT,
                    description=f"Activity '{act.activity_id}' has no associated text",
                    involved_ids=[act.activity_id],
                    severity=ValidationSeverity.INFO,
                ))

        # Check for activities with inferred IDs (needs review)
        for act in diagram.activities:
            if act.activity_id.startswith("INFERRED_"):
                diagram.ambiguities.append(AmbiguityIssue(
                    ambiguity_type=AmbiguityType.MISSING_ACTIVITY_ID,
                    description=(
                        f"Activity '{act.activity_id}' has no OCR-confirmed ID; "
                        f"ID was inferred from spatial position"
                    ),
                    involved_ids=[act.activity_id],
                    severity=ValidationSeverity.WARNING,
                ))

    # =========================================================================
    # Private: Utilities
    # =========================================================================

    def _build_text_region_map(
        self, ocr_result: Optional[OCRProcessingResult]
    ) -> Dict[str, OCRTextRegion]:
        """Build a mapping from region_id to OCRTextRegion."""
        if not ocr_result:
            return {}
        return {r.region_id: r for r in ocr_result.regions}

    def _extract_activity_id(self, text: str) -> Optional[str]:
        """
        Try to extract an activity ID from text.

        Looks for patterns like "A1", "B2", single letters, or
        short alphanumeric codes.
        """
        import re

        text = text.strip()

        # Pattern: Letter + digits (e.g., "A1", "B2", "T3")
        match = re.match(r'^([A-Za-z])\d*$', text)
        if match:
            return text.upper()

        # Pattern: Single letter
        if len(text) == 1 and text.isalpha():
            return text.upper()

        # Pattern: Short alphanumeric (1-4 chars)
        if re.match(r'^[A-Za-z0-9]{1,4}$', text):
            return text.upper()

        return None

    def _is_plausible_activity_id(self, text: str) -> bool:
        """Check if text is plausible as an AON activity ID.

        Rejects multi-word garbage, Arabic text, long strings,
        and strings that look like arrow labels or annotations.
        Accepts short alphabetic IDs (1-3 chars) typical for AON diagrams.
        """
        import re

        text = text.strip()
        if not text:
            return False

        # Reject multi-word text (more than 2 tokens)
        tokens = text.split()
        if len(tokens) > 2:
            return False

        # Reject very long text (> 6 characters)
        if len(text) > 6:
            return False

        # Reject text containing Arabic/Unicode characters
        if any('\u0600' <= ch <= '\u06FF' or '\u0750' <= ch <= '\u077F' for ch in text):
            return False

        # Reject text with special characters typical of arrow labels
        if re.search(r'[|\\/\-_=+<>{}()\[\]]', text):
            return False

        # Accept single letter (most common AON ID)
        if len(text) == 1 and text.isalpha():
            return True

        # Accept letter + digit (e.g., "A1")
        if re.match(r'^[A-Za-z]\d{0,2}$', text):
            return True

        # Accept short alphabetic (2-4 chars, all alpha)
        if len(text) <= 4 and text.isalpha():
            return True

        return False

    def _parse_numeric(self, text: str) -> Optional[float]:
        """Parse a numeric value from text."""
        import re

        text = text.strip()

        # Try direct float conversion
        try:
            return float(text)
        except ValueError:
            pass

        # Try extracting numbers from text
        match = re.search(r'[-+]?\d*\.?\d+', text)
        if match:
            try:
                return float(match.group())
            except ValueError:
                pass

        return None

    def _aggregate_confidence(self, values: List[float]) -> float:
        """
        Aggregate multiple confidence values into a single score.

        Uses a weighted average where higher individual scores
        contribute more, with a minimum floor.
        """
        if not values:
            return 0.0

        # Filter out zeros (no contribution)
        nonzero = [v for v in values if v > 0]
        if not nonzero:
            return 0.0

        # Weighted average: max value gets 40% weight, rest distributed
        max_val = max(nonzero)
        if len(nonzero) == 1:
            return max_val

        others = [v for v in nonzero if v != max_val]
        avg_others = sum(others) / len(others) if others else 0

        return 0.4 * max_val + 0.6 * avg_others

    def _compute_overall_confidence(self, diagram: ReconstructedDiagram) -> float:
        """Compute the overall confidence of a reconstructed diagram."""
        all_confidences = []
        for act in diagram.activities:
            all_confidences.append(act.confidence)
        for evt in diagram.events:
            all_confidences.append(evt.confidence)
        for dep in diagram.dependencies:
            all_confidences.append(dep.confidence)

        if not all_confidences:
            return 0.0

        return sum(all_confidences) / len(all_confidences)
