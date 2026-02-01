"""
Conflict Resolution Engine

Intelligent disagreement handling between modalities:
- OCR vs Vision detection
- Text vs Table values
- Cross-modal agreement scoring
- Explainable resolution decisions
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ConflictType(Enum):
    """Types of conflicts that can occur"""
    OCR_VISION = "ocr_vision"           # OCR text vs Vision detection
    TEXT_TABLE = "text_table"           # Text mentions vs Table values
    DUPLICATE_DETECTION = "duplicate"   # Same content detected twice
    VALUE_MISMATCH = "value_mismatch"   # Numeric/text value disagreement
    BOUNDARY_OVERLAP = "boundary"       # Overlapping bounding boxes


class ResolutionStrategy(Enum):
    """Resolution strategies"""
    TRUST_OCR = "trust_ocr"
    TRUST_VISION = "trust_vision"
    TRUST_TABLE = "trust_table"
    TRUST_TEXT = "trust_text"
    MERGE = "merge"
    AVERAGE = "average"
    HUMAN_REVIEW = "human_review"


@dataclass
class Conflict:
    """Represents a detected conflict"""
    conflict_id: str
    conflict_type: ConflictType
    source_a: Dict[str, Any]  # First conflicting source
    source_b: Dict[str, Any]  # Second conflicting source
    value_a: Any
    value_b: Any
    confidence_a: float
    confidence_b: float
    page_number: int
    location: Optional[Tuple[float, float, float, float]] = None  # bbox
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "conflict_type": self.conflict_type.value,
            "source_a": self.source_a,
            "source_b": self.source_b,
            "value_a": str(self.value_a),
            "value_b": str(self.value_b),
            "confidence_a": self.confidence_a,
            "confidence_b": self.confidence_b,
            "page_number": self.page_number,
            "location": self.location
        }


@dataclass
class Resolution:
    """Resolution decision for a conflict"""
    conflict_id: str
    strategy: ResolutionStrategy
    chosen_value: Any
    chosen_source: str
    confidence: float
    reasoning: str
    requires_review: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "strategy": self.strategy.value,
            "chosen_value": str(self.chosen_value),
            "chosen_source": self.chosen_source,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "requires_review": self.requires_review
        }


@dataclass
class ConflictReport:
    """Complete conflict analysis report"""
    document_id: str
    total_conflicts: int
    resolved_conflicts: int
    requires_review: int
    conflicts: List[Conflict] = field(default_factory=list)
    resolutions: List[Resolution] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "total_conflicts": self.total_conflicts,
            "resolved_conflicts": self.resolved_conflicts,
            "requires_review": self.requires_review,
            "conflicts": [c.to_dict() for c in self.conflicts],
            "resolutions": [r.to_dict() for r in self.resolutions]
        }


class ConflictResolutionEngine:
    """
    Intelligent conflict resolution between modalities.
    
    Features:
    - Detects conflicts between OCR, Vision, and LLM outputs
    - Applies rule-based and confidence-based resolution
    - Provides explainable decisions
    - Flags high-uncertainty cases for human review
    """
    
    def __init__(
        self,
        ocr_trust_threshold: float = 0.8,
        vision_trust_threshold: float = 0.85,
        conflict_threshold: float = 0.15,
        review_threshold: float = 0.6
    ):
        """
        Initialize conflict resolution engine.
        
        Args:
            ocr_trust_threshold: Confidence to fully trust OCR
            vision_trust_threshold: Confidence to fully trust Vision
            conflict_threshold: Minimum difference to consider a conflict
            review_threshold: Below this, flag for human review
        """
        self.ocr_trust_threshold = ocr_trust_threshold
        self.vision_trust_threshold = vision_trust_threshold
        self.conflict_threshold = conflict_threshold
        self.review_threshold = review_threshold
    
    def analyze(
        self,
        document_id: str,
        ocr_results: Dict[str, Any],
        vision_results: Dict[str, Any],
        fused_output: Dict[str, Any]
    ) -> ConflictReport:
        """
        Analyze document outputs for conflicts.
        
        Args:
            document_id: Document identifier
            ocr_results: OCR agent output
            vision_results: Vision agent output
            fused_output: Fusion agent output
            
        Returns:
            ConflictReport with detected conflicts and resolutions
        """
        conflicts = []
        resolutions = []
        conflict_counter = 0
        
        # 1. Detect OCR vs Vision conflicts
        ocr_vision_conflicts = self._detect_ocr_vision_conflicts(
            ocr_results, vision_results
        )
        
        for conflict in ocr_vision_conflicts:
            conflict.conflict_id = f"conflict_{conflict_counter}"
            conflicts.append(conflict)
            resolution = self._resolve_ocr_vision(conflict)
            resolutions.append(resolution)
            conflict_counter += 1
        
        # 2. Detect text vs table conflicts
        text_table_conflicts = self._detect_text_table_conflicts(fused_output)
        
        for conflict in text_table_conflicts:
            conflict.conflict_id = f"conflict_{conflict_counter}"
            conflicts.append(conflict)
            resolution = self._resolve_text_table(conflict)
            resolutions.append(resolution)
            conflict_counter += 1
        
        # 3. Detect value mismatches within fused output
        value_conflicts = self._detect_value_mismatches(fused_output)
        
        for conflict in value_conflicts:
            conflict.conflict_id = f"conflict_{conflict_counter}"
            conflicts.append(conflict)
            resolution = self._resolve_value_mismatch(conflict)
            resolutions.append(resolution)
            conflict_counter += 1
        
        requires_review = sum(1 for r in resolutions if r.requires_review)
        
        return ConflictReport(
            document_id=document_id,
            total_conflicts=len(conflicts),
            resolved_conflicts=len(conflicts) - requires_review,
            requires_review=requires_review,
            conflicts=conflicts,
            resolutions=resolutions
        )
    
    def _detect_ocr_vision_conflicts(
        self,
        ocr_results: Dict[str, Any],
        vision_results: Dict[str, Any]
    ) -> List[Conflict]:
        """Detect conflicts between OCR text and Vision detections"""
        conflicts = []
        
        ocr_blocks = ocr_results.get("text_blocks", [])
        vision_detections = vision_results.get("detections", [])
        
        # Check for tables detected by vision but no structured text from OCR
        for detection in vision_detections:
            if detection.get("label") == "table":
                detection_bbox = (
                    detection.get("x1", 0),
                    detection.get("y1", 0),
                    detection.get("x2", 0),
                    detection.get("y2", 0)
                )
                
                # Find OCR text in this region
                overlapping_text = self._find_text_in_region(
                    ocr_blocks, detection_bbox
                )
                
                if not overlapping_text:
                    # Vision detected table but no OCR text
                    conflicts.append(Conflict(
                        conflict_id="",
                        conflict_type=ConflictType.OCR_VISION,
                        source_a={"type": "vision", "detection": detection},
                        source_b={"type": "ocr", "text": ""},
                        value_a="table_detected",
                        value_b="no_text",
                        confidence_a=detection.get("confidence", 0),
                        confidence_b=0.0,
                        page_number=detection.get("page_number", 0),
                        location=detection_bbox
                    ))
        
        return conflicts
    
    def _detect_text_table_conflicts(
        self,
        fused_output: Dict[str, Any]
    ) -> List[Conflict]:
        """Detect conflicts between text mentions and table values"""
        conflicts = []
        
        elements = fused_output.get("elements", [])
        tables = fused_output.get("tables", [])
        
        # Extract numbers from text
        text_numbers = {}
        for element in elements:
            if element.get("type") in ["paragraph", "text"]:
                content = element.get("content", "")
                numbers = self._extract_numbers(content)
                for num, context in numbers:
                    text_numbers[context] = num
        
        # Check against table values
        for table in tables:
            for row in table.get("rows", []):
                for i, cell in enumerate(row):
                    cell_value = self._parse_number(str(cell))
                    if cell_value is not None:
                        # Look for similar context in text
                        for context, text_value in text_numbers.items():
                            if self._values_conflict(cell_value, text_value):
                                conflicts.append(Conflict(
                                    conflict_id="",
                                    conflict_type=ConflictType.TEXT_TABLE,
                                    source_a={"type": "table", "table_id": table.get("id")},
                                    source_b={"type": "text", "context": context},
                                    value_a=cell_value,
                                    value_b=text_value,
                                    confidence_a=table.get("confidence", 0.8),
                                    confidence_b=0.7,
                                    page_number=table.get("page_number", 0)
                                ))
        
        return conflicts
    
    def _detect_value_mismatches(
        self,
        fused_output: Dict[str, Any]
    ) -> List[Conflict]:
        """Detect value mismatches within the fused output"""
        conflicts = []
        
        # Check for duplicate entity extractions with different values
        entities = fused_output.get("entities", [])
        entity_values = {}
        
        for entity in entities:
            name = entity.get("name", "")
            value = entity.get("value", "")
            
            if name in entity_values:
                if entity_values[name] != value:
                    conflicts.append(Conflict(
                        conflict_id="",
                        conflict_type=ConflictType.VALUE_MISMATCH,
                        source_a={"type": "entity", "name": name},
                        source_b={"type": "entity", "name": name},
                        value_a=entity_values[name],
                        value_b=value,
                        confidence_a=0.7,
                        confidence_b=0.7,
                        page_number=entity.get("page_number", 0)
                    ))
            else:
                entity_values[name] = value
        
        return conflicts
    
    def _resolve_ocr_vision(self, conflict: Conflict) -> Resolution:
        """Resolve OCR vs Vision conflict"""
        # Generally trust vision for structure, OCR for text
        if conflict.confidence_a > self.vision_trust_threshold:
            return Resolution(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.TRUST_VISION,
                chosen_value=conflict.value_a,
                chosen_source="vision",
                confidence=conflict.confidence_a,
                reasoning=f"Vision detection has high confidence ({conflict.confidence_a:.2f}). "
                         f"Table structure detected visually but OCR may have failed to extract text. "
                         f"Recommend re-processing the region with enhanced OCR.",
                requires_review=True
            )
        elif conflict.confidence_b > self.ocr_trust_threshold:
            return Resolution(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.TRUST_OCR,
                chosen_value=conflict.value_b,
                chosen_source="ocr",
                confidence=conflict.confidence_b,
                reasoning=f"OCR has high confidence ({conflict.confidence_b:.2f})",
                requires_review=False
            )
        else:
            return Resolution(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.HUMAN_REVIEW,
                chosen_value=None,
                chosen_source="undetermined",
                confidence=max(conflict.confidence_a, conflict.confidence_b),
                reasoning=f"Both sources have moderate confidence. "
                         f"Vision: {conflict.confidence_a:.2f}, OCR: {conflict.confidence_b:.2f}. "
                         f"Human review required.",
                requires_review=True
            )
    
    def _resolve_text_table(self, conflict: Conflict) -> Resolution:
        """Resolve text vs table conflict"""
        # Generally trust tables for structured numeric data
        value_a = conflict.value_a
        value_b = conflict.value_b
        
        # Check if it's a magnitude difference (likely OCR error)
        if isinstance(value_a, (int, float)) and isinstance(value_b, (int, float)):
            ratio = max(value_a, value_b) / min(value_a, value_b) if min(value_a, value_b) > 0 else float('inf')
            
            if ratio in [10, 100, 1000]:  # Likely magnitude error
                # Trust the one that makes more sense contextually
                # Usually tables have structured data, so trust table
                return Resolution(
                    conflict_id=conflict.conflict_id,
                    strategy=ResolutionStrategy.TRUST_TABLE,
                    chosen_value=conflict.value_a,
                    chosen_source="table",
                    confidence=0.85,
                    reasoning=f"OCR read {value_b} but table structure suggests {value_a}. "
                             f"Magnitude difference of {ratio}x indicates likely OCR error. "
                             f"Table value chosen due to structural confidence.",
                    requires_review=ratio > 100
                )
        
        # Trust the higher confidence source
        if conflict.confidence_a > conflict.confidence_b:
            return Resolution(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.TRUST_TABLE,
                chosen_value=conflict.value_a,
                chosen_source="table",
                confidence=conflict.confidence_a,
                reasoning=f"Table has higher confidence ({conflict.confidence_a:.2f} vs {conflict.confidence_b:.2f})",
                requires_review=False
            )
        else:
            return Resolution(
                conflict_id=conflict.conflict_id,
                strategy=ResolutionStrategy.TRUST_TEXT,
                chosen_value=conflict.value_b,
                chosen_source="text",
                confidence=conflict.confidence_b,
                reasoning=f"Text has higher confidence ({conflict.confidence_b:.2f} vs {conflict.confidence_a:.2f})",
                requires_review=True  # Text over table is unusual, review
            )
    
    def _resolve_value_mismatch(self, conflict: Conflict) -> Resolution:
        """Resolve general value mismatch"""
        # Flag for human review with explanation
        return Resolution(
            conflict_id=conflict.conflict_id,
            strategy=ResolutionStrategy.HUMAN_REVIEW,
            chosen_value=conflict.value_a,  # Default to first occurrence
            chosen_source="first_occurrence",
            confidence=0.5,
            reasoning=f"Duplicate entity '{conflict.source_a.get('name')}' found with different values: "
                     f"'{conflict.value_a}' vs '{conflict.value_b}'. Using first occurrence, but review recommended.",
            requires_review=True
        )
    
    def _find_text_in_region(
        self,
        text_blocks: List[Dict],
        bbox: Tuple[float, float, float, float]
    ) -> List[Dict]:
        """Find text blocks within a bounding box region"""
        x1, y1, x2, y2 = bbox
        overlapping = []
        
        for block in text_blocks:
            bx1 = block.get("x1", 0)
            by1 = block.get("y1", 0)
            bx2 = block.get("x2", 0)
            by2 = block.get("y2", 0)
            
            # Check overlap
            if bx1 < x2 and bx2 > x1 and by1 < y2 and by2 > y1:
                overlapping.append(block)
        
        return overlapping
    
    def _extract_numbers(self, text: str) -> List[Tuple[float, str]]:
        """Extract numbers with context from text"""
        import re
        
        results = []
        # Match numbers including currency symbols
        pattern = r'[$€£]?\s*[\d,]+\.?\d*\s*[MBK]?'
        
        for match in re.finditer(pattern, text):
            num_str = match.group()
            value = self._parse_number(num_str)
            if value is not None:
                # Get surrounding context
                start = max(0, match.start() - 20)
                end = min(len(text), match.end() + 20)
                context = text[start:end].strip()
                results.append((value, context))
        
        return results
    
    def _parse_number(self, text: str) -> Optional[float]:
        """Parse number from text"""
        import re
        
        # Remove currency symbols and commas
        cleaned = re.sub(r'[$€£,\s]', '', text)
        
        # Handle M/B/K suffixes
        multiplier = 1
        if cleaned.endswith('M'):
            multiplier = 1_000_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith('B'):
            multiplier = 1_000_000_000
            cleaned = cleaned[:-1]
        elif cleaned.endswith('K'):
            multiplier = 1_000
            cleaned = cleaned[:-1]
        
        try:
            return float(cleaned) * multiplier
        except ValueError:
            return None
    
    def _values_conflict(
        self,
        value_a: float,
        value_b: float
    ) -> bool:
        """Determine if two values are in conflict"""
        if value_a == 0 and value_b == 0:
            return False
        
        if value_a == 0 or value_b == 0:
            return True
        
        # Check percentage difference
        diff = abs(value_a - value_b) / max(value_a, value_b)
        return diff > self.conflict_threshold
    
    def generate_summary(self, report: ConflictReport) -> str:
        """Generate human-readable conflict summary"""
        if report.total_conflicts == 0:
            return "No conflicts detected between modalities. All sources agree."
        
        summary_parts = [
            f"## Conflict Resolution Summary\n",
            f"**Document**: {report.document_id}\n",
            f"**Total Conflicts**: {report.total_conflicts}\n",
            f"**Auto-Resolved**: {report.resolved_conflicts}\n",
            f"**Requires Human Review**: {report.requires_review}\n\n"
        ]
        
        # Group by type
        by_type = {}
        for conflict, resolution in zip(report.conflicts, report.resolutions):
            ctype = conflict.conflict_type.value
            if ctype not in by_type:
                by_type[ctype] = []
            by_type[ctype].append((conflict, resolution))
        
        for ctype, items in by_type.items():
            summary_parts.append(f"### {ctype.replace('_', ' ').title()}\n")
            for conflict, resolution in items:
                summary_parts.append(
                    f"- **Page {conflict.page_number}**: "
                    f"'{conflict.value_a}' vs '{conflict.value_b}' → "
                    f"Chose: {resolution.chosen_value} ({resolution.strategy.value})\n"
                    f"  - Reasoning: {resolution.reasoning}\n"
                )
        
        return "".join(summary_parts)
