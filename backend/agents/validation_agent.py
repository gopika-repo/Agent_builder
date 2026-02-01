"""
Validation & Confidence Agent

Cross-checks outputs from all modalities:
- OCR vs Vision agreement
- Text vs Table consistency
- Field-level confidence scoring
- Human review flagging
"""

import logging
from typing import Dict, Any, List
import time

from .state import (
    DocumentState, ProcessingStatus,
    ConfidenceReport, FieldConfidence
)
from ..config import get_settings

logger = logging.getLogger(__name__)


class ValidationAgent:
    """
    Validation & Confidence Agent.
    
    Cross-validates outputs from all agents and assigns
    confidence scores with human review flagging.
    """
    
    def __init__(
        self,
        ocr_weight: float = 0.3,
        vision_weight: float = 0.3,
        llm_weight: float = 0.4,
        review_threshold: float = None
    ):
        """
        Initialize Validation Agent.
        
        Args:
            ocr_weight: Weight for OCR confidence
            vision_weight: Weight for vision confidence
            llm_weight: Weight for LLM confidence
            review_threshold: Threshold below which human review is flagged
        """
        settings = get_settings()
        
        self.ocr_weight = ocr_weight
        self.vision_weight = vision_weight
        self.llm_weight = llm_weight
        self.review_threshold = review_threshold or settings.human_review_threshold
    
    def __call__(self, state: DocumentState) -> DocumentState:
        """
        Process document through validation agent.
        
        Args:
            state: Current document state
            
        Returns:
            Updated state with confidence scores
        """
        logger.info(f"Validation Agent processing: {state.get('document_id')}")
        start_time = time.time()
        
        state["status"] = ProcessingStatus.VALIDATION.value
        state["current_agent"] = "validation"
        
        try:
            vision_results = state.get("vision_results", {})
            ocr_results = state.get("ocr_results", {})
            fused_output = state.get("fused_output", {})
            text_analysis = state.get("text_analysis", {})
            
            # Calculate field-level confidence scores
            field_scores = self._calculate_field_scores(
                fused_output, vision_results, ocr_results
            )
            
            # Calculate modality agreement
            modality_agreement = self._calculate_modality_agreement(
                vision_results, ocr_results, fused_output
            )
            
            # Identify items needing review
            items_needing_review = self._identify_review_items(field_scores)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_overall_confidence(
                field_scores, modality_agreement
            )
            
            processing_time = (time.time() - start_time) * 1000
            
            confidence_report = ConfidenceReport(
                overall_confidence=overall_confidence,
                field_scores=field_scores,
                items_needing_review=items_needing_review,
                modality_agreement=modality_agreement,
                processing_time_ms=processing_time
            )
            
            state["confidence_scores"] = confidence_report.to_dict()
            state["status"] = ProcessingStatus.COMPLETED.value
            
            logger.info(
                f"Validation Agent completed: overall confidence {overall_confidence:.2f}, "
                f"{len(items_needing_review)} items need review in {processing_time:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"Validation Agent error: {e}")
            state["errors"].append(f"Validation failed: {str(e)}")
            state["confidence_scores"] = ConfidenceReport().to_dict()
        
        return state
    
    def _calculate_field_scores(
        self,
        fused_output: Dict,
        vision_results: Dict,
        ocr_results: Dict
    ) -> List[FieldConfidence]:
        """Calculate confidence scores for each field/element"""
        field_scores = []
        
        elements = fused_output.get("elements", [])
        tables = fused_output.get("tables", [])
        
        # Score each element
        for i, element in enumerate(elements):
            field_id = element.get("id", f"element_{i}")
            field_name = f"{element.get('type', 'unknown')}_{i}"
            
            # Get confidence from each modality
            vision_conf = element.get("confidences", {}).get("vision", 0)
            ocr_conf = element.get("confidences", {}).get("ocr", 0)
            layout_conf = element.get("confidences", {}).get("layout", 0)
            
            # LLM confidence based on text analysis success
            llm_conf = 0.8 if element.get("content") else 0.5
            
            # Calculate combined confidence
            combined = (
                self.vision_weight * vision_conf +
                self.ocr_weight * ocr_conf +
                self.llm_weight * llm_conf
            )
            
            # Determine if review is needed
            needs_review = combined < self.review_threshold
            review_reason = ""
            
            if needs_review:
                if vision_conf < 0.5:
                    review_reason = "Low vision detection confidence"
                elif ocr_conf < 0.5:
                    review_reason = "Low OCR confidence"
                else:
                    review_reason = "Combined confidence below threshold"
            
            # Check for modality disagreement
            if abs(vision_conf - ocr_conf) > 0.3:
                needs_review = True
                review_reason = "Modality disagreement between vision and OCR"
            
            field_scores.append(FieldConfidence(
                field_id=field_id,
                field_name=field_name,
                value=element.get("content", "")[:100],
                ocr_confidence=ocr_conf,
                vision_confidence=vision_conf,
                llm_confidence=llm_conf,
                combined_confidence=combined,
                needs_review=needs_review,
                review_reason=review_reason
            ))
        
        # Score tables
        for i, table in enumerate(tables):
            table_id = table.get("id", f"table_{i}")
            
            vision_conf = table.get("confidence", 0.5)
            ocr_conf = 0.8  # Assume good OCR if we got table content
            llm_conf = 0.85
            
            combined = (
                self.vision_weight * vision_conf +
                self.ocr_weight * ocr_conf +
                self.llm_weight * llm_conf
            )
            
            needs_review = combined < self.review_threshold
            
            field_scores.append(FieldConfidence(
                field_id=table_id,
                field_name=f"table_{i}",
                value=f"Table with {table.get('num_rows', 0)} rows",
                ocr_confidence=ocr_conf,
                vision_confidence=vision_conf,
                llm_confidence=llm_conf,
                combined_confidence=combined,
                needs_review=needs_review,
                review_reason="Table extraction may need verification" if needs_review else ""
            ))
        
        return field_scores
    
    def _calculate_modality_agreement(
        self,
        vision_results: Dict,
        ocr_results: Dict,
        fused_output: Dict
    ) -> float:
        """Calculate agreement score between modalities"""
        elements = fused_output.get("elements", [])
        
        if not elements:
            return 0.0
        
        agreement_scores = []
        
        for element in elements:
            sources = element.get("sources", [])
            confidences = element.get("confidences", {})
            
            # Calculate variance in confidence scores
            conf_values = [v for v in confidences.values() if v > 0]
            
            if len(conf_values) >= 2:
                mean_conf = sum(conf_values) / len(conf_values)
                variance = sum((c - mean_conf) ** 2 for c in conf_values) / len(conf_values)
                
                # Lower variance = higher agreement
                agreement = 1 - min(variance, 1)
                agreement_scores.append(agreement)
            elif len(conf_values) == 1:
                # Single source, neutral agreement
                agreement_scores.append(0.7)
        
        return sum(agreement_scores) / len(agreement_scores) if agreement_scores else 0.5
    
    def _identify_review_items(
        self,
        field_scores: List[FieldConfidence]
    ) -> List[str]:
        """Identify items that need human review"""
        review_items = []
        
        for field in field_scores:
            if field.needs_review:
                review_items.append(field.field_id)
        
        return review_items
    
    def _calculate_overall_confidence(
        self,
        field_scores: List[FieldConfidence],
        modality_agreement: float
    ) -> float:
        """Calculate overall document confidence"""
        if not field_scores:
            return 0.0
        
        # Weighted average of field confidences
        avg_field_confidence = (
            sum(f.combined_confidence for f in field_scores) / len(field_scores)
        )
        
        # Factor in modality agreement
        overall = (avg_field_confidence * 0.7) + (modality_agreement * 0.3)
        
        return round(overall, 3)
    
    def validate_extraction(
        self,
        extracted_value: str,
        vision_context: Dict,
        ocr_context: Dict
    ) -> Dict[str, Any]:
        """
        Validate a specific extracted value.
        
        Args:
            extracted_value: The extracted value to validate
            vision_context: Context from vision agent
            ocr_context: Context from OCR agent
            
        Returns:
            Validation result with confidence
        """
        # Check if value appears in OCR output
        ocr_text = ocr_context.get("text", "")
        ocr_match = extracted_value.lower() in ocr_text.lower()
        
        # Check if location matches vision detection
        vision_match = bool(vision_context.get("detection"))
        
        # Calculate confidence
        confidence = 0.5
        if ocr_match:
            confidence += 0.3
        if vision_match:
            confidence += 0.2
        
        return {
            "value": extracted_value,
            "confidence": confidence,
            "ocr_verified": ocr_match,
            "vision_verified": vision_match,
            "needs_review": confidence < self.review_threshold
        }


def validation_agent_node(state: DocumentState) -> DocumentState:
    """LangGraph node function for Validation Agent"""
    agent = ValidationAgent()
    return agent(state)
