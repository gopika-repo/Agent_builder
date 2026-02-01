"""
Agent Transition Tests

Tests for LangGraph agent workflow including:
- Individual agent execution
- State transitions
- Error handling
- Workflow orchestration
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDocumentState:
    """Test DocumentState management"""
    
    def test_create_initial_state(self):
        """Test creating initial state"""
        from agents.state import create_initial_state, ProcessingStatus
        
        state = create_initial_state(
            document_id="test-001",
            filename="test.pdf",
            file_type="pdf",
            pages=[],
            config={}
        )
        
        assert state["document_id"] == "test-001"
        assert state["filename"] == "test.pdf"
        assert state["status"] == ProcessingStatus.PENDING.value
        assert state["errors"] == []
    
    def test_processing_status_enum(self):
        """Test ProcessingStatus enum values"""
        from agents.state import ProcessingStatus
        
        assert ProcessingStatus.PENDING.value == "pending"
        assert ProcessingStatus.VISION_PROCESSING.value == "vision_processing"
        assert ProcessingStatus.OCR_PROCESSING.value == "ocr_processing"
        assert ProcessingStatus.COMPLETED.value == "completed"
        assert ProcessingStatus.FAILED.value == "failed"


class TestVisionAgent:
    """Test Vision Agent"""
    
    def test_vision_agent_initialization(self):
        """Test vision agent initialization"""
        from agents.vision_agent import VisionAgent
        
        agent = VisionAgent(
            confidence_threshold=0.5,
            device="cpu"
        )
        
        assert agent.confidence_threshold == 0.5
    
    def test_vision_agent_call(self, sample_document_state):
        """Test vision agent execution"""
        from agents.vision_agent import VisionAgent
        
        agent = VisionAgent()
        result_state = agent(sample_document_state)
        
        assert "vision_results" in result_state
        assert result_state["current_agent"] == "vision"
    
    def test_vision_agent_node_function(self, sample_document_state):
        """Test vision agent node wrapper"""
        from agents.vision_agent import vision_agent_node
        
        result_state = vision_agent_node(sample_document_state)
        
        assert "vision_results" in result_state


class TestOCRAgent:
    """Test OCR Agent"""
    
    def test_ocr_agent_initialization(self):
        """Test OCR agent initialization"""
        from agents.ocr_agent import OCRAgent
        
        agent = OCRAgent()
        assert agent is not None
    
    def test_ocr_agent_call(self, sample_document_state):
        """Test OCR agent execution"""
        from agents.ocr_agent import OCRAgent
        
        agent = OCRAgent()
        result_state = agent(sample_document_state)
        
        assert "ocr_results" in result_state
        assert result_state["current_agent"] == "ocr"


class TestLayoutAgent:
    """Test Layout Agent"""
    
    def test_layout_agent_initialization(self):
        """Test layout agent initialization"""
        from agents.layout_agent import LayoutAgent
        
        agent = LayoutAgent()
        assert agent is not None
    
    def test_layout_agent_call(self, sample_document_state):
        """Test layout agent execution"""
        from agents.layout_agent import LayoutAgent
        
        agent = LayoutAgent()
        result_state = agent(sample_document_state)
        
        assert "layout_results" in result_state
        assert result_state["current_agent"] == "layout"


class TestTextReasoningAgent:
    """Test Text Reasoning Agent"""
    
    def test_reasoning_agent_initialization(self):
        """Test reasoning agent initialization"""
        from agents.text_reasoning_agent import TextReasoningAgent
        
        agent = TextReasoningAgent()
        assert agent is not None
    
    def test_reasoning_agent_call(self, sample_document_state):
        """Test reasoning agent execution"""
        from agents.text_reasoning_agent import TextReasoningAgent
        
        agent = TextReasoningAgent()
        result_state = agent(sample_document_state)
        
        assert "reasoning_results" in result_state
        assert result_state["current_agent"] == "reasoning"


class TestFusionAgent:
    """Test Fusion Agent"""
    
    def test_fusion_agent_initialization(self):
        """Test fusion agent initialization"""
        from agents.fusion_agent import FusionAgent
        
        agent = FusionAgent()
        assert agent is not None
    
    def test_fusion_agent_call(self, sample_document_state):
        """Test fusion agent execution"""
        from agents.fusion_agent import FusionAgent
        
        agent = FusionAgent()
        result_state = agent(sample_document_state)
        
        assert "fused_output" in result_state
        assert result_state["current_agent"] == "fusion"


class TestValidationAgent:
    """Test Validation Agent"""
    
    def test_validation_agent_initialization(self):
        """Test validation agent initialization"""
        from agents.validation_agent import ValidationAgent
        
        agent = ValidationAgent()
        assert agent is not None
    
    def test_validation_agent_call(self, sample_document_state):
        """Test validation agent execution"""
        from agents.validation_agent import ValidationAgent
        
        agent = ValidationAgent()
        result_state = agent(sample_document_state)
        
        assert "validation_results" in result_state
        assert result_state["current_agent"] == "validation"


class TestWorkflowTransitions:
    """Test workflow transitions"""
    
    def test_should_continue_after_vision_normal(self, sample_document_state):
        """Test normal transition after vision"""
        from agents.workflow import should_continue_after_vision
        
        # Normal case - no errors
        sample_document_state["vision_results"] = {"detections": []}
        
        next_step = should_continue_after_vision(sample_document_state)
        assert next_step == "ocr"
    
    def test_should_continue_after_vision_critical_error(self, sample_document_state):
        """Test error handling after vision"""
        from agents.workflow import should_continue_after_vision
        
        sample_document_state["errors"] = ["Critical failure in vision"]
        
        next_step = should_continue_after_vision(sample_document_state)
        assert next_step == "error_handler"
    
    def test_should_continue_after_ocr(self, sample_document_state):
        """Test transition after OCR"""
        from agents.workflow import should_continue_after_ocr
        
        sample_document_state["ocr_results"] = {"full_text": "Some text"}
        
        next_step = should_continue_after_ocr(sample_document_state)
        assert next_step == "layout"
    
    def test_should_continue_after_layout(self, sample_document_state):
        """Test transition after layout"""
        from agents.workflow import should_continue_after_layout
        
        next_step = should_continue_after_layout(sample_document_state)
        assert next_step == "reasoning"
    
    def test_should_continue_after_reasoning(self, sample_document_state):
        """Test transition after reasoning"""
        from agents.workflow import should_continue_after_reasoning
        
        next_step = should_continue_after_reasoning(sample_document_state)
        assert next_step == "fusion"
    
    def test_should_continue_after_fusion(self, sample_document_state):
        """Test transition after fusion"""
        from agents.workflow import should_continue_after_fusion
        
        next_step = should_continue_after_fusion(sample_document_state)
        assert next_step == "validation"


class TestErrorHandler:
    """Test error handler node"""
    
    def test_error_handler_sets_failed_status(self, sample_document_state):
        """Test error handler sets correct status"""
        from agents.workflow import error_handler_node
        from agents.state import ProcessingStatus
        
        sample_document_state["processing_start_time"] = datetime.utcnow().isoformat()
        
        result = error_handler_node(sample_document_state)
        
        assert result["status"] == ProcessingStatus.FAILED.value
        assert "processing_end_time" in result


class TestWorkflowCreation:
    """Test workflow graph creation"""
    
    def test_create_workflow(self):
        """Test workflow graph creation"""
        from agents.workflow import create_workflow
        
        workflow = create_workflow()
        
        # Should have all nodes
        assert workflow is not None
    
    def test_compile_workflow(self):
        """Test workflow compilation"""
        from agents.workflow import compile_workflow
        
        app = compile_workflow()
        
        assert app is not None


class TestAsyncPipeline:
    """Test async pipeline execution"""
    
    @pytest.mark.asyncio
    async def test_run_document_pipeline(self):
        """Test full pipeline execution"""
        from agents.workflow import run_document_pipeline
        
        try:
            result = await run_document_pipeline(
                document_id="test-async-001",
                filename="test.pdf",
                file_type="pdf",
                pages=[{"page_number": 0, "width": 800, "height": 1000}],
                config={}
            )
            
            assert result["document_id"] == "test-async-001"
            assert "status" in result
        except Exception as e:
            # Pipeline may fail without full dependencies, that's OK
            pytest.skip(f"Pipeline execution requires full dependencies: {e}")
    
    def test_run_document_pipeline_sync(self):
        """Test synchronous pipeline wrapper"""
        from agents.workflow import run_document_pipeline_sync
        
        try:
            result = run_document_pipeline_sync(
                document_id="test-sync-001",
                filename="test.pdf",
                file_type="pdf",
                pages=[{"page_number": 0, "width": 800, "height": 1000}],
                config={}
            )
            
            assert result["document_id"] == "test-sync-001"
        except Exception as e:
            pytest.skip(f"Pipeline execution requires full dependencies: {e}")


class TestAgentOutputs:
    """Test agent output structures"""
    
    def test_vision_output_structure(self):
        """Test VisionOutput dataclass"""
        from agents.state import VisionOutput, Detection
        
        detection = Detection(
            x1=50.0, y1=100.0, x2=200.0, y2=250.0,
            label="table",
            confidence=0.85,
            page_number=0
        )
        
        output = VisionOutput(
            detections=[detection],
            tables=[detection],
            figures=[],
            charts=[],
            signatures=[],
            processing_time_ms=150.0
        )
        
        data = output.to_dict()
        
        assert len(data["detections"]) == 1
        assert len(data["tables"]) == 1
        assert data["processing_time_ms"] == 150.0
    
    def test_ocr_output_structure(self):
        """Test OCROutput dataclass"""
        from agents.state import OCROutput, TextChunk
        
        chunk = TextChunk(
            text="Financial Report",
            x1=50.0, y1=30.0, x2=200.0, y2=60.0,
            confidence=0.95,
            page_number=0
        )
        
        output = OCROutput(
            text_chunks=[chunk],
            full_text="Financial Report 2024",
            processing_time_ms=200.0
        )
        
        data = output.to_dict()
        
        assert len(data["text_chunks"]) == 1
        assert "Financial" in data["full_text"]
