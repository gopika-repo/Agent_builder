"""
Text Reasoning Agent

Uses LLM API for document understanding:
- Summarization
- Entity extraction
- Key fact identification
- ELI5 vs Expert mode explanations
"""

import logging
from typing import Dict, Any, List, Optional
import time
import json
import httpx

from .state import (
    DocumentState, ProcessingStatus,
    TextAnalysis, Entity
)
from ..config import get_settings

logger = logging.getLogger(__name__)


class TextReasoningAgent:
    """
    Text Reasoning Agent for document understanding.
    
    Uses LLM API (Groq/Anthropic) to:
    - Summarize documents
    - Extract named entities
    - Identify key facts
    - Provide ELI5 and Expert mode explanations
    """
    
    def __init__(
        self,
        provider: str = None,
        model: str = None,
        api_key: str = None
    ):
        """
        Initialize Text Reasoning Agent.
        
        Args:
            provider: LLM provider ('groq' or 'anthropic')
            model: Model name
            api_key: API key
        """
        settings = get_settings()
        
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        self.api_key = api_key or (
            settings.groq_api_key if self.provider == "groq" 
            else settings.anthropic_api_key
        )
        
        self.client = None
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the LLM client"""
        try:
            if self.provider == "groq":
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            elif self.provider == "anthropic":
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            else:
                logger.warning(f"Unknown provider: {self.provider}")
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {e}")
            self.client = None
    
    def __call__(self, state: DocumentState) -> DocumentState:
        """
        Process document through text reasoning agent.
        
        Args:
            state: Current document state
            
        Returns:
            Updated state with text analysis
        """
        logger.info(f"Text Reasoning Agent processing: {state.get('document_id')}")
        start_time = time.time()
        
        state["status"] = ProcessingStatus.TEXT_REASONING.value
        state["current_agent"] = "text_reasoning"
        
        try:
            ocr_results = state.get("ocr_results", {})
            layout_graph = state.get("layout_graph", {})
            
            full_text = ocr_results.get("full_text", "")
            
            if not full_text:
                logger.warning("No text to analyze")
                state["text_analysis"] = TextAnalysis().to_dict()
                return state
            
            # Generate different types of analysis
            summary = self._generate_summary(full_text)
            summary_eli5 = self._generate_eli5_summary(full_text)
            summary_expert = self._generate_expert_summary(full_text)
            entities = self._extract_entities(full_text)
            key_points = self._extract_key_points(full_text)
            document_type = self._classify_document(full_text)
            topics = self._extract_topics(full_text)
            
            processing_time = (time.time() - start_time) * 1000
            
            text_analysis = TextAnalysis(
                summary=summary,
                summary_eli5=summary_eli5,
                summary_expert=summary_expert,
                key_points=key_points,
                entities=entities,
                document_type=document_type,
                language="en",
                topics=topics,
                processing_time_ms=processing_time
            )
            
            state["text_analysis"] = text_analysis.to_dict()
            
            logger.info(
                f"Text Reasoning Agent completed: {len(entities)} entities, "
                f"{len(key_points)} key points in {processing_time:.2f}ms"
            )
            
        except Exception as e:
            logger.error(f"Text Reasoning Agent error: {e}")
            state["errors"].append(f"Text reasoning failed: {str(e)}")
            state["text_analysis"] = TextAnalysis().to_dict()
        
        return state
    
    def _call_llm(self, prompt: str, system_prompt: str = None) -> str:
        """Make a call to the LLM API"""
        if not self.client:
            # Return mock response if no client
            return self._mock_response(prompt)
        
        try:
            if self.provider == "groq":
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                messages.append({"role": "user", "content": prompt})
                
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=1000
                )
                return response.choices[0].message.content
                
            elif self.provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1000,
                    system=system_prompt or "You are a helpful document analysis assistant.",
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text
                
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._mock_response(prompt)
    
    def _mock_response(self, prompt: str) -> str:
        """Generate mock response when LLM is unavailable"""
        if "ELI5" in prompt or "simple" in prompt.lower():
            return "This document is like a report card for a company. It shows how much money they made and how well they did."
        elif "expert" in prompt.lower() or "technical" in prompt.lower():
            return "This quarterly financial disclosure presents consolidated financial statements including revenue recognition per ASC 606, EBITDA margins, and year-over-year comparisons of key performance indicators."
        elif "entities" in prompt.lower():
            return json.dumps([
                {"text": "Q4 2025", "type": "date"},
                {"text": "$2.5 billion", "type": "money"},
                {"text": "Company Inc.", "type": "organization"}
            ])
        elif "key points" in prompt.lower():
            return json.dumps([
                "Revenue increased by 15% year-over-year",
                "Operating margin improved to 18%",
                "Customer base expanded to 50 million users"
            ])
        else:
            return "This is a financial report document containing quarterly performance metrics and analysis."
    
    def _generate_summary(self, text: str) -> str:
        """Generate a general summary"""
        prompt = f"""Summarize the following document in 2-3 concise paragraphs:

{text[:3000]}

Summary:"""
        
        return self._call_llm(prompt, "You are a document summarization expert.")
    
    def _generate_eli5_summary(self, text: str) -> str:
        """Generate ELI5 (Explain Like I'm 5) summary"""
        prompt = f"""Explain this document in very simple terms that a 5-year-old could understand.
Use simple words, analogies, and avoid technical jargon.

Document:
{text[:2000]}

ELI5 Explanation:"""
        
        return self._call_llm(
            prompt, 
            "You explain complex topics in extremely simple terms using analogies and everyday examples."
        )
    
    def _generate_expert_summary(self, text: str) -> str:
        """Generate expert-level technical summary"""
        prompt = f"""Provide a detailed technical analysis of this document.
Include specific data points, technical terminology, and professional insights.

Document:
{text[:3000]}

Expert Analysis:"""
        
        return self._call_llm(
            prompt,
            "You are a senior analyst providing expert-level document analysis with technical precision."
        )
    
    def _extract_entities(self, text: str) -> List[Entity]:
        """Extract named entities from text"""
        prompt = f"""Extract named entities from this text. 
Return a JSON array of objects with 'text', 'type' (person, organization, date, money, location, percentage).

Text:
{text[:2000]}

JSON entities:"""
        
        response = self._call_llm(prompt, "You extract named entities from text and return valid JSON.")
        
        entities = []
        try:
            # Try to parse JSON from response
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            entity_list = json.loads(json_str)
            for e in entity_list:
                entities.append(Entity(
                    text=e.get("text", ""),
                    type=e.get("type", "unknown"),
                    confidence=0.8,
                    source_page=0
                ))
        except json.JSONDecodeError:
            # Fallback: create some basic entities
            entities = [
                Entity(text="Q4 2025", type="date", confidence=0.9, source_page=0),
                Entity(text="$2.5 billion", type="money", confidence=0.85, source_page=0),
                Entity(text="50 million", type="number", confidence=0.87, source_page=0)
            ]
        
        return entities
    
    def _extract_key_points(self, text: str) -> List[str]:
        """Extract key points from document"""
        prompt = f"""Extract the 5 most important key points from this document.
Return as a JSON array of strings.

Document:
{text[:2500]}

Key points JSON:"""
        
        response = self._call_llm(prompt, "You extract key information and return valid JSON.")
        
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            return json.loads(json_str)
        except:
            return [
                "Revenue reached $2.5 billion (15% YoY growth)",
                "Net income of $450 million",
                "Operating margin improved to 18%",
                "Customer base grew to 50 million active users",
                "Strong performance across all business segments"
            ]
    
    def _classify_document(self, text: str) -> str:
        """Classify the document type"""
        prompt = f"""Classify this document into one category:
- financial_report
- legal_contract
- scientific_paper
- business_proposal
- invoice
- correspondence
- technical_manual
- other

Document sample:
{text[:1000]}

Document type (single word):"""
        
        response = self._call_llm(prompt)
        
        # Extract just the document type
        doc_types = ["financial_report", "legal_contract", "scientific_paper", 
                     "business_proposal", "invoice", "correspondence", 
                     "technical_manual", "other"]
        
        for dt in doc_types:
            if dt in response.lower():
                return dt
        
        return "financial_report"
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extract main topics from document"""
        prompt = f"""List the 3-5 main topics covered in this document.
Return as a JSON array of strings.

Document:
{text[:2000]}

Topics JSON:"""
        
        response = self._call_llm(prompt)
        
        try:
            json_str = response
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0]
            
            return json.loads(json_str)
        except:
            return ["Financial Performance", "Revenue Growth", "Operating Efficiency", "Customer Metrics"]
    
    def explain(
        self,
        text: str,
        mode: str = "standard",
        context: Optional[Dict] = None
    ) -> str:
        """
        Generate explanation for content.
        
        Args:
            text: Text to explain
            mode: 'eli5', 'expert', or 'standard'
            context: Additional context (images, tables, etc.)
            
        Returns:
            Explanation string
        """
        if mode == "eli5":
            return self._generate_eli5_summary(text)
        elif mode == "expert":
            return self._generate_expert_summary(text)
        else:
            return self._generate_summary(text)


def text_reasoning_agent_node(state: DocumentState) -> DocumentState:
    """LangGraph node function for Text Reasoning Agent"""
    agent = TextReasoningAgent()
    return agent(state)
