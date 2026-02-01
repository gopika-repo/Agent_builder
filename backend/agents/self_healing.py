"""
Self-Healing Pipeline

Production-grade error handling with:
- Automatic retry with exponential backoff
- Graceful degradation (fallback strategies)
- Recovery path logging
- Cached results for resilience
"""

import logging
import time
import asyncio
from typing import Dict, Any, Callable, Optional, TypeVar, Generic
from dataclasses import dataclass, field
from functools import wraps
from enum import Enum
import json
from pathlib import Path

logger = logging.getLogger(__name__)

T = TypeVar('T')


class FailureType(Enum):
    """Types of failures that can occur"""
    OCR_FAILURE = "ocr_failure"
    VISION_FAILURE = "vision_failure"
    LLM_TIMEOUT = "llm_timeout"
    LLM_QUOTA = "llm_quota"
    EMBEDDING_FAILURE = "embedding_failure"
    VECTOR_STORE_FAILURE = "vector_store_failure"
    NETWORK_ERROR = "network_error"
    MEMORY_ERROR = "memory_error"
    UNKNOWN = "unknown"


class RecoveryAction(Enum):
    """Recovery actions taken"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CACHE_HIT = "cache_hit"
    DEGRADED_MODE = "degraded_mode"
    SKIP = "skip"
    ABORT = "abort"


@dataclass
class RecoveryEvent:
    """Record of a recovery action"""
    timestamp: float
    failure_type: FailureType
    action: RecoveryAction
    component: str
    original_error: str
    success: bool
    fallback_used: Optional[str] = None
    retry_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "failure_type": self.failure_type.value,
            "action": self.action.value,
            "component": self.component,
            "original_error": self.original_error,
            "success": self.success,
            "fallback_used": self.fallback_used,
            "retry_count": self.retry_count
        }


@dataclass
class PipelineHealth:
    """Overall pipeline health status"""
    is_healthy: bool
    degraded_components: list = field(default_factory=list)
    recovery_events: list = field(default_factory=list)
    cache_stats: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_healthy": self.is_healthy,
            "degraded_components": self.degraded_components,
            "recovery_events": [e.to_dict() for e in self.recovery_events],
            "cache_stats": self.cache_stats
        }


class ResultCache:
    """Simple file-based cache for pipeline results"""
    
    def __init__(self, cache_dir: str = ".cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result"""
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, 'r') as f:
                    self.hits += 1
                    return json.load(f)
            except Exception:
                pass
        self.misses += 1
        return None
    
    def set(self, key: str, value: Dict[str, Any], ttl: int = 3600):
        """Cache a result"""
        cache_file = self.cache_dir / f"{key}.json"
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    "value": value,
                    "timestamp": time.time(),
                    "ttl": ttl
                }, f)
        except Exception as e:
            logger.warning(f"Failed to cache result: {e}")
    
    def get_stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


class SelfHealingPipeline:
    """
    Self-healing wrapper for document processing pipeline.
    
    Features:
    - Automatic retry with exponential backoff
    - Fallback strategies per component
    - Result caching for resilience
    - Health monitoring
    - Recovery logging
    """
    
    def __init__(
        self,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_delay: float = 30.0,
        enable_cache: bool = True,
        cache_dir: str = ".pipeline_cache"
    ):
        """
        Initialize self-healing pipeline.
        
        Args:
            max_retries: Maximum retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries
            enable_cache: Whether to use result caching
            cache_dir: Directory for cache files
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.enable_cache = enable_cache
        self.cache = ResultCache(cache_dir) if enable_cache else None
        self.recovery_events: list = []
        self.degraded_components: set = set()
        
        # Fallback strategies
        self.fallbacks = {
            "ocr": self._ocr_fallback,
            "vision": self._vision_fallback,
            "llm": self._llm_fallback,
            "embedding": self._embedding_fallback,
            "vector_store": self._vector_store_fallback
        }
    
    def execute_with_recovery(
        self,
        component: str,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Execute a component with self-healing capabilities.
        
        Args:
            component: Name of the component (ocr, vision, llm, etc.)
            primary_func: Primary function to execute
            fallback_func: Optional fallback function
            cache_key: Optional cache key for caching results
            **kwargs: Arguments to pass to the function
            
        Returns:
            Result dictionary with recovery metadata
        """
        # Check cache first
        if cache_key and self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                self._log_recovery(
                    FailureType.UNKNOWN,
                    RecoveryAction.CACHE_HIT,
                    component,
                    "N/A",
                    True
                )
                return {
                    "result": cached.get("value"),
                    "from_cache": True,
                    "component": component
                }
        
        # Attempt primary function with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                result = primary_func(**kwargs)
                
                # Cache successful result
                if cache_key and self.cache:
                    self.cache.set(cache_key, result)
                
                # Component recovered if it was degraded
                if component in self.degraded_components:
                    self.degraded_components.remove(component)
                    logger.info(f"Component '{component}' recovered")
                
                return {
                    "result": result,
                    "from_cache": False,
                    "attempts": attempt + 1,
                    "component": component
                }
                
            except Exception as e:
                last_error = e
                failure_type = self._classify_error(e, component)
                
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.base_delay * (2 ** attempt),
                        self.max_delay
                    )
                    logger.warning(
                        f"Attempt {attempt + 1} failed for {component}: {e}. "
                        f"Retrying in {delay:.1f}s..."
                    )
                    self._log_recovery(
                        failure_type,
                        RecoveryAction.RETRY,
                        component,
                        str(e),
                        False,
                        retry_count=attempt + 1
                    )
                    time.sleep(delay)
        
        # All retries failed, try fallback
        logger.error(f"All {self.max_retries} attempts failed for {component}")
        
        # Try provided fallback first
        if fallback_func:
            try:
                result = fallback_func(**kwargs)
                self._log_recovery(
                    self._classify_error(last_error, component),
                    RecoveryAction.FALLBACK,
                    component,
                    str(last_error),
                    True,
                    fallback_used="provided_fallback"
                )
                self.degraded_components.add(component)
                return {
                    "result": result,
                    "fallback": True,
                    "component": component,
                    "degraded": True
                }
            except Exception as fallback_error:
                logger.warning(f"Provided fallback also failed: {fallback_error}")
        
        # Try built-in fallback
        if component in self.fallbacks:
            try:
                result = self.fallbacks[component](**kwargs)
                self._log_recovery(
                    self._classify_error(last_error, component),
                    RecoveryAction.FALLBACK,
                    component,
                    str(last_error),
                    True,
                    fallback_used=f"builtin_{component}_fallback"
                )
                self.degraded_components.add(component)
                return {
                    "result": result,
                    "fallback": True,
                    "component": component,
                    "degraded": True
                }
            except Exception as builtin_error:
                logger.error(f"Built-in fallback failed: {builtin_error}")
        
        # Complete failure
        self._log_recovery(
            self._classify_error(last_error, component),
            RecoveryAction.ABORT,
            component,
            str(last_error),
            False
        )
        self.degraded_components.add(component)
        
        return {
            "result": None,
            "error": str(last_error),
            "component": component,
            "failed": True
        }
    
    async def execute_with_recovery_async(
        self,
        component: str,
        primary_func: Callable,
        fallback_func: Optional[Callable] = None,
        cache_key: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Async version of execute_with_recovery"""
        # Check cache first
        if cache_key and self.cache:
            cached = self.cache.get(cache_key)
            if cached:
                return {
                    "result": cached.get("value"),
                    "from_cache": True,
                    "component": component
                }
        
        last_error = None
        for attempt in range(self.max_retries):
            try:
                if asyncio.iscoroutinefunction(primary_func):
                    result = await primary_func(**kwargs)
                else:
                    result = primary_func(**kwargs)
                
                if cache_key and self.cache:
                    self.cache.set(cache_key, result)
                
                return {
                    "result": result,
                    "from_cache": False,
                    "attempts": attempt + 1,
                    "component": component
                }
                
            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    delay = min(self.base_delay * (2 ** attempt), self.max_delay)
                    await asyncio.sleep(delay)
        
        # Try fallbacks
        if fallback_func:
            try:
                if asyncio.iscoroutinefunction(fallback_func):
                    result = await fallback_func(**kwargs)
                else:
                    result = fallback_func(**kwargs)
                return {"result": result, "fallback": True, "component": component}
            except Exception:
                pass
        
        return {"result": None, "error": str(last_error), "failed": True}
    
    def _classify_error(self, error: Exception, component: str) -> FailureType:
        """Classify an error into a failure type"""
        error_str = str(error).lower()
        
        if "timeout" in error_str:
            return FailureType.LLM_TIMEOUT
        elif "quota" in error_str or "rate limit" in error_str:
            return FailureType.LLM_QUOTA
        elif "memory" in error_str or "oom" in error_str:
            return FailureType.MEMORY_ERROR
        elif "connection" in error_str or "network" in error_str:
            return FailureType.NETWORK_ERROR
        elif component == "ocr":
            return FailureType.OCR_FAILURE
        elif component == "vision":
            return FailureType.VISION_FAILURE
        elif component == "embedding":
            return FailureType.EMBEDDING_FAILURE
        elif component == "vector_store":
            return FailureType.VECTOR_STORE_FAILURE
        else:
            return FailureType.UNKNOWN
    
    def _log_recovery(
        self,
        failure_type: FailureType,
        action: RecoveryAction,
        component: str,
        error: str,
        success: bool,
        fallback_used: Optional[str] = None,
        retry_count: int = 0
    ):
        """Log a recovery event"""
        event = RecoveryEvent(
            timestamp=time.time(),
            failure_type=failure_type,
            action=action,
            component=component,
            original_error=error[:200],  # Truncate long errors
            success=success,
            fallback_used=fallback_used,
            retry_count=retry_count
        )
        self.recovery_events.append(event)
        
        # Keep only last 100 events
        if len(self.recovery_events) > 100:
            self.recovery_events = self.recovery_events[-100:]
    
    def _ocr_fallback(self, **kwargs) -> Dict[str, Any]:
        """Fallback OCR strategy"""
        logger.info("Using OCR fallback: basic image-to-text")
        
        # Return minimal OCR result
        return {
            "text_blocks": [],
            "full_text": "[OCR unavailable - text extraction failed]",
            "fallback": True,
            "confidence": 0.0
        }
    
    def _vision_fallback(self, **kwargs) -> Dict[str, Any]:
        """Fallback vision strategy using heuristics"""
        logger.info("Using Vision fallback: heuristic layout detection")
        
        # Return minimal detection result
        return {
            "detections": [],
            "tables": [],
            "figures": [],
            "fallback": True,
            "confidence": 0.0
        }
    
    def _llm_fallback(self, **kwargs) -> Dict[str, Any]:
        """Fallback LLM strategy using cached or template response"""
        logger.info("Using LLM fallback: cached reasoning")
        
        query = kwargs.get("query", "")
        context = kwargs.get("context", "")
        
        # Generate template response
        return {
            "response": f"Based on the document context: {context[:500]}...\n\n"
                       f"[Note: AI reasoning temporarily unavailable. "
                       f"Showing extracted content only.]",
            "fallback": True,
            "confidence": 0.3
        }
    
    def _embedding_fallback(self, **kwargs) -> Dict[str, Any]:
        """Fallback embedding strategy using simple hashing"""
        logger.info("Using Embedding fallback: TF-IDF or hash-based")
        
        text = kwargs.get("text", "")
        
        # Simple hash-based "embedding"
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        # Convert to 384-dim vector (simplified)
        embedding = [float(b) / 255.0 for b in hash_bytes] * 24
        
        return {
            "embedding": embedding,
            "fallback": True,
            "method": "hash_based"
        }
    
    def _vector_store_fallback(self, **kwargs) -> Dict[str, Any]:
        """Fallback vector store strategy using in-memory search"""
        logger.info("Using Vector Store fallback: in-memory similarity")
        
        return {
            "results": [],
            "fallback": True,
            "note": "Vector store unavailable"
        }
    
    def get_health(self) -> PipelineHealth:
        """Get current pipeline health status"""
        return PipelineHealth(
            is_healthy=len(self.degraded_components) == 0,
            degraded_components=list(self.degraded_components),
            recovery_events=self.recovery_events[-10:],  # Last 10 events
            cache_stats=self.cache.get_stats() if self.cache else {}
        )
    
    def reset_health(self):
        """Reset health status (after manual intervention)"""
        self.degraded_components.clear()
        self.recovery_events.clear()
        if self.cache:
            self.cache.hits = 0
            self.cache.misses = 0


def with_self_healing(
    component: str,
    max_retries: int = 3,
    fallback_func: Optional[Callable] = None
):
    """
    Decorator for adding self-healing to any function.
    
    Usage:
        @with_self_healing("ocr", max_retries=3)
        def extract_text(image):
            ...
    """
    pipeline = SelfHealingPipeline(max_retries=max_retries)
    
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return pipeline.execute_with_recovery(
                component=component,
                primary_func=lambda: func(*args, **kwargs),
                fallback_func=fallback_func
            )
        return wrapper
    return decorator
