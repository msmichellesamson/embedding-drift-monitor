import asyncio
import logging
import signal
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

@dataclass
class BatchConfig:
    batch_size: int = 100
    batch_timeout: float = 5.0
    max_workers: int = 4
    retry_attempts: int = 3
    backoff_factor: float = 1.5

class BatchProcessor:
    def __init__(self, config: BatchConfig):
        self.config = config
        self._shutdown_event = asyncio.Event()
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._setup_signal_handlers()
        
    def _setup_signal_handlers(self):
        """Setup graceful shutdown on SIGTERM/SIGINT"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()
            
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def process_embeddings_batch(self, embeddings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process batch of embeddings with error handling and retry logic"""
        if not embeddings:
            return {"processed": 0, "errors": []}
            
        batch_results = {"processed": 0, "errors": []}
        
        # Split into smaller chunks for parallel processing
        chunks = [embeddings[i:i + self.config.batch_size] 
                 for i in range(0, len(embeddings), self.config.batch_size)]
        
        futures = []
        for chunk in chunks:
            if self._shutdown_event.is_set():
                logger.warning("Shutdown requested, stopping batch processing")
                break
                
            future = self._executor.submit(self._process_chunk_with_retry, chunk)
            futures.append(future)
            
        # Collect results with timeout
        for future in as_completed(futures, timeout=self.config.batch_timeout * len(chunks)):
            try:
                chunk_result = future.result()
                batch_results["processed"] += chunk_result.get("processed", 0)
                batch_results["errors"].extend(chunk_result.get("errors", []))
            except Exception as e:
                logger.error(f"Chunk processing failed: {e}")
                batch_results["errors"].append(str(e))
                
        return batch_results
    
    def _process_chunk_with_retry(self, chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process chunk with exponential backoff retry"""
        for attempt in range(self.config.retry_attempts):
            try:
                return self._process_chunk(chunk)
            except Exception as e:
                if attempt == self.config.retry_attempts - 1:
                    logger.error(f"Chunk processing failed after {self.config.retry_attempts} attempts: {e}")
                    return {"processed": 0, "errors": [str(e)]}
                    
                wait_time = self.config.backoff_factor ** attempt
                logger.warning(f"Chunk processing attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
                
        return {"processed": 0, "errors": ["Max retries exceeded"]}
    
    def _process_chunk(self, chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process individual chunk - override in subclasses"""
        # Simulate processing
        logger.info(f"Processing chunk of {len(chunk)} embeddings")
        return {"processed": len(chunk), "errors": []}
    
    async def shutdown(self):
        """Graceful shutdown with connection cleanup"""
        logger.info("Shutting down batch processor...")
        self._shutdown_event.set()
        self._executor.shutdown(wait=True)
        logger.info("Batch processor shutdown complete")