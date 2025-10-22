"""
Simple summarization module using direct LLM calls.
No dependency on langchain.chains which doesn't exist in newer versions.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

from config import config


class Summarizer:
    """
    Simple LLM-powered summarizer.
    Uses direct LLM calls instead of deprecated chain modules.
    """
    
    def __init__(
        self,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """
        Initialize summarizer with LLM configuration.
        
        Args:
            model: OpenAI model name (defaults to config value)
            temperature: Sampling temperature (defaults to config value)
            max_tokens: Maximum tokens for response (defaults to config value)
        """
        self.model = model or config.llm_model
        self.temperature = temperature or config.temperature
        self.max_tokens = max_tokens or config.max_tokens
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            openai_api_key=config.openai_api_key
        )
    
    def summarize(self, chunks: List[Document]) -> str:
        """
        Generate summary from document chunks using simple approach.
        
        Args:
            chunks: List of Document chunks to summarize
            
        Returns:
            Generated summary text
            
        Raises:
            ValueError: If chunks is empty or invalid
        """
        if not chunks:
            raise ValueError("Cannot summarize empty chunks list")
        
        # Combine first 10 chunks to keep within token limits
        text_parts = [chunk.page_content for chunk in chunks[:10]]
        combined_text = "\n\n".join(text_parts)
        
        # Limit to approximately 4000 characters to stay within token limits
        if len(combined_text) > 4000:
            combined_text = combined_text[:4000] + "..."
        
        # Create prompt
        prompt = f"""Please provide a concise summary (200-300 words) of the following research paper excerpt.

            Structure your summary to include:
            1. Main research problem or objective
            2. Proposed method or approach
            3. Key results and conclusions

            Paper content:
            {combined_text}

            Summary:"""
        
        try:
            # Call LLM directly
            response = self.llm.invoke(prompt)
            summary = response.content.strip()
            return summary
            
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")
    
    def format_output(
        self,
        summary: str,
        chunks: List[Document],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Format summary with metadata into structured output.
        
        Args:
            summary: Generated summary text
            chunks: Original document chunks
            metadata: Additional metadata to include
            
        Returns:
            Structured dictionary with summary and metadata
        """
        output = {
            "summary": summary,
            "metadata": {
                "source_file": chunks[0].metadata.get('source_file', 'unknown') if chunks else None,
                "total_pages": len(set(c.metadata.get('page', 0) for c in chunks)),
                "chunks_processed": len(chunks),
                "model": self.model,
                "temperature": self.temperature,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add custom metadata if provided
        if metadata:
            output["metadata"].update(metadata)
        
        return output
    
    def __repr__(self) -> str:
        """String representation of summarizer."""
        return (
            f"Summarizer(model={self.model}, "
            f"temperature={self.temperature})"
        )

