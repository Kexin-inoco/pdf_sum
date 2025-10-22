import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document

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
    
    
    def summarize(self, chunks: List[Document], documents_data: List[Dict] = None) -> tuple[str, List[Dict]]:
        """
        Generate table of contents from document titles.
        Directly extracts titles from documents_data where is_title=True.
        
        Args:
            chunks: List of Document chunks (not used anymore)
            documents_data: List of document data with is_title information
            
        Returns:
            Generated table of contents text
            
        Raises:
            ValueError: If documents_data is empty or no titles found
        """
        if not documents_data:
            raise ValueError("Cannot summarize: documents_data is empty")
        
        # Extract titles directly from documents_data
        titles = []
        for doc in documents_data:
            if doc.get('is_title', False):
                text = doc.get('text', '').strip()
                page = doc.get('page', '')
                
                if text:
                    # Extract only the first line or first few lines for titles
                    # Split by newlines and take the first non-empty line
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    if lines:
                        # Take the first line as the title
                        title_text = lines[0]
                        
                        # If the first line is very short (like "2.1."), try to get the next line too
                        if len(title_text) < 10 and len(lines) > 1:
                            title_text = f"{title_text} {lines[1]}"
                        
                        # Filter out non-title content using both text patterns and font info
                        font_info = doc.get('font_info', [])
                        page_median_size = doc.get('page_median_size', None)
                        if self._is_valid_title(title_text, font_info, page_median_size):
                            if len(title_text) > 100:
                                title_text = title_text[:100] + "..."
                            
                            # Add page number if available
                            title_with_page = f"{title_text} (Page {page})" if page else title_text
                            titles.append(title_with_page)
        
        if not titles:
            return "No section titles found in the document."
        
        # Create structured titles data for JSON output
        titles_data = []
        for i, title in enumerate(titles):
            # Extract page number from title if present
            page_num = None
            if " (Page " in title:
                try:
                    page_part = title.split(" (Page ")[1].rstrip(")")
                except:
                    page_part = None
                if page_part and page_part.isdigit():
                    page_num = int(page_part)
            
            # Clean title text (remove page number)
            clean_title = title.split(" (Page ")[0] if " (Page " in title else title
            
            titles_data.append({
                "title": clean_title,
                "page": page_num,
                "original_text": title
            })
        
        # Create prompt for table of contents
        titles_text = "\n".join(f"- {title}" for title in titles)
        prompt = f"""Please format the following section titles into a table of contents in markdown format, keeping the EXACT SAME ORDER as provided.

Do NOT reorder or reorganize the sections. Just format them with proper numbering while maintaining the original sequence.

Include page numbers where available.
If a title appears to be a **person's name** (for example, "David M. Blei" or "John D. Lafferty") and it **appears multiple times**, ignore it completely — do NOT include it in the table of contents.

Section titles (in order):
{titles_text}

Table of Contents:"""
        
        try:
            # Call LLM directly
            response = self.llm.invoke(prompt)
            summary = response.content.strip()
            return summary, titles_data
            
        except Exception as e:
            raise Exception(f"Summarization failed: {str(e)}")
    
    def format_output(
        self,
        summary: str,
        documents_data: List[Dict] = None,
        metadata: Optional[Dict[str, Any]] = None,
        titles_data: List[Dict] = None
    ) -> Dict[str, Any]:

        # Calculate metadata from documents_data
        total_pages = 0
        if documents_data:
            total_pages = len(set(doc.get('page', 0) for doc in documents_data))
        
        output = {
            "ai_generated_toc": summary,
            "extracted_titles": titles_data or [],
            "metadata": {
                "total_pages": total_pages,
                "titles_found": len([d for d in documents_data or [] if d.get('is_title', False)]),
                "model": self.model,
                "temperature": self.temperature,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        # Add custom metadata if provided
        if metadata:
            output["metadata"].update(metadata)
        
        return output
    
    def _is_valid_title(self, title: str, font_info: List[Dict] = None, page_median_size: float = None) -> bool:
        """
        Check if text is a valid title (not author info, references, etc.).
        Uses both text patterns and font information for validation.
        
        Args:
            title: Text to validate
            font_info: Font information for the text
            page_median_size: Median font size for the page
            
        Returns:
            True if text appears to be a valid title
        """
        title = title.strip()
        
        # Basic length checks
        if len(title) < 3 or len(title) > 200:
            return False
        
        # Special case: Standard section titles should always be valid
        standard_sections = [
            'Abstract', 'Introduction', 'Related Work', 'Background', 
            'Methods', 'Materials and Methods', 'Experiments', 'Results', 
            'Discussion', 'Conclusion', 'Conclusions', 'References', 
            'Bibliography', 'Acknowledgments', 'Acknowledgements'
        ]
        
        # Check both exact case and case-insensitive
        if title in standard_sections or title.lower() in [s.lower() for s in standard_sections]:
            return True
        
        # Filter out common non-title patterns (but keep it minimal)
        non_title_patterns = [
            r'^As in \[',  # "As in [19] and unlike [18]"
        ]
        
        for pattern in non_title_patterns:
            if re.match(pattern, title):
                return False
        
        # Check for too many numbers (likely not a title)
        if len(re.findall(r'\d+', title)) > 3:
            return False
        
        # Check for too many special characters
        if len(re.findall(r'[^\w\s\.\-\(\)]', title)) > 5:
            return False
        
        # Check for very long sentences (likely not titles)
        # But be more lenient with section numbers like "2.1. Title"
        sentence_parts = title.split('.')
        if len(sentence_parts) > 3:  # Allow up to 3 parts (like "2.1. Title")
            return False
        
        # Font-based validation (if font info is available)
        if font_info and page_median_size:
            # Check if numbers in the title have appropriate font characteristics
            if self._has_number_font_issues(title, font_info, page_median_size):
                return False
        
        return True
    
    def _has_number_font_issues(self, title: str, font_info: List[Dict], page_median_size: float) -> bool:
        """
        Check if numbers in the title have font issues that suggest it's not a real title.
        This is now much more lenient - only filters out very obvious non-titles.
        
        Args:
            title: Title text
            font_info: Font information for the text
            page_median_size: Median font size for the page
            
        Returns:
            True if there are font issues, False otherwise
        """
        if not font_info or not page_median_size:
            return False
        
        # Find numbers in the title
        numbers = re.findall(r'\d+', title)
        if not numbers:
            return False
        
        # Only filter out if numbers are VERY small (less than 70% of median)
        # This catches things like page numbers or footnotes that got mixed in
        for span_info in font_info:
            text = span_info.get('text', '')
            size = span_info.get('size', 0)
            
            # If this span contains numbers
            if any(num in text for num in numbers):
                # Only filter if numbers are VERY small (likely footnotes/page numbers)
                if size < page_median_size * 0.7:  # Much more lenient threshold
                    return True
        
        return False
    
    def __repr__(self) -> str:
        """String representation of summarizer."""
        return (
            f"Summarizer(model={self.model}, "
            f"temperature={self.temperature})"
        )

