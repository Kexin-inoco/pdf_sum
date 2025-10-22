"""
Enhanced PDF processing module using PyMuPDF.
Provides better text extraction with structural information.
"""
import fitz  # PyMuPDF
import re
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.documents import Document

# No longer needed - we work directly with documents data


class EnhancedPDFProcessor:
    """
    Enhanced PDF processor using PyMuPDF for better structural analysis.
    """
    
    def __init__(self):
        """
        Initialize enhanced PDF processor.
        """
        # No longer needed - we work directly with documents data
    
    def load_pdf_with_structure(self, pdf_path: str) -> List[Document]:
        """
        Load PDF with structural information using PyMuPDF.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of Document objects with structural metadata
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if pdf_path.suffix.lower() != '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        try:
            doc = fitz.open(str(pdf_path))
            documents = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = ""
                structured_blocks = []
                
                # Get text blocks with formatting information
                blocks = page.get_text("dict")["blocks"]
                
                # First pass: collect all font sizes to calculate median
                all_page_sizes = []
                for block in blocks:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                all_page_sizes.append(span["size"])
                
                # Calculate page median font size
                page_median_size = None
                if all_page_sizes:
                    all_page_sizes.sort()
                    page_median_size = all_page_sizes[len(all_page_sizes) // 2]
                
                # Second pass: extract blocks and detect titles
                for block in blocks:
                    if "lines" in block:
                        block_text = ""
                        block_fonts = []
                        
                        for line in block["lines"]:
                            line_text = ""
                            line_fonts = []
                            
                            for span in line["spans"]:
                                text = span["text"]
                                font_size = span["size"]
                                font_flags = span["flags"]  # Bold, italic, etc.
                                
                                line_text += text
                                line_fonts.append({
                                    'size': font_size,
                                    'flags': font_flags,
                                    'text': text
                                })
                            
                            if line_text.strip():
                                block_text += line_text + "\n"
                                block_fonts.extend(line_fonts)
                        
                        if block_text.strip():
                            # Determine if this block is likely a title
                            text_for_judge = block_text.strip()
                            is_title = self._is_likely_title(text_for_judge, block_fonts, page_median_size)
                            
                            structured_blocks.append({
                                'text': block_text.strip(),
                                'is_title': is_title,
                                'page': page_num + 1
                            })
                            
                            page_text += block_text + "\n"
                
                # Special handling for first page: mark first substantial block as title
                if page_num == 0 and structured_blocks:
                    self._mark_first_page_title(structured_blocks)
                
                # Create document with structural metadata
                if page_text.strip():
                    doc_obj = Document(
                        page_content=page_text.strip(),
                        metadata={
                            'source_file': pdf_path.name,
                            'source_path': str(pdf_path),
                            'page': page_num + 1,
                            'structured_blocks': structured_blocks,
                            'has_titles': any(block['is_title'] for block in structured_blocks)
                        }
                    )
                    documents.append(doc_obj)
            
            doc.close()
            return documents
            
        except Exception as e:
            raise IOError(f"Failed to load PDF {pdf_path}: {str(e)}")
    
    def _mark_first_page_title(self, structured_blocks: List[Dict]) -> None:
        """
        Mark the first substantial text block on first page as document title.
        
        Args:
            structured_blocks: List of structured blocks from first page
        """
        for block in structured_blocks:
            text = block.get('text', '').strip()
            
            # Skip very short text or pure numbers
            if len(text) <= 5:
                continue
            
            # Skip page numbers
            if text.replace('.', '').replace('-', '').isdigit():
                continue
            
            # First substantial block is the document title
            if len(text) > 10:
                block['is_title'] = True
                block['is_document_title'] = True
                break
    
    def _is_likely_title(self, text: str, font_info: List[Dict], page_median_size: Optional[float]=None) -> bool:
        """
        Determine if text is likely a section title based on strict criteria.
        
        Args:
            text: Text content to analyze
            font_info: Font information for the text
            page_median_size: Median font size for the page
            
        Returns:
            True if text is likely a title, False otherwise
        """
        t = ' '.join(text.split())
        
        # Basic filters
        if len(t) <= 3 or t.replace('.', '').replace('-', '').isdigit():
            return False
        
        # Filter out common non-title patterns
        non_title_patterns = [
            r'^(example|proof|definition|proposition|theorem|lemma|corollary)\s*\d*',  # Mathematical terms
            r'^(fig\.|figure|table|algorithm)\s+\d+',  # Figure/Table captions
            r'^[a-z]',  # Starts with lowercase
            r'^\d+$',  # Pure numbers
            r'^[A-Za-z]\s*[=<>]',  # Mathematical expressions
            r'^[A-Za-z]\s*[+\-*/]',  # Mathematical operations
            r'^\w+\s*[=<>]\s*\w+',  # Equations
            r'^[A-Za-z]\s*\(',  # Function calls
            r'^\w+\s*:',  # Labels
            r'^[A-Za-z]\s*[0-9]',  # Mixed alphanumeric without proper structure
        ]
        
        for pattern in non_title_patterns:
            if re.match(pattern, t, re.I):
                return False
        
        # 1) Numbered sections (strict pattern)
        numbered_patterns = [
            r'^\d+\s+[A-Z]',  # "1 Introduction"
            r'^\d+\.\s*[A-Z]',  # "1. Introduction" (with optional space after dot)
            r'^\d+\.\d+\s*[A-Z]',  # "1.1 Methods" (with optional space)
            r'^\d+\.\d+\.\s*[A-Z]',  # "1.1. Details" (with optional space after dot)
            r'^\d+\.\d+\.\d+\s*[A-Z]',  # "1.1.1 Details" (with optional space)
            r'^\d+\.\d+\.\d+\.\s*[A-Z]',  # "1.1.1. Specifics" (with optional space after dot)
            r'^\d+\.\d+\.\d+\.\d+\s*[A-Z]',  # "1.1.1.1 Specifics" (with optional space)
            r'^\d+\n[A-Z]',  # "1\nIntroduction" (with line break)
            r'^\d+\.\n[A-Z]',  # "1.\nIntroduction" (with line break)
            r'^\d+\.\d+\n[A-Z]',  # "1.1\nMethods" (with line break)
            r'^\d+\.\d+\.\n[A-Z]',  # "1.1.\nMethods" (with line break)
            r'^\d+\.\d+\.\d+\n[A-Z]',  # "1.1.1\nDetails" (with line break)
        ]
        
        for pattern in numbered_patterns:
            if re.match(pattern, t):
                return True
        
        # 2) Special headers (Figure, Table, Algorithm) - these ARE titles
        special_headers = [
            r'^Figure\s+\d+',
            r'^Table\s+\d+',
            r'^Algorithm\s+\d+',
        ]
        
        for pattern in special_headers:
            if re.match(pattern, t, re.I):
                return True
        
        # 3) Common section names (only if they appear alone)
        section_names = [
            r'^Abstract$',
            r'^Introduction$',
            r'^Related Work$',
            r'^Background$',
            r'^Methods?$',
            r'^Materials? and Methods?$',
            r'^Experiments?$',
            r'^Results?$',
            r'^Discussion$',
            r'^Conclusion$',
            r'^Conclusions?$',
            r'^References$',
            r'^Acknowledgments?$',
            r'^Acknowledgements?$',
        ]
        
        for pattern in section_names:
            if re.match(pattern, t, re.I):
                return True
        
        # 4) Font-based analysis (only for very strong title signals)
        sizes = [s.get('size', 0) for s in font_info] or [0]
        is_large_font = any(s > page_median_size * 1.2 for s in sizes) if page_median_size else False
        is_bold = any(s.get('flags', 0) & 16 for s in font_info)  # Bold flag
        is_reasonable_length = 5 <= len(t) <= 100
        
        # Only use font analysis for very strong title signals
        return (is_large_font or is_bold) and is_reasonable_length

    
    # split_documents method removed - no longer needed
    
    def process(self, pdf_path: str) -> List[Document]:
        """
        Complete processing pipeline with enhanced structural analysis.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of processed Document chunks
            
        Raises:
            Exception: If processing fails at any stage
        """
        # Load PDF with structural information
        documents = self.load_pdf_with_structure(pdf_path)
        
        if not documents:
            raise ValueError(f"No content extracted from PDF: {pdf_path}")
        
        return documents
    
    # get_statistics method removed - no longer needed
    
    def __repr__(self) -> str:
        """String representation of processor."""
        return f"EnhancedPDFProcessor()"
