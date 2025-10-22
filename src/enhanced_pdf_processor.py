"""
Enhanced PDF processing module using PyMuPDF.
Provides better text extraction with structural information.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Optional
from langchain_core.documents import Document

from semantic_splitter import AcademicPaperSplitter


class EnhancedPDFProcessor:
    """
    Enhanced PDF processor using PyMuPDF for better structural analysis.
    """
    
    def __init__(self, use_semantic_splitting: bool = True):
        """
        Initialize enhanced PDF processor.
        
        Args:
            use_semantic_splitting: Whether to use semantic splitting
        """
        self.use_semantic_splitting = use_semantic_splitting
        self.semantic_splitter = AcademicPaperSplitter()
    
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
                            is_title = self._is_likely_title(block_fonts)
                            
                            structured_blocks.append({
                                'text': block_text.strip(),
                                'is_title': is_title,
                                'font_info': block_fonts,
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
    
    def _is_likely_title(self, font_info: List[Dict]) -> bool:
        """
        Determine if a text block is likely a title based on font information.
        Uses stricter criteria to avoid over-detection.
        
        Args:
            font_info: List of font information dictionaries
            
        Returns:
            True if likely a title
        """
        if not font_info:
            return False
        
        # Calculate average font size
        avg_font_size = sum(span['size'] for span in font_info) / len(font_info)
        
        # Count bold text
        bold_count = sum(1 for span in font_info if span['flags'] & 2**4)
        
        # Get text content
        total_text = ''.join(span['text'] for span in font_info)
        text = total_text.strip()
        text_length = len(text)
        
        # Skip very short text (likely page numbers or noise)
        if text_length <= 3:
            return False
        
        # Skip pure numbers
        if text.replace('.', '').replace('-', '').isdigit():
            return False
        
        # Check for numbered section patterns (required for title)
        numbered_section_patterns = [
            r'^\d+\.?\s+[A-Z]',              # "1. Introduction" or "1 Introduction"
            r'^\d+\.\d+\.?\s+',              # "1.1. Subsection"
            r'^\d+\.\d+\.\d+\.?\s+',         # "1.1.1. Subsubsection"
            r'^\d+\.\d+\.\d+\.\d+\.?\s+',    # "1.1.1.1. Level 4"
            r'^\d+\.\s*\n',                  # "1.\n" followed by text
            r'^\d+\.\d+\.\s*\n',             # "2.1.\n" followed by text
            r'^\d+\.\d+\.\d+\.\s*\n',        # "2.1.3.\n" followed by text
        ]
        
        is_numbered_section = any(
            __import__('re').search(pattern, text)
            for pattern in numbered_section_patterns
        )
        
        # Additional patterns (figures, tables, etc.)
        special_patterns = [
            r'^Figure\s+\d+',
            r'^Table\s+\d+',
            r'^Algorithm\s+\d+',
        ]
        
        is_special_header = any(
            __import__('re').search(pattern, text, __import__('re').IGNORECASE)
            for pattern in special_patterns
        )
        
        # Criteria for being a title
        is_large_font = avg_font_size > 11.5
        is_bold = bold_count > len(font_info) * 0.5
        is_reasonable_length = 5 < text_length < 200
        
        # MUST have numbered section pattern OR special header pattern
        if is_numbered_section or is_special_header:
            # Additional check: should have reasonable formatting
            return is_reasonable_length and (is_large_font or is_bold)
        
        return False
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using enhanced structural information.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects
        """
        if not documents:
            return []
        
        chunks = self.semantic_splitter.split_documents(documents)
        print(f"  Using enhanced semantic splitting: {len(chunks)} semantic chunks created")
        
        return chunks
    
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
        
        # Split into chunks
        chunks = self.split_documents(documents)
        
        return chunks
    
    def get_statistics(self, chunks: List[Document]) -> Dict:
        """
        Get processing statistics for chunks.
        
        Args:
            chunks: List of Document chunks
            
        Returns:
            Dictionary with statistics
        """
        if not chunks:
            return {
                "total_chunks": 0,
                "total_characters": 0,
                "avg_chunk_size": 0,
                "pages": 0,
                "title_chunks": 0
            }
        
        total_chars = sum(len(chunk.page_content) for chunk in chunks)
        pages = set()
        title_chunks = 0
        
        for chunk in chunks:
            if 'page' in chunk.metadata:
                pages.add(chunk.metadata['page'])
            if chunk.metadata.get('is_title', False):
                title_chunks += 1
        
        stats = {
            "total_chunks": len(chunks),
            "total_characters": total_chars,
            "avg_chunk_size": total_chars // len(chunks) if chunks else 0,
            "pages": len(pages),
            "title_chunks": title_chunks,
            "source_file": chunks[0].metadata.get('source_file', 'unknown') if chunks else None
        }
        
        # Add semantic splitting statistics if available
        if chunks and chunks[0].metadata.get('is_semantic_chunk', False):
            semantic_stats = self.semantic_splitter.get_chunk_statistics(chunks)
            stats.update({
                "section_distribution": semantic_stats.get('section_distribution', {}),
                "is_semantic_split": True
            })
        else:
            stats["is_semantic_split"] = False
        
        return stats
    
    def __repr__(self) -> str:
        """String representation of processor."""
        return f"EnhancedPDFProcessor(use_semantic_splitting={self.use_semantic_splitting})"
