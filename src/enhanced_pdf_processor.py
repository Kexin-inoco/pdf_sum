"""
Enhanced PDF processing module using PyMuPDF.
Provides better text extraction with structural information.
"""
import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document

from config import config
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
    
    def _is_likely_title(self, font_info: List[Dict]) -> bool:
        """
        Determine if a text block is likely a title based on font information.
        
        Args:
            font_info: List of font information dictionaries
            
        Returns:
            True if likely a title
        """
        if not font_info:
            return False
        
        # Check font size (titles are usually larger)
        avg_font_size = sum(span['size'] for span in font_info) / len(font_info)
        
        bold_count = sum(1 for span in font_info if span['flags'] & 2**4)  # Bold flag
        
        # Check text length (titles are usually shorter)
        total_text = ''.join(span['text'] for span in font_info)
        text_length = len(total_text.strip())
        
        # Title criteria
        is_large_font = avg_font_size > 12
        is_bold = bold_count > len(font_info) * 0.5
        is_short = text_length < 200
        is_short_line = text_length < 100
        
        # Check for common title patterns
        title_patterns = [
            r'^\d+\.?\s+[A-Z]',  # Numbered sections
            r'^(abstract|introduction|methodology|results|conclusion)',
            r'^[A-Z][A-Z\s]+$'  # All caps
        ]
        
        has_title_pattern = any(
            __import__('re').search(pattern, total_text, __import__('re').IGNORECASE)
            for pattern in title_patterns
        )
        
        return (is_large_font and (is_bold or is_short or has_title_pattern)) or is_short_line
    
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
