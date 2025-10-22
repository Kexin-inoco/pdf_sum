"""
Enhanced semantic splitter for academic papers.
Uses font size analysis and document structure to intelligently chunk content.
"""
import re
from typing import List, Dict, Optional
from langchain_core.documents import Document


class AcademicPaperSplitter:
    """
    Advanced semantic splitter based on font size hierarchy and document structure.
    
    Strategy:
    1. Sort by font size to find title (largest)
    2. Medium fonts near top = authors/metadata
    3. Font size changes = section headers
    4. Continuous same font = body paragraphs
    5. Y-coordinate proximity = same line
    6. Same block continuity = same paragraph
    """
    
    def __init__(self):
        self.min_chunk_length = 100
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents using font-based structure analysis.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of semantic chunks
        """
        if not documents:
            return []
        
        all_chunks = []
        global_chunk_index = 0
        
        for doc in documents:
            chunks = self._split_by_structure(doc, global_chunk_index)
            all_chunks.extend(chunks)
            global_chunk_index += len(chunks)
        
        return all_chunks
    
    def _split_by_structure(self, doc: Document, start_index: int = 0) -> List[Document]:
        """
        Split document based on font hierarchy and structure.
        
        Args:
            doc: Document with structured_blocks metadata
            start_index: Starting chunk index for this document
            
        Returns:
            List of semantic chunks
        """
        blocks = doc.metadata.get('structured_blocks', [])
        if not blocks:
            return []
        
        # Classify each block using PyMuPDF's is_title
        classified_blocks = self._classify_blocks(blocks)
        
        # Group into semantic chunks
        chunks = self._create_semantic_chunks(classified_blocks, doc.metadata, start_index)
        
        return chunks
    
    def _classify_blocks(self, blocks: List[Dict]) -> List[Dict]:
        """
        Classify each block using PyMuPDF's is_title field.
        """
        classified = []
        
        for i, block in enumerate(blocks):
            text = block.get('text', '').strip()
            is_title = block.get('is_title', False)
            
            # Skip empty or very short blocks
            if not text or len(text) < 3:
                continue
            
            # Skip noise (pure numbers, page numbers, etc.)
            if self._is_noise(text):
                continue
            
            # Use PyMuPDF's is_title detection directly
            if is_title:
                block_type = 'section_header'
            else:
                block_type = 'body'
            
            classified.append({
                'text': text,
                'type': block_type,
                'page': block.get('page', 1),
                'is_title': is_title
            })
        
        return classified
    
    def _is_noise(self, text: str) -> bool:
        """Identify noise content like single numbers, table data, etc."""
        # Single numbers or very short text
        if len(text) <= 3 and text.replace('.', '').replace('-', '').isdigit():
            return True
        
        # Mostly numbers and symbols (likely table data)
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count < len(text) * 0.3 and len(text) < 50:
            return True
        
        # Math formulas (lots of special characters)
        special_chars = sum(1 for c in text if c in '∑∫∂αβγδεθλμσω≤≥≠±×÷')
        if special_chars > 3 and len(text) < 50:
            return True
        
        return False
    
    def _create_semantic_chunks(self, classified_blocks: List[Dict], metadata: Dict, start_index: int = 0) -> List[Document]:
        """
        Create semantic chunks by grouping blocks.
        Each section_header starts a new chunk and includes following body blocks.
        """
        chunks = []
        current_section_title = None
        current_content = []
        chunk_index = start_index
        
        for block in classified_blocks:
            block_type = block['type']
            text = block['text']
            
            # Section header starts a new chunk
            if block_type == 'section_header':
                # Save previous chunk if exists
                if current_content:
                    chunk_text = '\n\n'.join(current_content)
                    if len(chunk_text) >= self.min_chunk_length:
                        chunks.append(self._make_chunk(
                            chunk_text, current_section_title, chunk_index, metadata
                        ))
                        chunk_index += 1
                
                # Start new section with numbered title
                current_section_title = text
                current_content = [text]
            
            # Body text - add to current section
            elif block_type == 'body':
                current_content.append(text)
        
        # Save final chunk
        if current_content:
            chunk_text = '\n\n'.join(current_content)
            if len(chunk_text) >= self.min_chunk_length:
                chunks.append(self._make_chunk(
                    chunk_text, current_section_title, chunk_index, metadata
                ))
        
        return chunks
    
    def _make_chunk(self, content: str, title: Optional[str], 
                   index: int, metadata: Dict) -> Document:
        """Create a Document chunk with metadata."""
        # Extract page number if available
        page = metadata.get('page', 1)
        
        return Document(
            page_content=content,
            metadata={
                **{k: v for k, v in metadata.items() if k != 'structured_blocks'},
                'chunk_index': index,
                'section_type': 'semantic_section',
                'section_title': title or 'Untitled Section',
                'chunk_length': len(content),
                'is_semantic_chunk': True,
                'page': page
            }
        )
    
    def get_chunk_statistics(self, chunks: List[Document]) -> Dict:
        """
        Get statistics about semantic chunks.
        
        Args:
            chunks: List of semantic chunks
            
        Returns:
            Dictionary with chunk statistics
        """
        if not chunks:
            return {}
        
        section_types = {}
        total_length = 0
        
        for chunk in chunks:
            section_type = chunk.metadata.get('section_type', 'unknown')
            section_types[section_type] = section_types.get(section_type, 0) + 1
            total_length += len(chunk.page_content)
        
        return {
            'total_chunks': len(chunks),
            'section_distribution': section_types,
            'average_chunk_length': total_length / len(chunks) if chunks else 0,
            'total_length': total_length
        }

