"""
Simple semantic splitter for academic papers.
Extracts key information: title, authors, year, abstract, introduction, etc.
"""
import re
from typing import List, Dict, Tuple, Optional
from langchain_core.documents import Document


class AcademicPaperSplitter:
    """
    Simple semantic splitter that extracts key paper information.
    """
    
    def __init__(self):
        """Initialize the simple semantic splitter."""
        # Section name patterns for academic papers
        self.SEC_NAMES = [
            r"abstract|summary",
            r"introduction|background", 
            r"related work|literature review",
            r"preliminaries|problem statement",
            r"method|methods|methodology|approach",
            r"experiments?|evaluation|results",
            r"analysis|discussion|ablation",
            r"conclusion(s)?|future work|limitations",
            r"acknowledg(e)?ments",
            r"references|bibliography|appendix|supplementary"
        ]
        
        self.patterns = {
            'title': r'^[A-Z][^.!?]*(?:\.[^.!?]*)*$',
            'authors': r'^[A-Z][a-z]+ [A-Z][a-z]+(?:, [A-Z][a-z]+ [A-Z][a-z]+)*$',
            'year': r'\b(19|20)\d{2}\b',
            'section': r'^\d+\.?\s+[A-Z][A-Za-z\s]+$'
        }
    
    def extract_key_info(self, text: str) -> Dict[str, str]:
        """
        Extract key information from paper text using section patterns.
        
        Args:
            text: Full text of the paper
            
        Returns:
            Dictionary with extracted key information
        """
        lines = text.split('\n')
        key_info = {
            'title': '',
            'authors': '',
            'year': '',
            'sections': {}
        }
        
        current_section = None
        current_content = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                if current_section:
                    current_content.append('')
                continue
            
            # Extract title (usually first substantial line)
            if not key_info['title'] and len(line) > 10 and len(line) < 200:
                if re.match(self.patterns['title'], line):
                    key_info['title'] = line
            
            # Extract authors
            if not key_info['authors'] and re.match(self.patterns['authors'], line):
                key_info['authors'] = line
            
            # Extract year
            year_match = re.search(self.patterns['year'], line)
            if year_match and not key_info['year']:
                key_info['year'] = year_match.group()
            
            # Check for section headers using SEC_NAMES patterns
            section_found = None
            for pattern in self.SEC_NAMES:
                if re.search(pattern, line, re.IGNORECASE):
                    section_found = pattern
                    break
            
            # Also check for numbered sections
            if not section_found and re.match(self.patterns['section'], line):
                section_found = 'numbered_section'
            
            if section_found:
                # Save previous section
                if current_section and current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        key_info['sections'][current_section] = content
                
                # Start new section
                current_section = section_found
                current_content = [line]
            else:
                # Add content to current section
                if current_section:
                    current_content.append(line)
        
        # Save final section
        if current_section and current_content:
            content = '\n'.join(current_content).strip()
            if content:
                key_info['sections'][current_section] = content
        
        return key_info
    
    def create_summary_chunks(self, key_info: Dict[str, str]) -> List[Dict[str, str]]:
        """
        Create chunks for LLM summarization from key information.
        
        Args:
            key_info: Extracted key information
            
        Returns:
            List of chunks ready for LLM processing
        """
        chunks = []
        
        # Title and authors chunk
        if key_info['title'] or key_info['authors'] or key_info['year']:
            title_chunk = []
            if key_info['title']:
                title_chunk.append(f"Title: {key_info['title']}")
            if key_info['authors']:
                title_chunk.append(f"Authors: {key_info['authors']}")
            if key_info['year']:
                title_chunk.append(f"Year: {key_info['year']}")
            
            chunks.append({
                'content': '\n'.join(title_chunk),
                'section_type': 'title_info',
                'section_title': 'Paper Information'
            })
        
        # Create chunks for each section
        for section_pattern, content in key_info['sections'].items():
            # Map section patterns to readable names
            section_name_map = {
                r"abstract|summary": "Abstract",
                r"introduction|background": "Introduction", 
                r"related work|literature review": "Related Work",
                r"preliminaries|problem statement": "Preliminaries",
                r"method|methods|methodology|approach": "Methodology",
                r"experiments?|evaluation|results": "Results",
                r"analysis|discussion|ablation": "Discussion",
                r"conclusion(s)?|future work|limitations": "Conclusion",
                r"acknowledg(e)?ments": "Acknowledgments",
                r"references|bibliography|appendix|supplementary": "References",
                "numbered_section": "Section"
            }
            
            section_title = "Section"
            for pattern, name in section_name_map.items():
                if re.search(pattern, section_pattern, re.IGNORECASE):
                    section_title = name
                    break
            
            chunks.append({
                'content': content,
                'section_type': section_pattern,
                'section_title': section_title
            })
        
        return chunks
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into semantic chunks based on PyMuPDF title detection.
        Uses is_title=True as section boundaries.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of chunked Document objects with semantic metadata
        """
        if not documents:
            return []
        
        all_chunks = []
        
        for doc in documents:
            chunks = self._split_by_title_blocks(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def _split_by_title_blocks(self, doc: Document) -> List[Document]:
        """
        Split document based on PyMuPDF title detection.
        Each chunk starts with is_title=True and includes all following is_title=False blocks.
        
        Args:
            doc: Document with structured_blocks metadata
            
        Returns:
            List of semantic chunks
        """
        chunks = []
        current_chunk = []
        current_title = None
        chunk_index = 0
        
        for block in doc.metadata['structured_blocks']:
            if block['is_title']:
                # Save previous chunk if it has content
                if current_chunk and current_title:
                    chunk_content = '\n'.join(current_chunk).strip()
                    if chunk_content:
                        chunk_doc = Document(
                            page_content=chunk_content,
                            metadata={
                                **doc.metadata,
                                'chunk_index': chunk_index,
                                'section_type': 'title_based_section',
                                'section_title': current_title,
                                'chunk_length': len(chunk_content),
                                'is_semantic_chunk': True,
                                'is_title_based': True
                            }
                        )
                        chunks.append(chunk_doc)
                        chunk_index += 1
                
                # Start new chunk with title
                current_title = block['text'].strip()
                current_chunk = [block['text']]
            else:
                # Add content to current chunk
                if current_title:  # Only add if we have a title
                    current_chunk.append(block['text'])
        
        # Save final chunk
        if current_chunk and current_title:
            chunk_content = '\n'.join(current_chunk).strip()
            if chunk_content:
                chunk_doc = Document(
                    page_content=chunk_content,
                    metadata={
                        **doc.metadata,
                        'chunk_index': chunk_index,
                        'section_type': 'title_based_section',
                        'section_title': current_title,
                        'chunk_length': len(chunk_content),
                        'is_semantic_chunk': True,
                        'is_title_based': True
                    }
                )
                chunks.append(chunk_doc)
        
        return chunks
    
    def get_chunk_statistics(self, chunks: List[Document]) -> Dict[str, any]:
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