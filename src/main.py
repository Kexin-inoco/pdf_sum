"""
Main application entry point.
Orchestrates PDF processing and summarization pipeline.
"""
import json
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from enhanced_pdf_processor import EnhancedPDFProcessor
from config import config
from summarizer import Summarizer
from langchain_core.documents import Document


class PDFSummarizationPipeline:
    """
    Main pipeline that coordinates PDF processing and summarization.
    """
    
    def __init__(
        self,
        input_dir: str = "/app/data/input",
        output_dir: str = "/app/data/output"
    ):
        """
        Initialize pipeline.
        
        Args:
            input_dir: Directory containing PDF files
            output_dir: Directory for output summaries
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.processor = EnhancedPDFProcessor()
        self.summarizer = Summarizer()
        
        print("Pipeline initialized:")
        print(f"  Input dir: {self.input_dir}")
        print(f"  Output dir: {self.output_dir}")
        print(f"  Processor: {self.processor}")
        print(f"  Summarizer: {self.summarizer}")
    
    def find_pdf_files(self) -> List[Path]:
        """
        Find all PDF files in input directory.
        
        Returns:
            List of PDF file paths
        """
        if not self.input_dir.exists():
            print(f"Warning: Input directory does not exist: {self.input_dir}")
            return []
        
        return sorted(self.input_dir.glob("*.pdf"))
    
    def save_json(self, data: Dict, output_path: Path):
        """Save data as JSON file."""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    
    def process_single_pdf(self, pdf_path: Path) -> Dict:
        """
        Process a single PDF file through the complete pipeline.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary with summary and metadata
            
        Raises:
            Exception: If processing fails
        """
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*60}")
        
        # Step 1: Extract and chunk text
        print("\n[1/3] Extracting and chunking text...")
        documents = self.processor.load_pdf_with_structure(pdf_path)
        chunks = self.processor.split_documents(documents)
        
        stats = self.processor.get_statistics(chunks)
        print(f"  ✓ Extracted {stats['pages']} pages")
        print(f"  ✓ Created {stats['total_chunks']} chunks")
        print(f"  ✓ Average chunk size: {stats['avg_chunk_size']} characters")
        
        # Step 2: Generate summary
        print("\n[2/3] Generating summary with LLM...")
        print(f"  Model: {self.summarizer.model}")
        
        summary = self.summarizer.summarize(chunks)

        
        print(f"  ✓ Summary generated ({len(summary)} characters)")
        
        # Step 3: Format output
        print("\n[3/3] Formatting output...")
        result = self.summarizer.format_output(summary, chunks)
        
        # Store chunks and documents for potential saving
        self._last_chunks = chunks
        self._last_documents = documents
        
        return result
    
    def save_outputs(self, result: Dict, pdf_name: str, chunks: List[Document] = None, documents: List[Document] = None):
        """
        Save results in configured output formats.
        
        Args:
            result: Summary result dictionary
            pdf_name: Original PDF filename
            chunks: List of document chunks (optional)
            documents: List of original documents (optional)
        """
        base_name = Path(pdf_name).stem
        
        # Save summary
        output_path = self.output_dir / f"{base_name}_summary.json"
        self.save_json(result, output_path)
        print(f"  ✓ Saved JSON: {output_path.name}")
        
        # Save semantic chunks if available
        if chunks and chunks[0].metadata.get('is_semantic_chunk', False):
            chunks_data = []
            for chunk in chunks:
                chunks_data.append({
                    'chunk_index': chunk.metadata.get('chunk_index', 0),
                    'section_type': chunk.metadata.get('section_type', 'unknown'),
                    'section_title': chunk.metadata.get('section_title', 'Untitled'),
                    'content': chunk.page_content,
                    'chunk_length': len(chunk.page_content),
                    'page': chunk.metadata.get('page', 'unknown')
                })
            
            chunks_output = {
                'source_file': pdf_name,
                'total_chunks': len(chunks),
                'chunks': chunks_data,
                'section_distribution': self._get_section_distribution(chunks),
                'timestamp': result['metadata']['timestamp']
            }
            
            chunks_path = self.output_dir / f"{base_name}_chunks.json"
            self.save_json(chunks_output, chunks_path)
            print(f"  ✓ Saved semantic chunks: {chunks_path.name}")
        
        # Save original documents if available
        if documents:
            documents_data = []
            for i, doc in enumerate(documents):
                doc_data = {
                    'document_index': i,
                    'page': doc.metadata.get('page', 'unknown'),
                    'content': doc.page_content,
                    'content_length': len(doc.page_content),
                    'source_file': doc.metadata.get('source_file', 'unknown'),
                    'has_titles': doc.metadata.get('has_titles', False)
                }
                
                # Add structured blocks if available
                if 'structured_blocks' in doc.metadata:
                    doc_data['structured_blocks'] = []
                    for block in doc.metadata['structured_blocks']:
                        doc_data['structured_blocks'].append({
                            'text': block['text'],
                            'is_title': block['is_title'],
                            'page': block['page'],
                            'text_length': len(block['text'])
                        })
                
                documents_data.append(doc_data)
            
            documents_output = {
                'source_file': pdf_name,
                'total_documents': len(documents),
                'documents': documents_data,
                'timestamp': result['metadata']['timestamp']
            }
            
            documents_path = self.output_dir / f"{base_name}_documents.json"
            self.save_json(documents_output, documents_path)
            print(f"  ✓ Saved documents: {documents_path.name}")
    
    def _get_section_distribution(self, chunks: List[Document]) -> Dict[str, int]:
        """Get distribution of chunks by section type."""
        distribution = {}
        for chunk in chunks:
            section_type = chunk.metadata.get('section_type', 'unknown')
            distribution[section_type] = distribution.get(section_type, 0) + 1
        return distribution
            
    
    def run(self):
        """
        Run the complete pipeline on all PDFs in input directory.
        """
        print("\n" + "="*60)
        print("PDF SUMMARIZATION PIPELINE")
        print("="*60)
        
        # Find PDF files
        pdf_files = self.find_pdf_files()
        
        if not pdf_files:
            print("\n No PDF files found in input directory!")
            print(f"   Please place PDF files in: {self.input_dir}")
            return
        
        print(f"\nFound {len(pdf_files)} PDF file(s) to process:")
        for pdf in pdf_files:
            print(f"  - {pdf.name}")
        
        results = []
        successful = 0
        failed = 0
        
        for idx, pdf_path in enumerate(pdf_files, 1):
            try:
                print(f"\n\n[{idx}/{len(pdf_files)}] Processing {pdf_path.name}")
                
                # Process PDF
                result = self.process_single_pdf(pdf_path)
                
                # Get chunks and documents for saving (if available)
                chunks = None
                documents = None
                if hasattr(self, '_last_chunks'):
                    chunks = self._last_chunks
                if hasattr(self, '_last_documents'):
                    documents = self._last_documents
                
                # Save outputs
                self.save_outputs(result, pdf_path.name, chunks, documents)
                
                results.append({
                    "file": pdf_path.name,
                    "status": "success",
                    "summary_length": len(result['summary'])
                })
                successful += 1
                
            except Exception as e:
                print(f"\nError processing {pdf_path.name}:")
                print(f"   {str(e)}")
                
                results.append({
                    "file": pdf_path.name,
                    "status": "failed",
                    "error": str(e)
                })
                failed += 1
        
        # Print summary
        print("\n\n" + "="*60)
        print("PIPELINE COMPLETED")
        print("="*60)
        print(f"\nTotal PDFs: {len(pdf_files)}")
        print(f"✓ Successful: {successful}")
        if failed > 0:
            print(f"✗ Failed: {failed}")
        
        print(f"\nOutputs saved to: {self.output_dir}")
        
        # Save processing log
        log_path = self.output_dir / f"processing_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        self.save_json({
            "timestamp": datetime.now().isoformat(),
            "total_files": len(pdf_files),
            "successful": successful,
            "failed": failed,
            "results": results
        }, log_path)
        print(f"Processing log: {log_path.name}")
        
        return successful, failed


def main():
    """Main entry point."""
    try:
        # Check for required environment variables
        if not config.openai_api_key:
            print("\n Error: OPENAI_API_KEY not set!")
            print("   Please set it in .env file or environment variables.")
            sys.exit(1)
        
        # Create and run pipeline
        pipeline = PDFSummarizationPipeline()
        _, failed = pipeline.run()
        
        # Exit with appropriate code
        if failed > 0:
            sys.exit(1)
        else:
            sys.exit(0)
            
    except KeyboardInterrupt:
        print("\n\n Pipeline interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        print(f"\n\n❌ Fatal error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()