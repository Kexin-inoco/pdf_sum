docker-compose build --no-cache app

docker-compose build app

docker-compose run --rm app python src/main.py



PDF
    ↓
EnhancedPDFProcessor.process()
    ↓
load_pdf_with_structure() [PyMuPDF]
    ↓
split_documents()
    ↓
AcademicPaperSplitter.split_documents()
    ↓
extract_key_info() [Using SEC_NAMES Mode]
    ↓
create_summary_chunks() [Creating structured blocks]
    ↓
return List[Document] chunks
    ↓
save_outputs()
    ↓
save to JSON file:
- {filename}_summary.json
- {filename}_chunks.json