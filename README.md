# PDF Summarization Pipeline

A Docker-based system for automatically generating table of contents from academic PDF papers using AI.

## 🚀 Quick Start

### 1. Build the Docker image

```bash
docker-compose build
```

### 2. Run the pipeline

```bash
docker-compose run --rm app python src/main.py
```

### 3. Add your PDFs

- Place PDF files in `data/input/` directory
- The system will automatically process all PDFs in this folder

### 4. Get results

- Processed results will appear in `data/output/`
- Each PDF gets its own folder with:
  - `table_of_contents.md` - Generated table of contents
  - `summary.json` - Structured data
  - `documents.json` - Raw document data

## 📁 Project Structure

```
pdf_sum/
├── data/
│   ├── input/          # Place your PDFs here
│   └── output/         # Results appear here
├── src/                # Source code
├── docker-compose.yml  # Docker configuration
└── requirements.txt    # Python dependencies
```

## ⚙️ Configuration

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your_api_key_here
```

## 📊 Output Example

Each processed PDF creates:

- **Markdown TOC**: Human-readable table of contents
- **JSON Summary**: Structured metadata and titles
- **Document Data**: Raw processing information


## 📝 Requirements

- Docker & Docker Compose
- OpenAI API key
- Python 3.11+ (for local development)
