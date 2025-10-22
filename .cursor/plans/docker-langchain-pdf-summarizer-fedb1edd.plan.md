<!-- fedb1edd-6b70-445e-8c4e-b7587fc93f5a 486563ad-8807-47bf-a9de-3b2340157121 -->
# Docker + LangChain PDF Summarization System

## Architecture Design

### Service Architecture

```
docker-compose.yml
├── app (Python Application)
│   ├── LangChain Core
│   ├── PDF Processing
│   └── Summary Generation
└── chromadb (Vector Database - Optional)
    └── Lightweight Vector Storage
```

### Data Flow

```
data/input/ (volume) → app container → processing → data/output/ (volume)
                          ↓ (optional)
                      chromadb container
```

### Volume Structure

```
volumes:
  - ./data/input:/app/data/input       # PDF inputs
  - ./data/output:/app/data/output     # Summary outputs
  - ./config:/app/config               # Configuration files
  - chroma_data:/chroma/data           # Vector data persistence
```

## Implementation Steps

### 1. Project Structure

```
pdf_sum/
├── docker-compose.yml          # Service orchestration
├── Dockerfile                  # Python app image
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variable template
├── .dockerignore              # Docker ignore file
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── pdf_processor.py       # PDF processing module
│   ├── summarizer.py          # Summarization module
│   └── main.py               # Main program
│
├── data/
│   ├── input/                 # PDF input directory
│   │   └── .gitkeep
│   └── output/                # Results output directory
│       └── .gitkeep
│
├── config/
│   └── settings.yaml          # Application configuration
│
└── README.md                  # Usage documentation
```

### 2. Core File Contents

#### docker-compose.yml

```yaml
services:
  app:
    build: .
    container_name: pdf_summarizer
    volumes:
      - ./data/input:/app/data/input
      - ./data/output:/app/data/output
      - ./config:/app/config
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
    depends_on:
      - chromadb
    networks:
      - pdf_network

  chromadb:
    image: chromadb/chroma:latest
    container_name: chromadb
    volumes:
      - chroma_data:/chroma/data
    environment:
      - ANONYMIZED_TELEMETRY=False
    ports:
      - "8000:8000"
    networks:
      - pdf_network

volumes:
  chroma_data:

networks:
  pdf_network:
```

#### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency file
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create data directories
RUN mkdir -p /app/data/input /app/data/output /app/config

CMD ["python", "src/main.py"]
```

#### requirements.txt

```txt
langchain==0.1.0
langchain-community==0.0.38
langchain-openai==0.0.5
chromadb==0.4.22
pypdf==3.17.4
python-dotenv==1.0.0
tiktoken==0.5.2
pyyaml==6.0.1
```

### 3. Application Code Modules

#### src/config.py

Configuration management: reads environment variables and config files, provides unified configuration interface

#### src/pdf_processor.py

- PDFProcessor class: uses LangChain PyPDFLoader to load PDFs
- Uses RecursiveCharacterTextSplitter for intelligent chunking
- Returns standard Document objects

#### src/summarizer.py

- Summarizer class: wraps LangChain summarization chain
- Supports map_reduce strategy
- Optional ChromaDB storage (interface reserved for future use)

#### src/main.py

- Scans data/input directory
- Invokes processor and summarizer
- Saves results to data/output
- CLI output for processing progress

### 4. Configuration Files

#### .env.example

```
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL=gpt-4-turbo-preview
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

#### config/settings.yaml

```yaml
pdf_processing:
  chunk_size: 1000
  chunk_overlap: 200

summarization:
  strategy: map_reduce
  model: gpt-4-turbo-preview
  temperature: 0.3

output:
  formats: [json, markdown]

chromadb:
  enabled: false  # Disabled for MVP phase
  host: chromadb
  port: 8000
```

### 5. README.md

Usage documentation:

- Environment setup
- Docker commands
- Usage examples
- Extension guide

## Usage

### Initialization

```bash
# Copy environment variables
cp .env.example .env
# Edit .env to add OpenAI API key

# Build image
docker-compose build
```

### Execution

```bash
# 1. Place PDF files in data/input/
# 2. Run processing
docker-compose run --rm app python src/main.py

# 3. View results
ls data/output/
```

### Development & Debugging

```bash
# Enter container shell
docker-compose run --rm app bash

# View logs
docker-compose logs -f
```

## Extension Path

### Enable ChromaDB

Modify config/settings.yaml: `chromadb.enabled: true`

Code automatically uses vector storage functionality

### Add RAG Q&A

Add rag_qa.py module in src/, reuse existing vector store

### Add Web Interface

Add new service to docker-compose.yml (Streamlit/FastAPI)

## Technical Highlights

1. **Containerized Isolation**: Dependencies fully independent, consistent environment
2. **Data Persistence**: Volumes ensure data is not lost
3. **Modular Design**: Each component has clear responsibilities
4. **Extensible Architecture**: ChromaDB reserved, easy to enable
5. **Developer Friendly**: Hot reload, logging, convenient debugging

## Key Design Decisions

### Why Docker?

- Reproducible environment across different machines
- Eliminates "works on my machine" issues
- Easy dependency management

### Why ChromaDB?

- Lightweight vector database
- Easy to integrate with LangChain
- No complex setup required
- Enables future RAG capabilities

### Why LangChain?

- Industry standard for LLM applications
- Rich ecosystem of components
- Reduces boilerplate code
- Facilitates future extensions

## Architecture Benefits

### Current (MVP)

- Simple PDF to summary pipeline
- Clean separation of concerns
- Easy to test and debug

### Future (Scalable)

- Enable vector storage for semantic search
- Add RAG for intelligent Q&A
- Scale to multiple papers
- Add batch processing
- Deploy as web service

### To-dos

- [ ] 创建项目目录结构和占位文件
- [ ] 编写 Dockerfile、docker-compose.yml、.dockerignore
- [ ] 创建 requirements.txt 和配置文件
- [ ] 实现核心模块：config.py、pdf_processor.py、summarizer.py
- [ ] 实现 main.py 主程序
- [ ] 编写 README.md 使用文档