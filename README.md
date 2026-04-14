# EU AI Act RAG with Confidence Scoring

A Retrieval-Augmented Generation (RAG) system for querying the EU AI Act with confidence-calibrated responses.

## Motivation

Legal documents are a domain where hallucination is particularly costly. This project applies confidence calibration to a RAG system over the EU AI Act, inspired by Yuan et al. (arXiv:2604.05952, 2026), which demonstrates that calibrated models encourage users to verify uncertain information rather than accepting it blindly. Responses below a confidence threshold are explicitly flagged rather than presented as fact.

## Overview

Ask questions about the EU AI Act in natural language. The system retrieves relevant articles, generates an answer using a local LLM, and provides a confidence score indicating how certain the system is about its response.

- High confidence (≥0.8) — answer is well-supported by the document
- Medium confidence (0.6–0.8) — answer is partial, verify manually  
- Low confidence (<0.6) — consult the official EU AI Act source

## Features

- Natural language querying of EU AI Act
- Confidence scoring per response
- Source citation (which articles were used)
- Local LLM inference (no API costs)
- REST API backend
- Interactive web interface

## Tech Stack

- **Python** — core language
- **FastAPI** — REST API backend
- **LangChain** — RAG pipeline
- **PostgreSQL + pgvector** — vector database for semantic search
- **Ollama + Llama3.2** — local LLM inference
- **Streamlit** — frontend UI
- **Docker** — containerized PostgreSQL

## Architecture

Streamlit frontend sends POST requests to a FastAPI backend. The backend uses LangChain to retrieve relevant EU AI Act chunks from PostgreSQL (pgvector) and passes them as context to a local Ollama LLM. The response includes an answer, confidence score, and source citations.

## Setup

### Prerequisites
- Docker
- Ollama

### Installation

1. Clone the repository
```bash
git clone https://github.com/Marko-Malenica/eu-ai-act-rag
cd eu-ai-act-rag
```

2. Pull Ollama models
```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

3. Download the EU AI Act PDF from [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689) and place it at `data/eu_ai_act.pdf`

4. Setup environment
```bash
make setup
# Edit .env with your credentials
```

5. Start the application
```bash
make start
```

6. First time only — ingest EU AI Act into the database
```bash
make ingest
```

7. Open [http://localhost:8501](http://localhost:8501)

## API Endpoints

### POST /query
Request:
```json
{
  "question": "What is a high-risk AI system?"
}
```

Response:
```json
{
  "answer": "According to Article 6 of the EU AI Act...",
  "confidence": 0.87,
  "sources": ["Page 12", "Page 53"],
  "flag": "high"
}
```

Low confidence response:
```json
{
  "answer": "...",
  "confidence": 0.45,
  "sources": ["Page 2"],
  "flag": "low"
}
```

### GET /health
```json
{"status": "ok"}
```

## Data Source

EU AI Act official text retrieved from [EUR-Lex](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:32024R1689)

## Model

Default: Llama3.2 via Ollama (local, free)  
Easily swappable to OpenAI or Anthropic by changing one line in `backend/rag.py`

## License

MIT