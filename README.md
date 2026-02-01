# Multi-Modal Document Intelligence Platform 🔮

> **Unlike typical document QA systems, our platform performs conflict-aware multi-modal reasoning, explicitly resolving disagreements between vision, OCR, and language models.**

A production-ready document processing platform powered by a **6-agent LangGraph pipeline**, **multi-modal RAG**, and a premium React frontend.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi)
![React](https://img.shields.io/badge/React-18+-61DAFB?logo=react)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker)

---

## 🏆 Competition-Winning Features

### 🔥 Technical Excellence (+5)

| Feature | Description |
|---------|-------------|
| **Hybrid Cross-Modal Retrieval** | Parallel search across text, tables, images with LLM-based re-ranking |
| **Vision-Grounded Answering** | Every answer includes bounding box references and page locations |
| **Efficient CV Deployment** | CPU-friendly mode with dynamic YOLO model selection |

### ⚡ Advanced Features (+5)

| Feature | Description |
|---------|-------------|
| **Multi-Document Reasoning** | Query and compare across 3+ documents simultaneously |
| **Table Reasoning Agent** | Pandas-powered aggregations, trends, and comparisons |
| **Visual Confidence Heatmap** | Color-coded overlays (green/yellow/red) for instant clarity |

### 💡 Innovation (+5)

| Feature | Description |
|---------|-------------|
| **Conflict Resolution Engine** | Detects OCR vs Vision vs Table disagreements with explainable resolutions |
| **Self-Healing Pipeline** | Auto-retry, fallback strategies, cached recovery paths |
| **ELI5 vs Expert Mode** | Side-by-side explanations at different complexity levels |

---

## ✨ Core Features

### 🤖 6-Agent LangGraph Pipeline
1. **Vision Agent** - YOLO-powered document layout detection
2. **OCR Agent** - Hybrid Tesseract + EasyOCR engine
3. **Layout Agent** - Spatial relationship analysis
4. **Text Reasoning Agent** - LLM summarization & entity extraction
5. **Fusion Agent** - Cross-modal output merging
6. **Validation Agent** - Confidence scoring & human review flagging

### 🎯 ELI5 vs Expert Mode
Get the same content explained at different levels:
- **🎈 ELI5**: Simple explanations for anyone
- **📝 Standard**: Balanced response
- **🎓 Expert**: Technical analysis with citations

### 🔍 Multi-Modal RAG
- **3 Vector Collections**: Text, tables, and images
- **Cross-Modal Retrieval**: Find relevant content across modalities
- **Reciprocal Rank Fusion**: Smart result ranking

### 👤 Human Review Workflow
- Field-level confidence scoring
- Automatic flagging of low-confidence extractions
- Correction workflow with history tracking

## 🏛️ Architectural Decisions

### Why LangGraph over CrewAI?
While CrewAI offers autonomous agentic behaviors, we prioritized **LangGraph** for this platform to ensure:
- **Deterministic Control Flow**: Critical for document processing pipelines where order matters (Vision → OCR → Layout).
- **State Management**: LangGraph's stateful graph architecture allows precise tracking of document processing stages.
- **Production Reliability**: Avoiding the non-deterministic loops common in fully autonomous agent frameworks.
- **Explicit Human-in-the-Loop**: Built-in support for interrupting execution for human review (validation stage).

## 🏢 Enterprise Readiness

### Scalability
- **Async Processing**: FastAPIs `async/await` pattern handles concurrent document uploads efficiently.
- **Vector Search**: Qdrant is optimized for high-dimensional vector search at scale (millions of chunks).
- **Stateless Agents**: Agents are designed to be stateless, allowing horizontal scaling of the backend services.

### Security
- **Containerization**: Full Docker support ensures consistent and isolated execution environments.
- **Input Validation**: Rigorous Pydantic validation sanitizes all inputs before processing.
- **Configurable LLM Backends**: Support for private LLM deployments (via standard OpenAI-compatible endpoints) prevents data leakage.

## 🏗️ Architecture

```
Document Upload
      │
      ▼
┌─────────────────────────────────────────┐
│           LangGraph Pipeline            │
├─────────────────────────────────────────┤
│ Vision → OCR → Layout → Reasoning →    │
│           Fusion → Validation          │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│          Qdrant Vector Store            │
│   ┌────────┬────────┬────────┐         │
│   │  Text  │ Tables │ Images │         │
│   └────────┴────────┴────────┘         │
└─────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────┐
│       Advanced Features Layer           │
│  • Cross-Modal Retriever                │
│  • Table Reasoning Agent                │
│  • Conflict Resolution Engine           │
│  • Self-Healing Pipeline                │
│  • Confidence Heatmaps                  │
└─────────────────────────────────────────┘
      │
      ▼
    Query Response (ELI5 / Expert)
```

## 🚀 Quick Start

### Docker (Recommended)

```bash
# Clone and configure
cd new_agent
cp backend/.env.example backend/.env
# Edit .env with your API keys

# Start all services
cd docker
docker-compose up -d

# Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

### Local Development

```bash
# Start Qdrant
docker run -p 6333:6333 qdrant/qdrant

# Backend
cd backend
python -m venv venv && venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env  # Configure API keys
uvicorn backend.api.main:app --reload

# Frontend (new terminal)
cd frontend
npm install && npm run dev
```

## 📁 Project Structure

```
new_agent/
├── backend/
│   ├── api/            # FastAPI routes & WebSocket
│   ├── agents/         # 6 LangGraph agents
│   ├── cv/             # YOLO detection
│   ├── ocr/            # Hybrid OCR
│   ├── rag/            # Multi-modal RAG
│   ├── config.py       # Configuration
│   └── requirements.txt
├── frontend/
│   └── src/
│       ├── components/ # React UI components
│       └── index.css   # Design system
├── docker/
│   ├── docker-compose.yml
│   └── Dockerfiles
├── demo_instructions.md
└── README.md
```

## 🛠️ Technology Stack

| Layer | Technology |
|-------|------------|
| **Backend** | FastAPI, LangGraph, Pydantic |
| **Agents** | LangChain, OpenAI/Anthropic APIs |
| **CV** | Ultralytics YOLO, OpenCV |
| **OCR** | Tesseract, EasyOCR |
| **Vector DB** | Qdrant |
| **Embeddings** | SentenceTransformers, CLIP |
| **Frontend** | React, Vite, CSS |
| **Deployment** | Docker, nginx |

## 📡 API Endpoints

### Documents
- `POST /api/documents/upload` - Upload document
- `GET /api/documents/{id}/status` - Processing status
- `GET /api/documents/{id}/results` - Get extracted data

### Chat
- `POST /api/chat/{id}` - Query with RAG (supports eli5/expert mode)
- `POST /api/chat/{id}/explain` - Compare ELI5 vs Expert

### Review
- `GET /api/review/{id}/flags` - Get flagged items
- `PUT /api/review/{id}/correct` - Submit correction

## ⚙️ Configuration

Environment variables (`.env`):

```env
# LLM
LLM_PROVIDER=openai
QROK_API_KEY=sk-...

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Processing
LOW_CONFIDENCE_THRESHOLD=0.6
HUMAN_REVIEW_THRESHOLD=0.7
```

## 🎨 Frontend Design

Premium dark mode UI featuring:
- Glassmorphism effects
- Gradient accents
- Smooth micro-animations
- Responsive layout
- Confidence heatmaps

## 📝 Documentation

- [Demo Instructions](demo_instructions.md) - Step-by-step walkthrough
- [API Docs](http://localhost:8000/docs) - Interactive Swagger UI

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

MIT License
