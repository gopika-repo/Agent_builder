# Demo Instructions

## Quick Start Guide

### Option 1: Docker (Recommended)

1. **Clone and Configure**
   ```bash
   cd new_agent
   cp backend/.env.example backend/.env
   # Edit .env with your API keys
   ```

2. **Start All Services**
   ```bash
   cd docker
   docker-compose up -d
   ```

3. **Access the Application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Docs: http://localhost:8000/docs
   - Qdrant UI: http://localhost:6333/dashboard

### Option 2: Local Development

1. **Start Qdrant**
   ```bash
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
   ```

2. **Setup Backend**
   ```bash
   cd backend
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   
   pip install -r requirements.txt
   
   # Configure environment
   cp .env.example .env
   # Edit .env with your OPENAI_API_KEY or ANTHROPIC_API_KEY
   
   # Run the backend
   uvicorn backend.api.main:app --reload
   ```

3. **Setup Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

4. **Access**
   - Frontend: http://localhost:5173
   - Backend: http://localhost:8000

---

## Demo Walkthrough

### 1. Upload a Document

1. Go to the **Documents** tab
2. Drag and drop a PDF or click to browse
3. Watch the processing status update in real-time

### 2. View Extracted Data

Once processing completes:

1. Click on the document in the list
2. Explore **Summary** tab:
   - Document type classification
   - AI-generated summary
   - Key points extracted
   - Topics identified
3. View **Tables** tab for structured table data
4. Check **Entities** for named entity recognition
5. See **Structure** for document hierarchy

### 3. Try the ELI5 vs Expert Feature 🌟

This is the standout feature!

1. Go to **Ask AI** tab
2. Use the mode toggle:
   - **🎈 ELI5**: Simple explanations for anyone
   - **📝 Standard**: Balanced response
   - **🎓 Expert**: Technical analysis

3. Ask a question like:
   - "What is this document about?"
   - "Explain the financial performance"

4. Click the **⚖️** button to see BOTH explanations side-by-side!

### 4. Human Review Workflow

1. Go to **Review** tab
2. See flagged items with low confidence
3. For each item:
   - **Confirm Original**: Mark as correct despite low confidence
   - **Save Correction**: Edit and save the correct value
   - **Reject**: Mark as incorrect

### 5. Try Different Document Types

Upload various documents to see agent specialization:
- **Financial reports**: Revenue, metrics, trends
- **Legal documents**: Parties, obligations, terms
- **Technical documents**: Specifications, diagrams

---

## API Examples

### Upload Document
```bash
curl -X POST "http://localhost:8000/api/documents/upload" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@document.pdf"
```

### Query Document
```bash
curl -X POST "http://localhost:8000/api/chat/{document_id}" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main topic?", "mode": "eli5"}'
```

### Compare Explanations
```bash
curl -X POST "http://localhost:8000/api/chat/{document_id}/explain?query=Summarize%20the%20document"
```

---

## Key Features Demonstrated

| Feature | Description |
|---------|-------------|
| 6-Agent Pipeline | Vision → OCR → Layout → Reasoning → Fusion → Validation |
| Multi-Modal RAG | Search across text, tables, and figures |
| ELI5 vs Expert | Same content, different explanation levels |
| Confidence Scoring | Field-level confidence with visual heatmap |
| Human Review | Flag and correct low-confidence extractions |
| Real-time Updates | WebSocket progress notifications |

---

## Troubleshooting

- **Backend won't start**: Check if Qdrant is running on port 6333
- **No API response**: Verify OPENAI_API_KEY in .env
- **OCR errors**: Install Tesseract: `apt-get install tesseract-ocr`
- **PDF issues**: Install Poppler: `apt-get install poppler-utils`
