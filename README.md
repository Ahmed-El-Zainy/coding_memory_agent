# AI Chatbot with Memory - Google Gemini Flash & Hugging Face

## 🚀 Project Overview

This is a production-ready chatbot with vector-based memory that intelligently decides when to retrieve relevant past conversations. The system uses **Google Gemini Flash** for chat responses and **Hugging Face Transformers** for embeddings, making it cost-effective and highly customizable.

### 🧠 Key Features
- **Smart Memory**: Vector-based conversation storage with intelligent retrieval
- **Cost-Effective**: Uses Google Gemini Flash (cheaper than GPT-4)
- **Flexible Embeddings**: Multiple Hugging Face models available
- **Production Ready**: FastAPI backend + Streamlit frontend
- **Scalable**: Docker containerized for easy deployment

## 🏗️ Architecture

```
┌─────────────────┐    HTTP    ┌─────────────────┐    Gemini API   ┌─────────────┐
│   Streamlit     │ ---------> │    FastAPI      │ --------------> │   Google     │
│   Frontend      │            │    Backend      │                 │   Gemini     │
└─────────────────┘            └─────────────────┘                 │   Flash      │
                                        │                          └─────────────┘
                                        │ HuggingFace
                                        ▼ Embeddings
                               ┌─────────────────┐
                               │   ChromaDB      │
                               │ Vector Database │
                               └─────────────────┘
```

## 📂 Project Structure

```
chatbot-with-memory/
├── backend/
│   ├── main.py              # FastAPI app with Gemini + HF
│   ├── requirements.txt     # Python dependencies
│   └── Dockerfile          # Backend container
├── frontend/
│   ├── app.py              # Streamlit application
│   ├── requirements.txt     # Frontend dependencies
│   └── Dockerfile          # Frontend container
├── docker-compose.yml       # Local development setup
├── .env.example            # Environment variables template
└── README.md               # This file
```

## 🛠️ Local Development Setup

### Prerequisites
- Docker and Docker Compose
- Google Gemini API key (free tier available)

### Step 1: Get API Keys

1. **Google Gemini API Key**:
   - Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create a new API key
   - Copy the key

### Step 2: Environment Setup
```bash
# Create project structure
mkdir chatbot-with-memory && cd chatbot-with-memory
mkdir backend frontend

# Create environment file
cat > .env << EOF
GEMINI_API_KEY=your_gemini_api_key_here
EMBEDDING_MODEL=all-MiniLM-L6-v2
EOF
```

### Step 3: Backend Setup
```bash
cd backend
# Create main.py with the FastAPI code
# Create requirements.txt with dependencies
# Create Dockerfile
```

### Step 4: Frontend Setup
```bash
cd ../frontend
# Create app.py with Streamlit code
# Create requirements.txt
# Create Dockerfile
```

### Step 5: Run with Docker Compose
```bash
# From project root
docker-compose up --build
```

**Access Points:**
- Frontend: http://localhost:8501
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs
- Health Check: http://localhost:8000/health

## 🤖 Available Embedding Models

You can change the embedding model by setting the `EMBEDDING_MODEL` environment variable:

### Recommended Models:
```bash
# Small & Fast (default)
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Better Quality
EMBEDDING_MODEL=all-mpnet-base-v2

# Multilingual
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# Large & High Quality
EMBEDDING_MODEL=all-MiniLM-L12-v2
```

### Runtime Model Switching:
```bash
# Change model via API (experimental)
curl -X POST "http://localhost:8000/change-embedding-model?model_name=all-mpnet-base-v2"
```

## ☁️ Cloud Deployment