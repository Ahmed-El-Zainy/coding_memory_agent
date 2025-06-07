"""
Main FastAPI application for the Memory Chatbot
"""

import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Add project paths
Main_Path = os.path.dirname(os.path.abspath(__file__))
Add_Path = os.path.dirname(Main_Path)
sys.path.append(Add_Path)

# Import utilities
from utilities import (
    # Models
    ChatMessage, ChatResponse, MemoryItem, ModelChangeRequest,
    # Configuration
    load_config,
    # Model initialization
    load_embedding_model, initialize_gemini, initialize_chromadb,
    # Core functions
    decide_memory_retrieval, store_memory, retrieve_memories, generate_response,
    # Health checks
    health_check_embedding_model, health_check_gemini,
    # Memory management
    get_user_memories, clear_user_memories
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Chatbot with Memory API (Gemini + HuggingFace)", 
    version="1.0.0",
    description="A chatbot with persistent memory using Gemini LLM and HuggingFace embeddings"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models and services
embedding_model = None
embedding_model_name = None
gemini_model = None
collection = None

@app.on_event("startup")
async def startup_event():
    """Initialize all models and services on startup"""
    global embedding_model, embedding_model_name, gemini_model, collection
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = load_config('src/config.yaml')
        gemini_api_key = config.get("google_api_key")
        
        if not gemini_api_key:
            raise ValueError("Google API key not found in configuration")
        
        # Initialize Gemini
        logger.info("Initializing Gemini model...")
        gemini_model = initialize_gemini(gemini_api_key)
        
        # Initialize embedding model
        logger.info("Loading embedding model...")
        embedding_model, embedding_model_name = load_embedding_model()
        
        # Initialize ChromaDB
        logger.info("Initializing ChromaDB...")
        collection = initialize_chromadb()
        
        logger.info("All models and services initialized successfully!")
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise

# API Endpoints

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    """Main chat endpoint"""
    try:
        message = chat_message.message
        user_id = chat_message.user_id
        
        logger.info(f"Received message from {user_id}: {message}")
        
        # Step 1: Decide if we need to retrieve memories
        should_retrieve, decision_reason = await decide_memory_retrieval(
            message, user_id, gemini_model
        )
        
        # Step 2: Retrieve memories if needed
        memories = []
        if should_retrieve:
            memories = await retrieve_memories(
                message, user_id, collection, embedding_model
            )
            logger.info(f"Retrieved {len(memories)} memories")
        
        # Step 3: Generate response
        response = await generate_response(message, memories, gemini_model)
        
        # Step 4: Store this conversation in memory
        await store_memory(message, user_id, response, collection, embedding_model)
        
        return ChatResponse(
            response=response,
            memories_used=memories,
            memory_retrieval_decision=decision_reason,
            model_info={
                "llm_model": "Google Gemini 1.5 Flash",
                "embedding_model": embedding_model_name,
                "vector_store": "ChromaDB"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        if "rate limit" in str(e).lower() or "quota" in str(e).lower():
            raise HTTPException(
                status_code=429, 
                detail="API rate limit exceeded. Please try again later."
            )
        else:
            raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{user_id}")
async def get_memories_endpoint(user_id: str, limit: int = 10):
    """Get all memories for a user"""
    try:
        memories = get_user_memories(user_id, collection, limit)
        return {"memories": memories, "total": len(memories)}
    except Exception as e:
        logger.error(f"Error retrieving user memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memories")

@app.delete("/memories/{user_id}")
async def clear_memories_endpoint(user_id: str):
    """Clear all memories for a user"""
    try:
        result = clear_user_memories(user_id, collection)
        return result
    except Exception as e:
        logger.error(f"Error clearing memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear memories")

@app.get("/models")
async def get_model_info():
    """Get information about the models being used"""
    try:
        test_embedding = embedding_model.encode("test")
        return {
            "llm_model": "Google Gemini 1.5 Flash",
            "embedding_model": embedding_model_name,
            "vector_store": "ChromaDB",
            "embedding_dimensions": len(test_embedding),
            "status": "active"
        }
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model information")

@app.post("/change-embedding-model")
async def change_embedding_model_endpoint(request: ModelChangeRequest):
    """Change the embedding model (for experimentation)"""
    global embedding_model, embedding_model_name
    
    try:
        logger.info(f"Attempting to change embedding model to: {request.model_name}")
        new_model, new_model_name = load_embedding_model(request.model_name)
        
        # Update global variables
        embedding_model = new_model
        embedding_model_name = new_model_name
        
        logger.info(f"Successfully changed embedding model to: {new_model_name}")
        
        test_embedding = embedding_model.encode("test")
        return {
            "message": f"Successfully changed embedding model to {new_model_name}",
            "embedding_dimensions": len(test_embedding),
            "model_name": new_model_name
        }
    except Exception as e:
        logger.error(f"Failed to change embedding model: {e}")
        raise HTTPException(
            status_code=400, 
            detail=f"Failed to load model {request.model_name}: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Test embedding model
        embedding_health = health_check_embedding_model(
            embedding_model, embedding_model_name
        )
        
        # Test Gemini
        gemini_health = health_check_gemini(gemini_model)
        
        # Overall status
        overall_status = "healthy" if (
            embedding_health["status"] == "OK" and 
            gemini_health["status"] == "OK"
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "models": {
                "llm": f"Google Gemini 1.5 Flash - {gemini_health['status']}",
                "embedding": f"{embedding_model_name} - {embedding_health['status']}",
                "vector_store": "ChromaDB - OK"
            },
            "details": {
                "embedding": embedding_health,
                "gemini": gemini_health
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Memory Chatbot API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "memories": "/memories/{user_id}",
            "models": "/models",
            "health": "/health"
        }
    }

# Run the application
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )