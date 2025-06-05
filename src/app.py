from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings
import uuid
from datetime import datetime
import os
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from utils import load_config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatbot with Memory API (Gemini + HuggingFace)", version="1.0.0")

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
config = load_config('config.yaml')
GEMINI_API_KEY = config["google_api_key"]
print(GEMINI_API_KEY)
print(lol)
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize Hugging Face embedding model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
logger.info(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")

try:
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    # Fallback to a smaller model
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Initialize ChromaDB (vector store)
chroma_client = chromadb.Client(Settings(
    chroma_db_impl="duckdb+parquet",
    persist_directory="./chroma_db"
))

# Create or get collection
try:
    collection = chroma_client.create_collection(
        name="chat_memories",
        metadata={"hnsw:space": "cosine"}
    )
except:
    collection = chroma_client.get_collection("chat_memories")

# Pydantic models
class ChatMessage(BaseModel):
    message: str
    user_id: Optional[str] = "default_user"

class ChatResponse(BaseModel):
    response: str
    memories_used: List[str]
    memory_retrieval_decision: str
    model_info: Dict[str, str]

class MemoryItem(BaseModel):
    id: str
    content: str
    timestamp: str
    user_id: str

# Helper functions
def get_embedding_sync(text: str) -> List[float]:
    """Get Hugging Face embedding for text (synchronous)"""
    try:
        embedding = embedding_model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise

async def get_embedding(text: str) -> List[float]:
    """Get Hugging Face embedding for text (async wrapper)"""
    loop = asyncio.get_event_loop()
    try:
        embedding = await loop.run_in_executor(executor, get_embedding_sync, text)
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding")

async def decide_memory_retrieval(message: str, user_id: str) -> tuple[bool, str]:
    """Use Gemini to decide if memory retrieval is needed"""
    try:
        decision_prompt = f"""
        You are an AI assistant that helps decide when to retrieve memories from past conversations.
        
        Current user message: "{message}"
        
        Should I retrieve relevant memories from past conversations to better answer this message?
        
        Consider retrieving memories if:
        - The user refers to past conversations ("remember when...", "like we discussed...")
        - The question relates to previous topics or context
        - The user asks about their preferences or past interactions
        - The conversation would benefit from historical context
        - The user mentions something they told you before
        
        Do NOT retrieve memories if:
        - This is a simple greeting or casual message
        - The question is completely self-contained
        - It's a generic question that doesn't need personal context
        - It's the first interaction
        
        Respond with either "YES" or "NO" followed by a brief explanation in one sentence.
        """
        
        response = gemini_model.generate_content(decision_prompt)
        decision_text = response.text.strip()
        should_retrieve = decision_text.upper().startswith("YES")
        
        return should_retrieve, decision_text
    except Exception as e:
        logger.error(f"Error in memory decision: {e}")
        return False, "Error in decision making - defaulting to no memory retrieval"

async def store_memory(message: str, user_id: str, response: str) -> str:
    """Store conversation in vector database"""
    try:
        memory_content = f"User: {message}\nAssistant: {response}"
        embedding = await get_embedding(memory_content)
        
        memory_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        collection.add(
            embeddings=[embedding],
            documents=[memory_content],
            metadatas=[{
                "user_id": user_id,
                "timestamp": timestamp,
                "user_message": message,
                "assistant_response": response
            }],
            ids=[memory_id]
        )
        
        logger.info(f"Stored memory with ID: {memory_id}")
        return memory_id
    except Exception as e:
        logger.error(f"Error storing memory: {e}")
        raise HTTPException(status_code=500, detail="Failed to store memory")

async def retrieve_memories(message: str, user_id: str, n_results: int = 3) -> List[str]:
    """Retrieve relevant memories from vector database"""
    try:
        embedding = await get_embedding(message)
        
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where={"user_id": user_id}
        )
        
        memories = []
        if results['documents'] and results['documents'][0]:
            for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
                timestamp = metadata.get('timestamp', 'Unknown time')
                # Make timestamp more readable
                try:
                    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    readable_time = dt.strftime('%Y-%m-%d %H:%M')
                except:
                    readable_time = timestamp
                memories.append(f"[{readable_time}] {doc}")
        
        return memories
    except Exception as e:
        logger.error(f"Error retrieving memories: {e}")
        return []

async def generate_response(message: str, memories: List[str]) -> str:
    """Generate AI response using Gemini with current message and relevant memories"""
    try:
        system_context = """You are a helpful and friendly AI assistant with access to conversation history. 
        Use the provided memories to give contextual and personalized responses when relevant.
        Be natural, conversational, and engaging. Remember details about the user when appropriate.
        If you reference past conversations, do so naturally without explicitly mentioning "memories" or "past conversations"."""
        
        user_prompt = f"Current message: {message}"
        
        if memories:
            memory_context = "\n\nRelevant conversation history:\n" + "\n".join(memories)
            user_prompt += memory_context
            user_prompt += "\n\nPlease respond to the current message, using the conversation history for context when relevant."
        
        full_prompt = f"{system_context}\n\n{user_prompt}"
        
        response = gemini_model.generate_content(full_prompt)
        return response.text.strip()
        
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        # Fallback response
        if "rate limit" in str(e).lower() or "quota" in str(e).lower():
            raise HTTPException(status_code=429, detail="API rate limit exceeded. Please try again later.")
        else:
            raise HTTPException(status_code=500, detail="Failed to generate response")

# API Endpoints
@app.post("/chat", response_model=ChatResponse)
async def chat(chat_message: ChatMessage):
    """Main chat endpoint"""
    try:
        message = chat_message.message
        user_id = chat_message.user_id
        
        logger.info(f"Received message from {user_id}: {message}")
        
        # Step 1: Decide if we need to retrieve memories
        should_retrieve, decision_reason = await decide_memory_retrieval(message, user_id)
        
        # Step 2: Retrieve memories if needed
        memories = []
        if should_retrieve:
            memories = await retrieve_memories(message, user_id)
            logger.info(f"Retrieved {len(memories)} memories")
        
        # Step 3: Generate response
        response = await generate_response(message, memories)
        
        # Step 4: Store this conversation in memory
        await store_memory(message, user_id, response)
        
        return ChatResponse(
            response=response,
            memories_used=memories,
            memory_retrieval_decision=decision_reason,
            model_info={
                "llm_model": "Google Gemini 1.5 Flash",
                "embedding_model": EMBEDDING_MODEL_NAME,
                "vector_store": "ChromaDB"
            }
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memories/{user_id}")
async def get_user_memories(user_id: str, limit: int = 10):
    """Get all memories for a user"""
    try:
        results = collection.get(
            where={"user_id": user_id},
            limit=limit
        )
        
        memories = []
        if results['documents']:
            for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
                memories.append(MemoryItem(
                    id=results['ids'][i],
                    content=doc,
                    timestamp=metadata.get('timestamp', ''),
                    user_id=metadata.get('user_id', user_id)
                ))
        
        return {"memories": memories, "total": len(memories)}
    except Exception as e:
        logger.error(f"Error retrieving user memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve memories")

@app.delete("/memories/{user_id}")
async def clear_user_memories(user_id: str):
    """Clear all memories for a user"""
    try:
        # Get all memory IDs for the user
        results = collection.get(where={"user_id": user_id})
        
        if results['ids']:
            collection.delete(ids=results['ids'])
            logger.info(f"Cleared {len(results['ids'])} memories for user {user_id}")
            return {"message": f"Cleared {len(results['ids'])} memories", "user_id": user_id}
        else:
            return {"message": "No memories found for user", "user_id": user_id}
            
    except Exception as e:
        logger.error(f"Error clearing memories: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear memories")

@app.get("/models")
async def get_model_info():
    """Get information about the models being used"""
    return {
        "llm_model": "Google Gemini 1.5 Flash",
        "embedding_model": EMBEDDING_MODEL_NAME,
        "vector_store": "ChromaDB",
        "embedding_dimensions": len(embedding_model.encode("test")),
        "status": "active"
    }

@app.post("/change-embedding-model")
async def change_embedding_model(model_name: str):
    """Change the embedding model (for experimentation)"""
    global embedding_model, EMBEDDING_MODEL_NAME
    try:
        logger.info(f"Attempting to load new embedding model: {model_name}")
        new_model = SentenceTransformer(model_name)
        embedding_model = new_model
        EMBEDDING_MODEL_NAME = model_name
        logger.info(f"Successfully changed embedding model to: {model_name}")
        return {
            "message": f"Successfully changed embedding model to {model_name}",
            "embedding_dimensions": len(embedding_model.encode("test"))
        }
    except Exception as e:
        logger.error(f"Failed to change embedding model: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load model {model_name}: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test embedding model
        test_embedding = embedding_model.encode("health check")
        
        # Test Gemini (with a simple prompt)
        test_response = gemini_model.generate_content("Say 'OK' if you're working.")
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "models": {
                "llm": "Google Gemini 1.5 Flash - OK",
                "embedding": f"{EMBEDDING_MODEL_NAME} - OK",
                "vector_store": "ChromaDB - OK"
            },
            "embedding_test": len(test_embedding),
            "gemini_test": "OK" if test_response.text else "Failed"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)