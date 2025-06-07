import os 
import yaml
from sentence_transformers import SentenceTransformer
import logging
from chromadb import PersistentClient

logger = logging.getLogger(__name__)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


"""
Core utilities for the memory chatbot application
"""

import os
import logging
import asyncio
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from concurrent.futures import ThreadPoolExecutor

import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive tasks
executor = ThreadPoolExecutor(max_workers=4)

# Pydantic Models
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

class ModelChangeRequest(BaseModel):
    model_name: str

# Configuration Management
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logger.error(f"Configuration file not found: {config_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML configuration: {e}")
        raise

# Model Loading Functions
def load_embedding_model(model_name: Optional[str] = None) -> Tuple[SentenceTransformer, str]:
    """
    Load embedding model with fallbacks and proper error handling
    
    Args:
        model_name: Optional specific model name to load
        
    Returns:
        Tuple of (model, actual_model_name)
    """
    if model_name:
        models_to_try = [model_name]
    else:
        models_to_try = [
            os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
            "all-MiniLM-L6-v2",
            "sentence-transformers/all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2",
            "all-mpnet-base-v2"
        ]
    
    # First try: Load without authentication
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting to load model: {model_name}")
            model = SentenceTransformer(model_name, use_auth_token=False)
            logger.info(f"Successfully loaded model: {model_name}")
            return model, model_name
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            continue
    
    # Second try: Load with local_files_only (for cached models)
    for model_name in models_to_try:
        try:
            logger.info(f"Attempting to load model locally: {model_name}")
            model = SentenceTransformer(model_name, local_files_only=True)
            logger.info(f"Successfully loaded local model: {model_name}")
            return model, model_name
        except Exception as e:
            logger.warning(f"Failed to load local {model_name}: {e}")
            continue
    
    raise Exception("Failed to load any embedding model")

def initialize_gemini(api_key: str) -> genai.GenerativeModel:
    """Initialize Gemini model"""
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required")
    
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash-latest')

# def initialize_chromadb(persist_directory: str = "./chroma_db") -> chromadb.Collection:
#     """Initialize ChromaDB collection"""

def initialize_chromadb(persist_directory: str = "./chroma_db") -> chromadb.Collection:
    client = PersistentClient(path=persist_directory)

    try:
        collection = client.get_or_create_collection(
            name="chat_memories",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Initialized ChromaDB collection")
        return collection
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB collection: {e}")
        raise
    
    try:
        collection = client.create_collection(
            name="chat_memories",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Created new ChromaDB collection")
    except Exception:
        collection = client.get_collection("chat_memories")
        logger.info("Using existing ChromaDB collection")
    
    return collection

# Embedding Functions
def get_embedding_sync(text: str, model: SentenceTransformer) -> List[float]:
    """Get Hugging Face embedding for text (synchronous)"""
    try:
        embedding = model.encode(text, convert_to_tensor=False)
        return embedding.tolist()
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise

async def get_embedding_async(text: str, model: SentenceTransformer) -> List[float]:
    """Get Hugging Face embedding for text (async wrapper)"""
    loop = asyncio.get_event_loop()
    try:
        embedding = await loop.run_in_executor(
            executor, get_embedding_sync, text, model
        )
        return embedding
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        raise

# Memory Management Functions
async def decide_memory_retrieval(
    message: str, 
    user_id: str, 
    gemini_model: genai.GenerativeModel
) -> Tuple[bool, str]:
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

async def store_memory(
    message: str, 
    user_id: str, 
    response: str,
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer
) -> str:
    """Store conversation in vector database"""
    try:
        memory_content = f"User: {message}\nAssistant: {response}"
        embedding = await get_embedding_async(memory_content, embedding_model)
        
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
        raise

async def retrieve_memories(
    message: str, 
    user_id: str, 
    collection: chromadb.Collection,
    embedding_model: SentenceTransformer,
    n_results: int = 3
) -> List[str]:
    """Retrieve relevant memories from vector database"""
    try:
        embedding = await get_embedding_async(message, embedding_model)
        
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

async def generate_response(
    message: str, 
    memories: List[str],
    gemini_model: genai.GenerativeModel
) -> str:
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
        raise

# Health Check Functions
def health_check_embedding_model(model: SentenceTransformer, model_name: str) -> Dict[str, Any]:
    """Test embedding model health"""
    try:
        test_embedding = model.encode("health check")
        return {
            "status": "OK",
            "model_name": model_name,
            "embedding_dimensions": len(test_embedding)
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "model_name": model_name,
            "error": str(e)
        }

def health_check_gemini(model: genai.GenerativeModel) -> Dict[str, str]:
    """Test Gemini model health"""
    try:
        test_response = model.generate_content("Say 'OK' if you're working.")
        return {
            "status": "OK" if test_response.text else "ERROR",
            "model_name": "Google Gemini 1.5 Flash"
        }
    except Exception as e:
        return {
            "status": "ERROR",
            "model_name": "Google Gemini 1.5 Flash",
            "error": str(e)
        }

# Memory Management Utilities
def get_user_memories(
    user_id: str, 
    collection: chromadb.Collection, 
    limit: int = 10
) -> List[MemoryItem]:
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
        
        return memories
    except Exception as e:
        logger.error(f"Error retrieving user memories: {e}")
        raise

def clear_user_memories(user_id: str, collection: chromadb.Collection) -> Dict[str, Any]:
    """Clear all memories for a user"""
    try:
        results = collection.get(where={"user_id": user_id})
        
        if results['ids']:
            collection.delete(ids=results['ids'])
            logger.info(f"Cleared {len(results['ids'])} memories for user {user_id}")
            return {
                "message": f"Cleared {len(results['ids'])} memories",
                "user_id": user_id,
                "count": len(results['ids'])
            }
        else:
            return {
                "message": "No memories found for user",
                "user_id": user_id,
                "count": 0
            }
    except Exception as e:
        logger.error(f"Error clearing memories: {e}")
        raise