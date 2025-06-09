# """
# Optimized Memory Chatbot - Core Utilities
# Functional programming approach with improved error handling and integration
# """

# import os
# import logging
# import asyncio
# import uuid
# from datetime import datetime
# from typing import List, Optional, Dict, Any, Tuple, Callable
# from functools import wraps, reduce
# from concurrent.futures import ThreadPoolExecutor
# import json

# import google.generativeai as genai
# from sentence_transformers import SentenceTransformer
# import chromadb
# from chromadb.config import Settings
# from pydantic import BaseModel, Field
# import yaml

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)

# # Thread pool for CPU-intensive tasks
# executor = ThreadPoolExecutor(max_workers=4)

# # =====================================
# # PYDANTIC MODELS
# # =====================================

# class ChatMessage(BaseModel):
#     message: str = Field(..., min_length=1, max_length=10000)
#     user_id: str = Field(default="default_user", min_length=1, max_length=100)
#     session_id: Optional[str] = Field(default=None, max_length=100)

# class ChatResponse(BaseModel):
#     response: str
#     memories_used: List[Dict[str, Any]]
#     memory_retrieval_decision: str
#     model_info: Dict[str, str]
#     processing_time: float
#     timestamp: str

# class MemoryItem(BaseModel):
#     id: str
#     content: str
#     timestamp: str
#     user_id: str
#     session_id: Optional[str] = None
#     relevance_score: Optional[float] = None

# class ModelChangeRequest(BaseModel):
#     model_name: str = Field(..., min_length=1)

# class HealthCheck(BaseModel):
#     status: str
#     timestamp: str
#     components: Dict[str, Dict[str, Any]]
#     overall_health: bool

# # =====================================
# # FUNCTIONAL UTILITIES
# # =====================================

# def safe_execute(func: Callable, *args, **kwargs) -> Tuple[Any, Optional[str]]:
#     """Safely execute a function and return result with error handling"""
#     try:
#         result = func(*args, **kwargs)
#         return result, None
#     except Exception as e:
#         logger.error(f"Error executing {func.__name__}: {e}")
#         return None, str(e)

# def compose(*functions):
#     """Compose multiple functions into a single function"""
#     return reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

# def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
#     """Decorator for retrying functions with exponential backoff"""
#     def decorator(func):
#         @wraps(func)
#         async def wrapper(*args, **kwargs):
#             for attempt in range(max_retries):
#                 try:
#                     return await func(*args, **kwargs)
#                 except Exception as e:
#                     if attempt == max_retries - 1:
#                         raise e
#                     delay = base_delay * (2 ** attempt)
#                     logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {delay}s...")
#                     await asyncio.sleep(delay)
#         return wrapper
#     return decorator

# def validate_input(validator: Callable[[Any], bool], error_message: str):
#     """Decorator for input validation"""
#     def decorator(func):
#         @wraps(func)
#         def wrapper(*args, **kwargs):
#             if not validator(*args, **kwargs):
#                 raise ValueError(error_message)
#             return func(*args, **kwargs)
#         return wrapper
#     return decorator

# # =====================================
# # CONFIGURATION MANAGEMENT
# # =====================================

# def load_config(config_path: str) -> Dict[str, Any]:
#     """Load configuration from YAML file with validation"""
#     def validate_config(config: Dict[str, Any]) -> bool:
#         required_keys = ['google_api_key']
#         return all(key in config for key in required_keys)
    
#     try:
#         with open(config_path, 'r') as file:
#             config = yaml.safe_load(file)
        
#         if not validate_config(config):
#             raise ValueError("Invalid configuration: missing required keys")
        
#         # Set environment variables for consistency
#         os.environ['GOOGLE_API_KEY'] = config.get('google_api_key', '')
        
#         logger.info("Configuration loaded successfully")
#         return config
#     except FileNotFoundError:
#         logger.error(f"Configuration file not found: {config_path}")
#         raise
#     except yaml.YAMLError as e:
#         logger.error(f"Error parsing YAML configuration: {e}")
#         raise

# def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
#     """Get nested configuration value using dot notation"""
#     keys = key_path.split('.')
#     value = config
    
#     for key in keys:
#         if isinstance(value, dict) and key in value:
#             value = value[key]
#         else:
#             return default
    
#     return value

# # =====================================
# # MODEL MANAGEMENT
# # =====================================

# def create_model_loader(model_configs: List[str]) -> Callable[[], Tuple[SentenceTransformer, str]]:
#     """Create a model loader function with fallback configurations"""
#     def load_model(model_name: Optional[str] = None) -> Tuple[SentenceTransformer, str]:
#         models_to_try = [model_name] if model_name else model_configs
        
#         for model_config in models_to_try:
#             if not model_config:
#                 continue
                
#             for use_auth in [False, True]:
#                 try:
#                     logger.info(f"Loading model: {model_config} (auth: {use_auth})")
#                     model = SentenceTransformer(
#                         model_config, 
#                         use_auth_token=use_auth if use_auth else None
#                     )
#                     logger.info(f"Successfully loaded: {model_config}")
#                     return model, model_config
#                 except Exception as e:
#                     logger.warning(f"Failed to load {model_config} with auth={use_auth}: {e}")
#                     continue
        
#         raise Exception(f"Failed to load any embedding model from: {models_to_try}")
    
#     return load_model

# def load_embedding_model(model_name: Optional[str] = None) -> Tuple[SentenceTransformer, str]:
#     """Load embedding model with comprehensive fallback strategy"""
#     model_configs = [
#         model_name,
#         os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
#         "sentence-transformers/all-MiniLM-L6-v2",
#         "all-MiniLM-L6-v2",
#         "paraphrase-MiniLM-L6-v2",
#         "all-mpnet-base-v2",
#         "distilbert-base-nli-mean-tokens"
#     ]
    
#     loader = create_model_loader([m for m in model_configs if m])
#     return loader()

# def initialize_gemini(api_key: str) -> genai.GenerativeModel:
#     """Initialize Gemini model with validation"""
#     if not api_key or len(api_key) < 10:
#         raise ValueError("Valid GEMINI_API_KEY is required")
    
#     try:
#         genai.configure(api_key=api_key)
#         model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
#         # Test the model
#         test_response = model.generate_content("Test")
#         if not test_response.text:
#             raise Exception("Model test failed")
        
#         logger.info("Gemini model initialized successfully")
#         return model
    
#     except Exception as e:
#         logger.error(f"Failed to initialize Gemini: {e}")
#         raise

# def initialize_chromadb(persist_directory: str = "./chroma_db") -> chromadb.Collection:
#     """Initialize ChromaDB with proper error handling"""
#     try:
#         # Ensure directory exists
#         os.makedirs(persist_directory, exist_ok=True)
        
#         client = chromadb.PersistentClient(path=persist_directory)
        
#         collection = client.get_or_create_collection(
#             name="chat_memories",
#             metadata={"hnsw:space": "cosine"}
#         )
        
#         logger.info(f"ChromaDB initialized with {collection.count()} existing memories")
#         return collection
    
#     except Exception as e:
#         logger.error(f"Failed to initialize ChromaDB: {e}")
#         raise

# # =====================================
# # EMBEDDING OPERATIONS
# # =====================================

# def create_embedding_function(model: SentenceTransformer) -> Callable[[str], List[float]]:
#     """Create a reusable embedding function"""
#     def get_embedding(text: str) -> List[float]:
#         if not text or not text.strip():
#             raise ValueError("Text cannot be empty")
        
#         try:
#             embedding = model.encode(text.strip(), convert_to_tensor=False)
#             return embedding.tolist()
#         except Exception as e:
#             logger.error(f"Error generating embedding: {e}")
#             raise
    
#     return get_embedding

# async def get_embedding_async(text: str, model: SentenceTransformer) -> List[float]:
#     """Async wrapper for embedding generation"""
#     embedding_func = create_embedding_function(model)
#     loop = asyncio.get_event_loop()
    
#     try:
#         embedding = await loop.run_in_executor(executor, embedding_func, text)
#         return embedding
#     except Exception as e:
#         logger.error(f"Error in async embedding: {e}")
#         raise

# # =====================================
# # MEMORY OPERATIONS
# # =====================================

# @retry_with_backoff(max_retries=3)
# async def decide_memory_retrieval(
#     message: str, 
#     user_id: str, 
#     gemini_model: genai.GenerativeModel,
#     context: Optional[Dict[str, Any]] = None
# ) -> Tuple[bool, str]:
#     """Enhanced memory retrieval decision with context awareness"""
    
#     # Quick heuristic checks
#     if len(message.strip()) < 3:
#         return False, "Message too short for memory retrieval"
    
#     # Keywords that suggest memory retrieval is needed
#     memory_keywords = [
#         'remember', 'recall', 'before', 'previous', 'earlier', 'last time',
#         'we discussed', 'you said', 'my preference', 'favorite', 'usual'
#     ]
    
#     if any(keyword in message.lower() for keyword in memory_keywords):
#         return True, "Memory retrieval triggered by keywords"
    
#     try:
#         context_info = ""
#         if context:
#             context_info = f"\nContext: {json.dumps(context, indent=2)}"
        
#         decision_prompt = f"""
#         Analyze if retrieving past conversation memories would help answer this message.
        
#         Current message: "{message}"
#         User ID: {user_id}{context_info}
        
#         Retrieve memories if:
#         - User references past conversations
#         - Question needs personal context/preferences
#         - Continuation of previous topics
#         - User asks about their information
        
#         Don't retrieve if:
#         - Simple greetings/casual chat
#         - Self-contained questions
#         - Generic requests
#         - First-time interactions
        
#         Respond: YES/NO [brief reason]
#         """
        
#         response = gemini_model.generate_content(decision_prompt)
#         decision_text = response.text.strip()
#         should_retrieve = decision_text.upper().startswith("YES")
        
#         logger.info(f"Memory decision for '{message[:50]}...': {decision_text}")
#         return should_retrieve, decision_text
        
#     except Exception as e:
#         logger.error(f"Error in memory decision: {e}")
#         return False, f"Error in decision making: {str(e)}"

# async def store_memory(
#     message: str, 
#     user_id: str, 
#     response: str,
#     collection: chromadb.Collection,
#     embedding_model: SentenceTransformer,
#     session_id: Optional[str] = None
# ) -> str:
#     """Store conversation with enhanced metadata"""
    
#     try:
#         # Create comprehensive memory content
#         memory_content = f"User: {message}\nAssistant: {response}"
        
#         # Generate embedding
#         embedding = await get_embedding_async(memory_content, embedding_model)
        
#         # Create memory record
#         memory_id = str(uuid.uuid4())
#         timestamp = datetime.now().isoformat()
        
#         metadata = {
#             "user_id": user_id,
#             "session_id": session_id or "default",
#             "timestamp": timestamp,
#             "user_message": message,
#             "assistant_response": response,
#             "message_length": len(message),
#             "response_length": len(response),
#             "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         }
        
#         # Store in vector database
#         collection.add(
#             embeddings=[embedding],
#             documents=[memory_content],
#             metadatas=[metadata],
#             ids=[memory_id]
#         )
        
#         logger.info(f"Stored memory {memory_id} for user {user_id}")
#         return memory_id
        
#     except Exception as e:
#         logger.error(f"Error storing memory: {e}")
#         raise

# async def retrieve_memories(
#     message: str, 
#     user_id: str, 
#     collection: chromadb.Collection,
#     embedding_model: SentenceTransformer,
#     n_results: int = 3,
#     relevance_threshold: float = 0.7
# ) -> List[Dict[str, Any]]:
#     """Retrieve relevant memories with enhanced filtering"""
    
#     try:
#         # Generate query embedding
#         embedding = await get_embedding_async(message, embedding_model)
        
#         # Query vector database
#         results = collection.query(
#             query_embeddings=[embedding],
#             n_results=min(n_results * 2, 10),  # Get more results for filtering
#             where={"user_id": user_id},
#             include=['documents', 'metadatas', 'distances']
#         )
        
#         memories = []
#         if results['documents'] and results['documents'][0]:
#             for doc, metadata, distance in zip(
#                 results['documents'][0], 
#                 results['metadatas'][0], 
#                 results['distances'][0]
#             ):
#                 # Calculate relevance score (cosine similarity)
#                 relevance = 1 - distance
                
#                 # Filter by relevance threshold
#                 if relevance >= relevance_threshold:
#                     # Format timestamp
#                     timestamp = metadata.get('timestamp', '')
#                     try:
#                         dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
#                         readable_time = dt.strftime('%Y-%m-%d %H:%M')
#                     except:
#                         readable_time = timestamp
                    
#                     memory_item = {
#                         'content': doc,
#                         'timestamp': readable_time,
#                         'relevance_score': round(relevance, 3),
#                         'metadata': metadata
#                     }
#                     memories.append(memory_item)
        
#         # Sort by relevance and limit results
#         memories.sort(key=lambda x: x['relevance_score'], reverse=True)
#         memories = memories[:n_results]
        
#         logger.info(f"Retrieved {len(memories)} relevant memories for user {user_id}")
#         return memories
        
#     except Exception as e:
#         logger.error(f"Error retrieving memories: {e}")
#         return []

# @retry_with_backoff(max_retries=2)
# async def generate_response(
#     message: str, 
#     memories: List[Dict[str, Any]],
#     gemini_model: genai.GenerativeModel,
#     system_context: Optional[str] = None
# ) -> str:
#     """Generate AI response with enhanced context management"""
    
#     try:
#         # Default system context
#         default_context = """You are a helpful and friendly AI assistant with access to conversation history. 
#         Use the provided memories to give contextual and personalized responses when relevant.
#         Be natural, conversational, and engaging. Reference past conversations naturally when appropriate.
#         Maintain consistency with previous interactions while being helpful and informative."""
        
#         context = system_context or default_context
        
#         # Build prompt with memories
#         prompt_parts = [context, f"\nCurrent message: {message}"]
        
#         if memories:
#             memory_context = "\n\nRelevant conversation history:"
#             for i, memory in enumerate(memories, 1):
#                 score = memory.get('relevance_score', 0)
#                 timestamp = memory.get('timestamp', 'Unknown')
#                 content = memory.get('content', '')
                
#                 memory_context += f"\n\nMemory {i} (relevance: {score}, time: {timestamp}):\n{content}"
            
#             prompt_parts.append(memory_context)
#             prompt_parts.append("\n\nPlease respond to the current message, using the conversation history for context when relevant.")
        
#         full_prompt = "".join(prompt_parts)
        
#         # Generate response
#         response = gemini_model.generate_content(full_prompt)
        
#         if not response.text:
#             raise Exception("Empty response from Gemini")
        
#         return response.text.strip()
        
#     except Exception as e:
#         logger.error(f"Error generating response: {e}")
#         raise

# # =====================================
# # HEALTH CHECK FUNCTIONS
# # =====================================

# def create_health_checker(component_name: str) -> Callable:
#     """Create a health check function for a specific component"""
#     def health_check(component: Any, **kwargs) -> Dict[str, Any]:
#         try:
#             # Component-specific health checks
#             if component_name == "embedding":
#                 model = component
#                 test_embedding = model.encode("health check")
#                 return {
#                     "status": "OK",
#                     "component": component_name,
#                     "embedding_dimensions": len(test_embedding),
#                     "details": kwargs
#                 }
            
#             elif component_name == "gemini":
#                 model = component
#                 test_response = model.generate_content("Say 'OK' if you're working.")
#                 return {
#                     "status": "OK" if test_response.text else "ERROR",
#                     "component": component_name,
#                     "details": kwargs
#                 }
            
#             elif component_name == "chromadb":
#                 collection = component
#                 count = collection.count()
#                 return {
#                     "status": "OK",
#                     "component": component_name,
#                     "memory_count": count,
#                     "details": kwargs
#                 }
            
#             else:
#                 return {
#                     "status": "UNKNOWN",
#                     "component": component_name,
#                     "details": kwargs
#                 }
        
#         except Exception as e:
#             return {
#                 "status": "ERROR",
#                 "component": component_name,
#                 "error": str(e),
#                 "details": kwargs
#             }
    
#     return health_check

# def comprehensive_health_check(
#     embedding_model: SentenceTransformer,
#     embedding_model_name: str,
#     gemini_model: genai.GenerativeModel,
#     collection: chromadb.Collection
# ) -> HealthCheck:
#     """Perform comprehensive health check of all components"""
    
#     # Create health checkers
#     check_embedding = create_health_checker("embedding")
#     check_gemini = create_health_checker("gemini")
#     check_chromadb = create_health_checker("chromadb")
    
#     # Perform checks
#     embedding_health = check_embedding(embedding_model, model_name=embedding_model_name)
#     gemini_health = check_gemini(gemini_model)
#     chromadb_health = check_chromadb(collection)
    
#     # Determine overall health
#     all_healthy = all(
#         check["status"] == "OK" 
#         for check in [embedding_health, gemini_health, chromadb_health]
#     )
    
#     return HealthCheck(
#         status="healthy" if all_healthy else "unhealthy",
#         timestamp=datetime.now().isoformat(),
#         components={
#             "embedding": embedding_health,
#             "gemini": gemini_health,
#             "chromadb": chromadb_health
#         },
#         overall_health=all_healthy
#     )

# # =====================================
# # MEMORY MANAGEMENT UTILITIES
# # =====================================

# def get_user_memories(
#     user_id: str, 
#     collection: chromadb.Collection, 
#     limit: int = 10,
#     session_id: Optional[str] = None
# ) -> List[MemoryItem]:
#     """Get user memories with optional session filtering"""
    
#     try:
#         where_clause = {"user_id": user_id}
#         if session_id:
#             where_clause["session_id"] = session_id
        
#         results = collection.get(
#             where=where_clause,
#             limit=limit
#         )
        
#         memories = []
#         if results['documents']:
#             for i, (doc, metadata) in enumerate(zip(results['documents'], results['metadatas'])):
#                 memories.append(MemoryItem(
#                     id=results['ids'][i],
#                     content=doc,
#                     timestamp=metadata.get('timestamp', ''),
#                     user_id=metadata.get('user_id', user_id),
#                     session_id=metadata.get('session_id')
#                 ))
        
#         # Sort by timestamp (newest first)
#         memories.sort(key=lambda x: x.timestamp, reverse=True)
#         return memories
        
#     except Exception as e:
#         logger.error(f"Error retrieving user memories: {e}")
#         raise

# def clear_user_memories(
#     user_id: str, 
#     collection: chromadb.Collection,
#     session_id: Optional[str] = None
# ) -> Dict[str, Any]:
#     """Clear user memories with optional session filtering"""
    
#     try:
#         where_clause = {"user_id": user_id}
#         if session_id:
#             where_clause["session_id"] = session_id
        
#         results = collection.get(where=where_clause)
        
#         if results['ids']:
#             collection.delete(ids=results['ids'])
#             logger.info(f"Cleared {len(results['ids'])} memories for user {user_id}")
            
#             return {
#                 "message": f"Successfully cleared {len(results['ids'])} memories",
#                 "user_id": user_id,
#                 "session_id": session_id,
#                 "count": len(results['ids']),
#                 "timestamp": datetime.now().isoformat()
#             }
#         else:
#             return {
#                 "message": "No memories found to clear",
#                 "user_id": user_id,
#                 "session_id": session_id,
#                 "count": 0,
#                 "timestamp": datetime.now().isoformat()
#             }
    
#     except Exception as e:
#         logger.error(f"Error clearing memories: {e}")
#         raise

# # =====================================
# # PERFORMANCE MONITORING
# # =====================================

# def measure_performance(func):
#     """Decorator to measure function performance"""
#     @wraps(func)
#     async def wrapper(*args, **kwargs):
#         start_time = datetime.now()
#         try:
#             result = await func(*args, **kwargs)
#             end_time = datetime.now()
#             processing_time = (end_time - start_time).total_seconds()
            
#             logger.info(f"{func.__name__} completed in {processing_time:.3f}s")
            
#             # Add timing info to result if it's a dict
#             if isinstance(result, dict):
#                 result['processing_time'] = processing_time
#                 result['timestamp'] = end_time.isoformat()
            
#             return result
            
#         except Exception as e:
#             end_time = datetime.now()
#             processing_time = (end_time - start_time).total_seconds()
#             logger.error(f"{func.__name__} failed after {processing_time:.3f}s: {e}")
#             raise
    
#     return wrapper

# # =====================================
# # INITIALIZATION FUNCTION
# # =====================================

# async def initialize_application(config_path: str) -> Dict[str, Any]:
#     """Initialize all application components"""
    
#     try:
#         logger.info("Starting application initialization...")
        
#         # Load configuration
#         config = load_config(config_path)
        
#         # Initialize models
#         embedding_model, embedding_model_name = load_embedding_model()
#         gemini_model = initialize_gemini(config['google_api_key'])
        
#         # Initialize ChromaDB
#         persist_dir = get_config_value(config, 'chromadb.persist_directory', './chroma_db')
#         collection = initialize_chromadb(persist_dir)
        
#         # Perform health check
#         health_status = comprehensive_health_check(
#             embedding_model, embedding_model_name, gemini_model, collection
#         )
        
#         if not health_status.overall_health:
#             logger.warning("Some components are not healthy!")
        
#         components = {
#             'config': config,
#             'embedding_model': embedding_model,
#             'embedding_model_name': embedding_model_name,
#             'gemini_model': gemini_model,
#             'collection': collection,
#             'health_status': health_status
#         }
        
#         logger.info("Application initialization completed successfully!")
#         return components
        
#     except Exception as e:
#         logger.error(f"Application initialization failed: {e}")
#         raise


import os
import json
import numpy as np
from typing import List, Dict, Any
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime

class MemoryStore:
    def __init__(self):
        self.memories = []
        self.embeddings = []
        
    def add_memory(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
        memory = {
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        self.memories.append(memory)
        self.embeddings.append(embedding)
    
    def search_similar(self, query_embedding: List[float], top_k: int = 3, threshold: float = 0.7) -> List[Dict]:
        if not self.embeddings:
            return []
        
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        relevant_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
        relevant_indices.sort(key=lambda i: similarities[i], reverse=True)
        
        return [self.memories[i] for i in relevant_indices[:top_k]]

class GeminiClient:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_response(self, prompt: str, memories: List[Dict] = None) -> str:
        try:
            context = ""
            if memories:
                context = "\nRelevant memories:\n" + "\n".join([f"- {m['text']}" for m in memories])
            
            full_prompt = f"{prompt}{context}"
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            print(f"Generation error: {e}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def get_embedding(self, text: str) -> List[float]:
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            print(f"Embedding error: {e}")
            return [0.0] * 768
    
    def should_retrieve_memory(self, message: str) -> bool:
        try:
            prompt = f"Does this message reference past conversations or need context? Answer only 'yes' or 'no': '{message}'"
            response = self.model.generate_content(prompt)
            return 'yes' in response.text.lower()
        except Exception as e:
            print(f"Memory decision error: {e}")
            return len(message.split()) > 5

def load_config():
    return {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "memory_threshold": float(os.getenv("MEMORY_THRESHOLD", "0.7")),
        "max_memories": int(os.getenv("MAX_MEMORIES", "3"))
    }