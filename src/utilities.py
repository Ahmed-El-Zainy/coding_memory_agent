import os
import json
import yaml
import sqlite3
import hashlib
import base64
from PIL import Image
import io
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import logging
from functools import lru_cache
import time
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseManager:
    def __init__(self, db_path: str = "./chat_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_activity TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_messages (
                id TEXT PRIMARY KEY,
                session_id TEXT,
                user_id TEXT NOT NULL,
                message_type TEXT DEFAULT 'text',
                content TEXT NOT NULL,
                response TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                user_id TEXT PRIMARY KEY,
                preferences TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                file_type TEXT,
                file_size INTEGER,
                file_path TEXT,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, user_id: str) -> str:
        session_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO chat_sessions (id, user_id) VALUES (?, ?)",
            (session_id, user_id)
        )
        conn.commit()
        conn.close()
        return session_id
    
    def save_message(self, session_id: str, user_id: str, message: str, 
                    response: str, message_type: str = 'text', metadata: Dict = None):
        message_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_messages (id, session_id, user_id, message_type, content, response, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (message_id, session_id, user_id, message_type, message, response, json.dumps(metadata or {})))
        conn.commit()
        conn.close()
        return message_id
    
    def get_recent_messages(self, user_id: str, limit: int = 10) -> List[Dict]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            SELECT content, response, timestamp, message_type, metadata
            FROM chat_messages 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'user_message': row[0],
                'bot_response': row[1],
                'timestamp': row[2],
                'type': row[3],
                'metadata': json.loads(row[4]) if row[4] else {}
            })
        
        conn.close()
        return list(reversed(messages))
    
    def save_uploaded_file(self, user_id: str, filename: str, file_path: str, 
                          file_type: str, file_size: int) -> str:
        file_id = str(uuid.uuid4())
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO uploaded_files (id, user_id, filename, file_type, file_size, file_path)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (file_id, user_id, filename, file_type, file_size, file_path))
        conn.commit()
        conn.close()
        return file_id

class ImageProcessor:
    def __init__(self, config: Dict):
        self.max_size = config.get('max_file_size', 10485760)
        self.allowed_formats = config.get('allowed_formats', ['jpg', 'jpeg', 'png', 'gif', 'webp'])
        self.max_dimensions = config.get('max_dimensions', [1024, 1024])
        self.upload_dir = "./uploads"
        os.makedirs(self.upload_dir, exist_ok=True)
    
    def validate_image(self, file_data: bytes, filename: str) -> Tuple[bool, str]:
        if len(file_data) > self.max_size:
            return False, "File too large"
        
        ext = filename.lower().split('.')[-1]
        if ext not in self.allowed_formats:
            return False, f"Format not supported. Allowed: {self.allowed_formats}"
        
        try:
            Image.open(io.BytesIO(file_data))
            return True, "Valid image"
        except Exception:
            return False, "Invalid image file"
    
    def process_image(self, file_data: bytes, filename: str) -> Tuple[str, Dict]:
        try:
            image = Image.open(io.BytesIO(file_data))
            
            original_size = image.size
            if image.size[0] > self.max_dimensions[0] or image.size[1] > self.max_dimensions[1]:
                image.thumbnail(self.max_dimensions, Image.Resampling.LANCZOS)
            
            file_id = hashlib.md5(file_data).hexdigest()
            save_path = os.path.join(self.upload_dir, f"{file_id}_{filename}")
            
            if image.mode in ('RGBA', 'LA', 'P'):
                image = image.convert('RGB')
            
            image.save(save_path, 'JPEG', quality=85, optimize=True)
            
            metadata = {
                'original_size': original_size,
                'processed_size': image.size,
                'format': image.format,
                'mode': image.mode,
                'file_path': save_path
            }
            
            return save_path, metadata
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise
    
    def encode_image_for_api(self, image_path: str) -> str:
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

class AdvancedMemoryStore:
    def __init__(self, max_memories: int = 10000):
        self.memories = []
        self.embeddings = []
        self.max_memories = max_memories
        self.memory_index = {}
        
    def add_memory(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
        if len(self.memories) >= self.max_memories:
            self._cleanup_old_memories()
        
        memory_id = str(uuid.uuid4())
        memory = {
            "id": memory_id,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "access_count": 0,
            "last_accessed": datetime.now().isoformat()
        }
        
        self.memories.append(memory)
        self.embeddings.append(embedding)
        self.memory_index[memory_id] = len(self.memories) - 1
    
    def search_similar(self, query_embedding: List[float], top_k: int = 5, 
                      threshold: float = 0.7, user_id: str = None) -> List[Dict]:
        if not self.embeddings:
            return []
        
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]
        
        scored_memories = []
        for i, (sim, memory) in enumerate(zip(similarities, self.memories)):
            if sim > threshold:
                if user_id and memory['metadata'].get('user_id') != user_id:
                    continue
                
                memory['similarity_score'] = float(sim)
                memory['access_count'] += 1
                memory['last_accessed'] = datetime.now().isoformat()
                scored_memories.append(memory)
        
        scored_memories.sort(key=lambda x: x['similarity_score'], reverse=True)
        return scored_memories[:top_k]
    
    def _cleanup_old_memories(self):
        if len(self.memories) < self.max_memories * 0.8:
            return
        
        cutoff_date = datetime.now() - timedelta(days=30)
        
        indices_to_remove = []
        for i, memory in enumerate(self.memories):
            memory_date = datetime.fromisoformat(memory['timestamp'])
            if memory_date < cutoff_date and memory['access_count'] < 2:
                indices_to_remove.append(i)
        
        for i in reversed(indices_to_remove):
            del self.memories[i]
            del self.embeddings[i]
        
        self._rebuild_index()
    
    def _rebuild_index(self):
        self.memory_index = {}
        for i, memory in enumerate(self.memories):
            self.memory_index[memory['id']] = i
    
    def get_memory_stats(self) -> Dict[str, Any]:
        total_memories = len(self.memories)
        if total_memories == 0:
            return {"total": 0, "avg_similarity": 0, "recent_count": 0}
        
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_count = sum(1 for m in self.memories 
                          if datetime.fromisoformat(m['timestamp']) > recent_cutoff)
        
        return {
            "total": total_memories,
            "recent_count": recent_count,
            "memory_utilization": f"{(total_memories / self.max_memories) * 100:.1f}%"
        }

class EnhancedGeminiClient:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.embedding_model = "models/text-embedding-004"
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            max_output_tokens=2048
        )
    
    async def generate_response_async(self, prompt: str, memories: List[Dict] = None, 
                                    image_data: str = None, recent_messages: List[Dict] = None) -> str:
        try:
            context_parts = []
            
            if recent_messages:
                context_parts.append("Recent conversation:")
                for msg in recent_messages[-3:]:
                    context_parts.append(f"User: {msg['user_message']}")
                    context_parts.append(f"Assistant: {msg['bot_response']}")
                context_parts.append("")
            
            if memories:
                context_parts.append("Relevant memories:")
                for memory in memories:
                    score = memory.get('similarity_score', 0)
                    context_parts.append(f"- {memory['text']} (relevance: {score:.2f})")
                context_parts.append("")
            
            if image_data:
                content_parts = [
                    "\n".join(context_parts) + f"\nCurrent message: {prompt}",
                    {
                        "mime_type": "image/jpeg",
                        "data": image_data
                    }
                ]
            else:
                content_parts = ["\n".join(context_parts) + f"\nCurrent message: {prompt}"]
            
            response = await asyncio.to_thread(
                self.model.generate_content,
                content_parts,
                generation_config=self.generation_config
            )
            
            return response.text if response.text else "I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"I encountered an error while processing your request: {str(e)}"
    
    def generate_response(self, prompt: str, memories: List[Dict] = None, 
                         image_data: str = None, recent_messages: List[Dict] = None) -> str:
        return asyncio.run(self.generate_response_async(prompt, memories, image_data, recent_messages))
    
    async def get_embedding_async(self, text: str) -> List[float]:
        try:
            result = await asyncio.to_thread(
                genai.embed_content,
                model=self.embedding_model,
                content=text,
                task_type="retrieval_document"
            )
            return result['embedding']
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return [0.0] * 768
    
    def get_embedding(self, text: str) -> List[float]:
        return asyncio.run(self.get_embedding_async(text))
    
    def should_retrieve_memory(self, message: str) -> bool:
        memory_indicators = [
            'remember', 'recall', 'before', 'previous', 'earlier', 'last time',
            'we discussed', 'you said', 'my preference', 'favorite', 'usual',
            'continue', 'follow up', 'as we talked', 'mentioned'
        ]
        
        message_lower = message.lower()
        
        if any(indicator in message_lower for indicator in memory_indicators):
            return True
        
        if len(message.split()) > 10:
            return True
        
        return False
    
    def analyze_image(self, image_data: str, question: str = None) -> str:
        try:
            prompt = question or "Describe this image in detail."
            
            response = self.model.generate_content([
                prompt,
                {
                    "mime_type": "image/jpeg", 
                    "data": image_data
                }
            ])
            
            return response.text if response.text else "Could not analyze the image."
            
        except Exception as e:
            logger.error(f"Image analysis error: {e}")
            return f"Error analyzing image: {str(e)}"

class ConfigManager:
    def __init__(self, config_path: str = "src/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            required_keys = ['google_api_key']
            for key in required_keys:
                if not config.get(key):
                    env_key = key.upper()
                    config[key] = os.getenv(env_key, config.get(key, ''))
            
            return config
        except Exception as e:
            logger.error(f"Config loading error: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        return {
            "google_api_key": os.getenv("GOOGLE_API_KEY", ""),
            "memory": {"memory_relevance_threshold": 0.75, "max_memories_per_query": 5},
            "ui": {"features": {"image_upload": True}},
            "performance": {"caching": {"enabled": True}}
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value

class CacheManager:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        self.cache = {}
        self.access_times = {}
        self.max_size = max_size
        self.ttl = ttl
    
    def get(self, key: str) -> Any:
        if key not in self.cache:
            return None
        
        if time.time() - self.access_times[key] > self.ttl:
            del self.cache[key]
            del self.access_times[key]
            return None
        
        self.access_times[key] = time.time()
        return self.cache[key]
    
    def set(self, key: str, value: Any):
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]
        
        self.cache[key] = value
        self.access_times[key] = time.time()
    
    def clear(self):
        self.cache.clear()
        self.access_times.clear()

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {
            'requests': 0,
            'total_response_time': 0,
            'errors': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self.start_time = time.time()
    
    def record_request(self, response_time: float, error: bool = False):
        self.metrics['requests'] += 1
        self.metrics['total_response_time'] += response_time
        if error:
            self.metrics['errors'] += 1
    
    def record_cache_hit(self):
        self.metrics['cache_hits'] += 1
    
    def record_cache_miss(self):
        self.metrics['cache_misses'] += 1
    
    def get_stats(self) -> Dict[str, Any]:
        uptime = time.time() - self.start_time
        avg_response_time = (self.metrics['total_response_time'] / 
                           max(self.metrics['requests'], 1))
        
        cache_total = self.metrics['cache_hits'] + self.metrics['cache_misses']
        cache_hit_rate = (self.metrics['cache_hits'] / max(cache_total, 1)) * 100
        
        return {
            'uptime_seconds': uptime,
            'total_requests': self.metrics['requests'],
            'average_response_time': avg_response_time,
            'error_rate': (self.metrics['errors'] / max(self.metrics['requests'], 1)) * 100,
            'cache_hit_rate': cache_hit_rate
        }

def load_config(config_path: str = "src/config.yaml") -> Dict[str, Any]:
    manager = ConfigManager(config_path)
    return manager.config

# import os
# import json
# import numpy as np
# from typing import List, Dict, Any
# import google.generativeai as genai
# from sklearn.metrics.pairwise import cosine_similarity
# from datetime import datetime

# class MemoryStore:
#     def __init__(self):
#         self.memories = []
#         self.embeddings = []
        
#     def add_memory(self, text: str, embedding: List[float], metadata: Dict[str, Any] = None):
#         memory = {
#             "text": text,
#             "timestamp": datetime.now().isoformat(),
#             "metadata": metadata or {}
#         }
#         self.memories.append(memory)
#         self.embeddings.append(embedding)
    
#     def search_similar(self, query_embedding: List[float], top_k: int = 3, threshold: float = 0.7) -> List[Dict]:
#         if not self.embeddings:
#             return []
        
#         similarities = cosine_similarity([query_embedding], self.embeddings)[0]
#         relevant_indices = [i for i, sim in enumerate(similarities) if sim > threshold]
#         relevant_indices.sort(key=lambda i: similarities[i], reverse=True)
        
#         return [self.memories[i] for i in relevant_indices[:top_k]]

# class GeminiClient:
#     def __init__(self, api_key: str):
#         genai.configure(api_key=api_key)
#         self.model = genai.GenerativeModel('gemini-1.5-flash')
    
#     def generate_response(self, prompt: str, memories: List[Dict] = None) -> str:
#         try:
#             context = ""
#             if memories:
#                 context = "\nRelevant memories:\n" + "\n".join([f"- {m['text']}" for m in memories])
            
#             full_prompt = f"{prompt}{context}"
#             response = self.model.generate_content(full_prompt)
#             return response.text
#         except Exception as e:
#             print(f"Generation error: {e}")
#             return f"Sorry, I encountered an error: {str(e)}"
    
#     def get_embedding(self, text: str) -> List[float]:
#         try:
#             result = genai.embed_content(
#                 model="models/text-embedding-004",
#                 content=text,
#                 task_type="retrieval_document"
#             )
#             return result['embedding']
#         except Exception as e:
#             print(f"Embedding error: {e}")
#             return [0.0] * 768
    
#     def should_retrieve_memory(self, message: str) -> bool:
#         try:
#             prompt = f"Does this message reference past conversations or need context? Answer only 'yes' or 'no': '{message}'"
#             response = self.model.generate_content(prompt)
#             return 'yes' in response.text.lower()
#         except Exception as e:
#             print(f"Memory decision error: {e}")
#             return len(message.split()) > 5

# def load_config():
#     return {
#         "api_key": os.getenv("GOOGLE_API_KEY"),
#         "memory_threshold": float(os.getenv("MEMORY_THRESHOLD", "0.7")),
#         "max_memories": int(os.getenv("MAX_MEMORIES", "3"))
#     }