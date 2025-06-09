# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from typing import List, Dict, Optional
# import argparse
# import traceback
# from utilities import MemoryStore, GeminiClient, load_config

# app = FastAPI(title="Chatbot Memory API", version="1.0.0")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# memory_store = MemoryStore()
# config = load_config()

# if not config["api_key"]:
#     raise ValueError("GOOGLE_API_KEY environment variable is required")

# gemini_client = GeminiClient(config["api_key"])

# class ChatRequest(BaseModel):
#     message: str
#     user_id: Optional[str] = "default"

# class ChatResponse(BaseModel):
#     response: str
#     memories_used: List[Dict]

# @app.post("/chat", response_model=ChatResponse)
# async def chat(request: ChatRequest):
#     try:
#         message = request.message
#         print(f"Received message: {message}")
        
#         message_embedding = gemini_client.get_embedding(message)
#         print(f"Generated embedding of length: {len(message_embedding)}")
        
#         memories_used = []
#         if gemini_client.should_retrieve_memory(message):
#             memories_used = memory_store.search_similar(
#                 message_embedding, 
#                 top_k=config["max_memories"],
#                 threshold=config["memory_threshold"]
#             )
#             print(f"Retrieved {len(memories_used)} memories")
        
#         response = gemini_client.generate_response(message, memories_used)
#         print(f"Generated response: {response[:100]}...")
        
#         memory_store.add_memory(
#             text=f"User: {message}\nBot: {response}",
#             embedding=message_embedding,
#             metadata={"user_id": request.user_id}
#         )
        
#         return ChatResponse(response=response, memories_used=memories_used)
    
#     except Exception as e:
#         print(f"Error in chat endpoint: {str(e)}")
#         print(traceback.format_exc())
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/")
# async def root():
#     return {"message": "Chatbot Memory API", "status": "running"}

# @app.get("/health")
# async def health():
#     return {"status": "healthy", "memories": len(memory_store.memories)}

# @app.get("/memories")
# async def get_memories():
#     return {"count": len(memory_store.memories), "memories": memory_store.memories[-10:]}

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--host", default="0.0.0.0")
#     parser.add_argument("--port", type=int, default=8000)
#     args = parser.parse_args()
    
#     import uvicorn
#     uvicorn.run(app, host=args.host, port=args.port)


from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import argparse
import traceback
import time
import asyncio
from datetime import datetime
import os
import json
import aiofiles
import uvicorn
from utilities import (
    AdvancedMemoryStore, EnhancedGeminiClient, ConfigManager, 
    DatabaseManager, ImageProcessor, CacheManager, PerformanceMonitor
)

app = FastAPI(
    title="Advanced Chatbot Memory API", 
    version="2.0.0",
    description="Enterprise-grade chatbot with advanced memory, image processing, and analytics"
)

config_manager = ConfigManager()
config = config_manager.config

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api', {}).get('cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer(auto_error=False)

memory_store = AdvancedMemoryStore(max_memories=config.get('memory', {}).get('max_total_memories', 10000))
db_manager = DatabaseManager()
image_processor = ImageProcessor(config.get('ui', {}).get('image_processing', {}))
cache_manager = CacheManager(
    max_size=config.get('performance', {}).get('caching', {}).get('cache_size', 1000),
    ttl=config.get('performance', {}).get('caching', {}).get('cache_ttl', 3600)
)
performance_monitor = PerformanceMonitor()

if not config["google_api_key"]:
    raise ValueError("GOOGLE_API_KEY is required in configuration")

gemini_client = EnhancedGeminiClient(
    config["google_api_key"],
    config.get('models', {}).get('language', {}).get('primary', 'gemini-1.5-flash')
)

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    user_id: Optional[str] = "default"
    session_id: Optional[str] = None
    include_recent: Optional[bool] = True
    image_analysis: Optional[str] = None

class ImageChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    user_id: Optional[str] = "default"
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    memories_used: List[Dict[str, Any]]
    recent_messages: List[Dict[str, Any]]
    session_id: str
    processing_time: float
    timestamp: str
    image_analysis: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    uptime: float
    memory_stats: Dict[str, Any]
    performance_stats: Dict[str, Any]
    components: Dict[str, str]

class UserStatsResponse(BaseModel):
    user_id: str
    total_messages: int
    recent_messages: List[Dict[str, Any]]
    memory_count: int
    last_activity: str

class ConfigResponse(BaseModel):
    models: Dict[str, Any]
    memory_settings: Dict[str, Any]
    ui_features: Dict[str, Any]

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if config.get('api', {}).get('authentication', {}).get('enabled', False):
        if not credentials:
            raise HTTPException(status_code=401, detail="Authentication required")
    return "authenticated_user"

@app.middleware("http")
async def performance_middleware(request, call_next):
    start_time = time.time()
    try:
        response = await call_next(request)
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time)
        return response
    except Exception as e:
        processing_time = time.time() - start_time
        performance_monitor.record_request(processing_time, error=True)
        raise

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, current_user: str = Depends(get_current_user)):
    start_time = time.time()
    
    try:
        message = request.message.strip()
        user_id = request.user_id
        session_id = request.session_id or db_manager.create_session(user_id)
        
        cache_key = f"chat:{user_id}:{hash(message)}"
        cached_response = cache_manager.get(cache_key)
        
        if cached_response and config.get('performance', {}).get('caching', {}).get('enabled', True):
            performance_monitor.record_cache_hit()
            return cached_response
        
        performance_monitor.record_cache_miss()
        
        message_embedding = await gemini_client.get_embedding_async(message)
        
        memories_used = []
        if gemini_client.should_retrieve_memory(message):
            memories_used = memory_store.search_similar(
                message_embedding,
                top_k=config.get('memory', {}).get('max_memories_per_query', 5),
                threshold=config.get('memory', {}).get('memory_relevance_threshold', 0.75),
                user_id=user_id
            )
        
        recent_messages = []
        if request.include_recent:
            recent_messages = db_manager.get_recent_messages(
                user_id, 
                limit=config.get('memory', {}).get('recent_messages', {}).get('max_display', 10)
            )
        
        response = await gemini_client.generate_response_async(
            message, memories_used, None, recent_messages
        )
        
        memory_store.add_memory(
            text=f"User: {message}\nBot: {response}",
            embedding=message_embedding,
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        db_manager.save_message(session_id, user_id, message, response)
        
        processing_time = time.time() - start_time
        
        chat_response = ChatResponse(
            response=response,
            memories_used=memories_used,
            recent_messages=recent_messages,
            session_id=session_id,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        if config.get('performance', {}).get('caching', {}).get('enabled', True):
            cache_manager.set(cache_key, chat_response)
        
        return chat_response
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat/image", response_model=ChatResponse)
async def chat_with_image(
    message: str = Form(...),
    user_id: str = Form("default"),
    session_id: Optional[str] = Form(None),
    image: UploadFile = File(...),
    current_user: str = Depends(get_current_user)
):
    start_time = time.time()
    
    try:
        if not config.get('ui', {}).get('features', {}).get('image_upload', True):
            raise HTTPException(status_code=400, detail="Image upload disabled")
        
        image_data = await image.read()
        
        is_valid, validation_msg = image_processor.validate_image(image_data, image.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_msg)
        
        image_path, metadata = image_processor.process_image(image_data, image.filename)
        encoded_image = image_processor.encode_image_for_api(image_path)
        
        session_id = session_id or db_manager.create_session(user_id)
        
        message_embedding = await gemini_client.get_embedding_async(message)
        
        memories_used = []
        if gemini_client.should_retrieve_memory(message):
            memories_used = memory_store.search_similar(
                message_embedding,
                top_k=config.get('memory', {}).get('max_memories_per_query', 5),
                threshold=config.get('memory', {}).get('memory_relevance_threshold', 0.75),
                user_id=user_id
            )
        
        recent_messages = db_manager.get_recent_messages(user_id, limit=5)
        
        image_analysis = gemini_client.analyze_image(encoded_image, message)
        
        response = await gemini_client.generate_response_async(
            f"{message}\n\nImage analysis: {image_analysis}",
            memories_used,
            encoded_image,
            recent_messages
        )
        
        memory_store.add_memory(
            text=f"User: {message} [with image: {image.filename}]\nBot: {response}",
            embedding=message_embedding,
            metadata={
                "user_id": user_id,
                "session_id": session_id,
                "has_image": True,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        db_manager.save_message(
            session_id, user_id, message, response, 
            message_type='image', 
            metadata={"image_path": image_path, "image_analysis": image_analysis}
        )
        
        file_id = db_manager.save_uploaded_file(
            user_id, image.filename, image_path, 
            image.content_type, len(image_data)
        )
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=response,
            memories_used=memories_used,
            recent_messages=recent_messages,
            session_id=session_id,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat(),
            image_analysis=image_analysis
        )
    
    except Exception as e:
        print(f"Error in image chat endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "message": "Advanced Chatbot Memory API",
        "status": "running",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health", response_model=HealthResponse)
async def health():
    stats = performance_monitor.get_stats()
    memory_stats = memory_store.get_memory_stats()
    
    return HealthResponse(
        status="healthy",
        uptime=stats['uptime_seconds'],
        memory_stats=memory_stats,
        performance_stats=stats,
        components={
            "database": "connected",
            "memory_store": "active",
            "gemini_client": "ready",
            "cache": "active"
        }
    )

@app.get("/memories")
async def get_memories(
    user_id: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=100),
    current_user: str = Depends(get_current_user)
):
    try:
        if user_id:
            user_memories = [m for m in memory_store.memories 
                           if m.get('metadata', {}).get('user_id') == user_id]
            return {
                "count": len(user_memories),
                "memories": user_memories[-limit:],
                "user_id": user_id
            }
        
        return {
            "count": len(memory_store.memories),
            "memories": memory_store.memories[-limit:],
            "total_capacity": memory_store.max_memories
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/stats", response_model=UserStatsResponse)
async def get_user_stats(
    user_id: str,
    current_user: str = Depends(get_current_user)
):
    try:
        recent_messages = db_manager.get_recent_messages(user_id, limit=20)
        
        user_memories = [m for m in memory_store.memories 
                        if m.get('metadata', {}).get('user_id') == user_id]
        
        last_activity = "Never"
        if recent_messages:
            last_activity = recent_messages[0]['timestamp']
        
        return UserStatsResponse(
            user_id=user_id,
            total_messages=len(recent_messages),
            recent_messages=recent_messages[:10],
            memory_count=len(user_memories),
            last_activity=last_activity
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/config", response_model=ConfigResponse)
async def get_config(current_user: str = Depends(get_current_user)):
    return ConfigResponse(
        models=config.get('models', {}),
        memory_settings=config.get('memory', {}),
        ui_features=config.get('ui', {}).get('features', {})
    )

@app.post("/cache/clear")
async def clear_cache(current_user: str = Depends(get_current_user)):
    try:
        cache_manager.clear()
        return {"message": "Cache cleared successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    limit: int = Query(50, ge=1, le=200),
    current_user: str = Depends(get_current_user)
):
    try:
        import sqlite3
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT content, response, timestamp, message_type, metadata
            FROM chat_messages 
            WHERE session_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (session_id, limit))
        
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
        return {"session_id": session_id, "messages": list(reversed(messages))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics/performance")
async def get_performance_analytics(current_user: str = Depends(get_current_user)):
    try:
        stats = performance_monitor.get_stats()
        memory_stats = memory_store.get_memory_stats()
        
        return {
            "performance": stats,
            "memory": memory_stats,
            "cache": {
                "size": len(cache_manager.cache),
                "max_size": cache_manager.max_size,
                "hit_rate": stats.get('cache_hit_rate', 0)
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/users/{user_id}/data")
async def delete_user_data(
    user_id: str,
    current_user: str = Depends(get_current_user)
):
    try:
        import sqlite3
        conn = sqlite3.connect(db_manager.db_path)
        cursor = conn.cursor()
        
        cursor.execute("DELETE FROM chat_messages WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM chat_sessions WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM user_preferences WHERE user_id = ?", (user_id,))
        cursor.execute("DELETE FROM uploaded_files WHERE user_id = ?", (user_id,))
        
        conn.commit()
        conn.close()
        
        memory_store.memories = [
            m for m in memory_store.memories 
            if m.get('metadata', {}).get('user_id') != user_id
        ]
        memory_store.embeddings = [
            emb for i, emb in enumerate(memory_store.embeddings)
            if memory_store.memories[i].get('metadata', {}).get('user_id') != user_id
        ]
        memory_store._rebuild_index()
        
        return {"message": f"All data for user {user_id} has been deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=config.get('api', {}).get('host', '0.0.0.0'))
    parser.add_argument("--port", type=int, default=config.get('api', {}).get('port', 8000))
    parser.add_argument("--reload", action="store_true", default=config.get('development', {}).get('auto_reload', False))
    args = parser.parse_args()
    
    uvicorn.run(
        "fastapi_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )