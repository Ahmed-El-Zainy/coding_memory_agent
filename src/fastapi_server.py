from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import argparse
import traceback
from utilities import MemoryStore, GeminiClient, load_config

app = FastAPI(title="Chatbot Memory API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

memory_store = MemoryStore()
config = load_config()

if not config["api_key"]:
    raise ValueError("GOOGLE_API_KEY environment variable is required")

gemini_client = GeminiClient(config["api_key"])

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    response: str
    memories_used: List[Dict]

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        message = request.message
        print(f"Received message: {message}")
        
        message_embedding = gemini_client.get_embedding(message)
        print(f"Generated embedding of length: {len(message_embedding)}")
        
        memories_used = []
        if gemini_client.should_retrieve_memory(message):
            memories_used = memory_store.search_similar(
                message_embedding, 
                top_k=config["max_memories"],
                threshold=config["memory_threshold"]
            )
            print(f"Retrieved {len(memories_used)} memories")
        
        response = gemini_client.generate_response(message, memories_used)
        print(f"Generated response: {response[:100]}...")
        
        memory_store.add_memory(
            text=f"User: {message}\nBot: {response}",
            embedding=message_embedding,
            metadata={"user_id": request.user_id}
        )
        
        return ChatResponse(response=response, memories_used=memories_used)
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Chatbot Memory API", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy", "memories": len(memory_store.memories)}

@app.get("/memories")
async def get_memories():
    return {"count": len(memory_store.memories), "memories": memory_store.memories[-10:]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    
    import uvicorn
    uvicorn.run(app, host=args.host, port=args.port)