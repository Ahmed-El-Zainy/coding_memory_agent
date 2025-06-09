import os
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Server is running!", "api_key_set": bool(os.getenv("GOOGLE_API_KEY"))}

@app.get("/test")
async def test():
    return {"status": "ok"}

if __name__ == "__main__":
    print("Starting test server...")
    print(f"API key set: {bool(os.getenv('GOOGLE_API_KEY'))}")
    uvicorn.run(app, host="0.0.0.0", port=8000)