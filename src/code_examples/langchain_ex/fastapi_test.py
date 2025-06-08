from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import uuid
from enum import Enum
import json
import re
import math
import os
import logging

# Install required packages
try:
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate
except ImportError:
    print("Installing required packages...")
    import subprocess
    subprocess.run(["pip", "install", "google-generativeai", "langchain-google-genai", "langchain-core"])
    import google.generativeai as genai
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
    from langchain_core.prompts import ChatPromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic Models for API
class ChatMessage(BaseModel):
    message: str = Field(..., description="The user's message")
    thread_id: Optional[str] = Field(None, description="Optional thread ID for conversation continuity")

class ChatResponse(BaseModel):
    response: str = Field(..., description="The agent's response")
    thread_id: str = Field(..., description="Thread ID for this conversation")
    context_info: Dict[str, Any] = Field(..., description="Analysis and context information")

class MemoryType(str, Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"

class ClearMemoryRequest(BaseModel):
    memory_type: Optional[MemoryType] = Field(None, description="Specific memory type to clear, or None for all")

class AgentStats(BaseModel):
    short_term_messages: int
    long_term_facts: int
    working_memory_items: int
    episodic_memories: int
    total_tool_calls: int
    current_state: str
    model: str
    conversation_id: str

# Tool Results (same as original)
class ToolResult:
    def __init__(self, success: bool, result: Any, error: str = None):
        self.success = success
        self.result = result
        self.error = error

# Agent Tools powered by Gemini (same as original)
class GeminiTools:
    """Collection of tools that use Gemini for enhanced capabilities"""
    
    def __init__(self, model):
        self.model = model
    
    def calculator(self, expression: str) -> ToolResult:
        """Perform mathematical calculations with Gemini verification"""
        try:
            # Safety check first
            allowed_chars = set('0123456789+-*/.() ')
            if not all(c in allowed_chars for c in expression):
                return ToolResult(False, None, "Invalid characters in expression")
            
            # Calculate result
            result = eval(expression)
            
            # Verify with Gemini for complex calculations
            if len(expression) > 20 or any(op in expression for op in ['**', '//', '%']):
                prompt = f"Verify this calculation: {expression} = {result}. Is this correct? Just answer 'Yes' or 'No' and provide the correct answer if wrong."
                verification = self.model.invoke([HumanMessage(content=prompt)])
                
                if "no" in verification.content.lower():
                    return ToolResult(False, None, f"Calculation verification failed: {verification.content}")
            
            return ToolResult(True, result)
        except Exception as e:
            return ToolResult(False, None, f"Calculation error: {str(e)}")
    
    def get_weather(self, location: str) -> ToolResult:
        """Get weather information using Gemini's knowledge"""
        try:
            prompt = f"""
            Please provide current weather information for {location}. 
            If you don't have real-time data, provide:
            1. Typical weather patterns for this location at this time of year
            2. General climate information
            3. Clearly state that this is general information, not real-time data
            
            Format your response as a brief, informative weather summary.
            """
            
            response = self.model.invoke([HumanMessage(content=prompt)])
            return ToolResult(True, response.content)
            
        except Exception as e:
            return ToolResult(False, None, f"Weather lookup error: {str(e)}")
    
    def set_reminder(self, task: str, time_str: str) -> ToolResult:
        """Set a reminder with Gemini's help to parse time"""
        try:
            reminder_id = str(uuid.uuid4())[:8]
            
            # Use Gemini to parse and validate the time
            time_prompt = f"""
            Parse this time/date reference: "{time_str}"
            Current time is: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            Please respond with:
            1. A standardized time format
            2. Whether this is a reasonable time for the reminder
            
            Keep your response brief and clear.
            """
            
            time_response = self.model.invoke([HumanMessage(content=time_prompt)])
            
            return ToolResult(True, {
                "id": reminder_id,
                "task": task,
                "time": time_str,
                "parsed_time": time_response.content,
                "status": "set"
            })
        except Exception as e:
            return ToolResult(False, None, f"Failed to set reminder: {str(e)}")
    
    def search_knowledge(self, query: str) -> ToolResult:
        """Search Gemini's knowledge base"""
        try:
            search_prompt = f"""
            Please provide informative and accurate information about: {query}
            
            Structure your response with:
            1. Key facts and definitions
            2. Important details or context
            3. Any relevant examples or applications
            
            Keep it concise but comprehensive.
            """
            
            response = self.model.invoke([HumanMessage(content=search_prompt)])
            return ToolResult(True, response.content)
            
        except Exception as e:
            return ToolResult(False, None, f"Knowledge search error: {str(e)}")
    
    def analyze_sentiment(self, text: str) -> ToolResult:
        """Analyze sentiment and emotion in text"""
        try:
            sentiment_prompt = f"""
            Analyze the sentiment and emotional tone of this text: "{text}"
            
            Provide:
            1. Overall sentiment (positive/negative/neutral)
            2. Confidence level (high/medium/low)
            3. Key emotional indicators
            4. Brief explanation
            
            Keep it concise and professional.
            """
            
            response = self.model.invoke([HumanMessage(content=sentiment_prompt)])
            return ToolResult(True, response.content)
            
        except Exception as e:
            return ToolResult(False, None, f"Sentiment analysis error: {str(e)}")

# Memory Management (same as original)
@dataclass
class Memory:
    short_term: List[Dict] = field(default_factory=list)  # Recent conversation
    long_term: Dict[str, Any] = field(default_factory=dict)  # Persistent facts
    working: Dict[str, Any] = field(default_factory=dict)  # Current task context
    episodic: List[Dict] = field(default_factory=list)  # Past experiences/episodes

# Agent State Management
@dataclass
class AgentState:
    current_task: Optional[str] = None
    task_progress: Dict[str, Any] = field(default_factory=dict)
    active_tools: List[str] = field(default_factory=list)
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

# Main Gemini-Powered Agent Class (with all original methods)
class GeminiAgent:
    def __init__(self):
        self.memory = Memory()
        self.state = AgentState()
        # Initialize model only if API key is available
        self.model = None
        self.tools = None
        self.conversation_id = str(uuid.uuid4())
        
        # System prompt for the agent
        self.system_prompt = """
        You are an intelligent AI agent with access to various tools and persistent memory.
        
        Your capabilities include:
        - Performing calculations
        - Providing weather information
        - Setting reminders
        - Searching your knowledge base
        - Analyzing sentiment
        - Remembering user information and conversation history
        
        Always be helpful, accurate, and considerate. Use tools when appropriate, and maintain context from previous conversations.
        
        When responding:
        1. Consider the conversation history and user's previous interactions
        2. Use tools if the request requires specific functionality
        3. Provide clear, informative responses
        4. Remember important user information for future reference
        """
    
    def initialize_model(self, api_key: str):
        """Initialize the Gemini model with API key"""
        try:
            genai.configure(api_key=api_key)
            self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            self.tools = GeminiTools(self.model)
            logger.info("Gemini model initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Gemini model: {e}")
            return False
    
    def process_message(self, message: str, thread_id: str) -> Tuple[str, Dict]:
        """Main message processing function using Gemini"""
        
        if not self.model:
            return "Error: Gemini model not initialized. Please set GOOGLE_API_KEY.", {}
        
        # Update short-term memory
        self.memory.short_term.append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat(),
            "thread_id": thread_id
        })
        
        # Analyze message intent using Gemini
        intent, entities, should_use_tools = self._analyze_message_with_gemini(message)
        
        # Update working memory
        self.memory.working.update({
            "current_intent": intent,
            "entities": entities,
            "thread_id": thread_id
        })
        
        # Process with tools if needed, otherwise use Gemini directly
        if should_use_tools:
            response, tool_calls = self._process_with_tools(intent, entities, message)
        else:
            response, tool_calls = self._process_with_gemini(message)
        
        # Update memories
        self._update_memories(message, response, intent, entities, tool_calls)
        
        # Store response in short-term memory
        self.memory.short_term.append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now().isoformat(),
            "thread_id": thread_id,
            "tool_calls": tool_calls
        })
        
        # Prepare context info
        context_info = {
            "intent": intent,
            "entities": entities,
            "tool_calls": tool_calls,
            "should_use_tools": should_use_tools,
            "memory_stats": self._get_memory_stats(),
            "agent_state": asdict(self.state)
        }
        
        return response, context_info
    
    # All the original private methods remain the same
    def _analyze_message_with_gemini(self, message: str) -> Tuple[str, Dict, bool]:
        """Use Gemini to analyze message intent and determine tool usage"""
        analysis_prompt = f"""
        Analyze this user message: "{message}"
        
        Determine:
        1. Intent category (calculation, weather, reminder, search, personal_info, question, conversation)
        2. Key entities (names, numbers, locations, dates, etc.)
        3. Whether tools should be used (yes/no)
        
        Respond in JSON format:
        {{
            "intent": "category",
            "entities": {{"key": "value"}},
            "should_use_tools": true/false,
            "reasoning": "brief explanation"
        }}
        """
        
        try:
            analysis_response = self.model.invoke([HumanMessage(content=analysis_prompt)])
            analysis_data = json.loads(analysis_response.content)
            
            return (
                analysis_data.get("intent", "conversation"),
                analysis_data.get("entities", {}),
                analysis_data.get("should_use_tools", False)
            )
        except:
            return self._simple_intent_analysis(message)
    
    def _simple_intent_analysis(self, message: str) -> Tuple[str, Dict, bool]:
        """Fallback simple intent analysis"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ["calculate", "math", "+", "-", "*", "/"]):
            return "calculation", {"expression": message}, True
        elif any(word in message_lower for word in ["weather", "temperature"]):
            return "weather", {"location": message}, True
        elif any(word in message_lower for word in ["remind", "reminder"]):
            return "reminder", {"task": message}, True
        elif any(word in message_lower for word in ["search", "find", "what is"]):
            return "search", {"query": message}, True
        else:
            return "conversation", {}, False
    
    def _process_with_tools(self, intent: str, entities: Dict, message: str) -> Tuple[str, List[Dict]]:
        """Process message using appropriate tools"""
        tool_calls = []
        
        if intent == "calculation":
            expression = self._extract_math_expression(message)
            if expression:
                result = self.tools.calculator(expression)
                tool_calls.append({"tool": "calculator", "input": expression, "result": result})
                if result.success:
                    response = f"The calculation result is: {result.result}"
                else:
                    response = f"I couldn't perform that calculation: {result.error}"
            else:
                response = "I can help with calculations, but I need a clear mathematical expression."
                
        elif intent == "weather":
            location = self._extract_location(message)
            result = self.tools.get_weather(location)
            tool_calls.append({"tool": "weather", "input": location, "result": result})
            if result.success:
                response = result.result
            else:
                response = f"I couldn't get weather information: {result.error}"
                
        elif intent == "reminder":
            task, time_str = self._extract_reminder_info(message)
            result = self.tools.set_reminder(task, time_str)
            tool_calls.append({"tool": "reminder", "input": {"task": task, "time": time_str}, "result": result})
            if result.success:
                reminder = result.result
                response = f"I've set a reminder (ID: {reminder['id']}) for: {reminder['task']}. {reminder['parsed_time']}"
            else:
                response = f"I couldn't set the reminder: {result.error}"
                
        elif intent == "search":
            query = self._extract_search_query(message)
            result = self.tools.search_knowledge(query)
            tool_calls.append({"tool": "search", "input": query, "result": result})
            if result.success:
                response = result.result
            else:
                response = f"I couldn't search for that: {result.error}"
        
        else:
            response, tool_calls = self._process_with_gemini(message)
            
        return response, tool_calls
    
    def _process_with_gemini(self, message: str) -> Tuple[str, List[Dict]]:
        """Process message directly with Gemini using conversation context"""
        
        context_messages = []
        context_messages.append(SystemMessage(content=self.system_prompt))
        
        if self.memory.long_term:
            memory_context = f"User information I remember: {json.dumps(self.memory.long_term, indent=2)}"
            context_messages.append(SystemMessage(content=memory_context))
        
        for msg in self.memory.short_term[-10:]:
            if msg["role"] == "user":
                context_messages.append(HumanMessage(content=msg["content"]))
            else:
                context_messages.append(AIMessage(content=msg["content"]))
        
        context_messages.append(HumanMessage(content=message))
        
        try:
            response = self.model.invoke(context_messages)
            self._extract_personal_info(message)
            return response.content, []
            
        except Exception as e:
            return f"I apologize, but I'm having trouble processing your request: {str(e)}", []
    
    def _extract_math_expression(self, message: str) -> str:
        """Extract mathematical expression from message"""
        math_patterns = [
            r'[\d+\-*/().\s]+',
            r'\d+\s*[\+\-\*/]\s*\d+',
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, message)
            if matches:
                return max(matches, key=len).strip()
        
        return message
    
    def _extract_location(self, message: str) -> str:
        """Extract location from message"""
        location_patterns = [
            r"in ([a-zA-Z\s]+)",
            r"for ([a-zA-Z\s]+)",
            r"at ([a-zA-Z\s]+)",
            r"weather ([a-zA-Z\s]+)"
        ]
        
        for pattern in location_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return "your location"
    
    def _extract_reminder_info(self, message: str) -> Tuple[str, str]:
        """Extract task and time from reminder message"""
        time_patterns = [r"at (.*)", r"in (.*)", r"on (.*)"]
        
        time_str = "later"
        for pattern in time_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                time_str = match.group(1).strip()
                break
        
        return message, time_str
    
    def _extract_search_query(self, message: str) -> str:
        """Extract search query from message"""
        query_patterns = [
            r"search (?:for )?(.+)",
            r"what is (.+)",
            r"find (.+)",
            r"look up (.+)",
            r"tell me about (.+)"
        ]
        
        for pattern in query_patterns:
            match = re.search(pattern, message, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return message
    
    def _extract_personal_info(self, message: str):
        """Extract and store personal information"""
        message_lower = message.lower()
        
        name_patterns = [
            r"my name is (\w+)",
            r"i am (\w+)",
            r"i'm (\w+)",
            r"call me (\w+)"
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, message_lower)
            if match:
                name = match.group(1).strip()
                self.memory.long_term["user_name"] = name
                self.memory.long_term["name_learned_at"] = datetime.now().isoformat()
                break
    
    def _update_memories(self, user_message: str, ai_response: str, intent: str, entities: Dict, tool_calls: List[Dict]):
        """Update different types of memory"""
        
        if intent in ["calculation", "weather", "reminder", "search", "personal_info"] or tool_calls:
            episode = {
                "timestamp": datetime.now().isoformat(),
                "intent": intent,
                "entities": entities,
                "user_message": user_message,
                "ai_response": ai_response,
                "tool_calls": tool_calls,
                "success": all(call.get("result", {}).get("success", False) for call in tool_calls) if tool_calls else True
            }
            self.memory.episodic.append(episode)
        
        if len(self.memory.short_term) > 50:
            self.memory.short_term = self.memory.short_term[-50:]
        
        if len(self.memory.episodic) > 100:
            self.memory.episodic = self.memory.episodic[-100:]
    
    def _get_memory_stats(self) -> Dict:
        """Get current memory statistics"""
        return {
            "short_term_messages": len(self.memory.short_term),
            "long_term_facts": len(self.memory.long_term),
            "working_memory_items": len(self.memory.working),
            "episodic_memories": len(self.memory.episodic),
            "total_tool_calls": sum(len(episode.get("tool_calls", [])) for episode in self.memory.episodic)
        }
    
    def get_memory_dump(self) -> Dict:
        """Get complete memory dump for debugging/display"""
        return {
            "short_term": self.memory.short_term[-10:],
            "long_term": self.memory.long_term,
            "working": self.memory.working,
            "episodic": self.memory.episodic[-5:],
            "stats": self._get_memory_stats()
        }
    
    def clear_memory(self, memory_type: MemoryType = None):
        """Clear specific or all memory types"""
        if memory_type == MemoryType.SHORT_TERM or memory_type is None:
            self.memory.short_term = []
        if memory_type == MemoryType.LONG_TERM or memory_type is None:
            self.memory.long_term = {}
        if memory_type == MemoryType.WORKING or memory_type is None:
            self.memory.working = {}
        if memory_type == MemoryType.EPISODIC or memory_type is None:
            self.memory.episodic = []

# Initialize FastAPI app
app = FastAPI(
    title="Gemini Agent API",
    description="AI Agent powered by Google's Gemini with advanced memory management",
    version="1.0.0"
)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the agent
agent = GeminiAgent()

# Initialize model on startup if API key is available
@app.on_event("startup")
async def startup_event():
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        success = agent.initialize_model(api_key)
        if success:
            logger.info("Gemini Agent initialized successfully")
        else:
            logger.warning("Failed to initialize Gemini Agent")
    else:
        logger.warning("GOOGLE_API_KEY not found in environment variables")

# API Routes
@app.post("/chat", response_model=ChatResponse)
async def chat_with_agent(chat_request: ChatMessage):
    """Chat with the Gemini-powered agent"""
    try:
        thread_id = chat_request.thread_id or f"thread_{str(uuid.uuid4())[:8]}"
        response, context_info = agent.process_message(chat_request.message, thread_id)
        
        return ChatResponse(
            response=response,
            thread_id=thread_id,
            context_info=context_info
        )
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats", response_model=AgentStats)
async def get_agent_stats():
    """Get current agent statistics"""
    try:
        stats = agent._get_memory_stats()
        return AgentStats(
            short_term_messages=stats["short_term_messages"],
            long_term_facts=stats["long_term_facts"],
            working_memory_items=stats["working_memory_items"],
            episodic_memories=stats["episodic_memories"],
            total_tool_calls=stats["total_tool_calls"],
            current_state=agent.state.current_task or "Ready",
            model="Gemini 1.5 Flash",
            conversation_id=agent.conversation_id
        )
    except Exception as e:
        logger.error(f"Error getting agent stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory")
async def get_memory_dump():
    """Get complete memory dump"""
    try:
        return agent.get_memory_dump()
    except Exception as e:
        logger.error(f"Error getting memory dump: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/memory/clear")
async def clear_memory(clear_request: ClearMemoryRequest):
    """Clear specific or all memory types"""
    try:
        agent.clear_memory(clear_request.memory_type)
        memory_type_str = clear_request.memory_type.value if clear_request.memory_type else "all"
        return {"message": f"Cleared {memory_type_str} memory"}
    except Exception as e:
        logger.error(f"Error clearing memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_initialized": agent.model is not None,
        "timestamp": datetime.now().isoformat()
    }

# Simple web interface
@app.get("/", response_class=HTMLResponse)
async def web_interface():
    """Simple web interface for testing the API"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Gemini Agent API</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .chat-container { border: 1px solid #ddd; height: 400px; overflow-y: auto; padding: 10px; margin: 10px 0; }
            .message { margin: 10px 0; padding: 10px; border-radius: 5px; }
            .user { background-color: #e3f2fd; text-align: right; }
            .assistant { background-color: #f3e5f5; }
            input, button { padding: 10px; margin: 5px; }
            input[type="text"] { width: 60%; }
            .stats { background-color: #f5f5f5; padding: 10px; margin: 10px 0; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>🤖 Gemini AI Agent</h1>
        <div class="stats" id="stats">Loading stats...</div>
        
        <div class="chat-container" id="chatContainer"></div>
        
        <div>
            <input type="text" id="messageInput" placeholder="Type your message..." onkeypress="handleKeyPress(event)">
            <button onclick="sendMessage()">Send</button>
            <button onclick="clearMemory()">Clear Memory</button>
            <button onclick="refreshStats()">Refresh Stats</button>
        </div>

        <script>
            let threadId = null;
            
            async function sendMessage() {
                const input = document.getElementById('messageInput');
                const message = input.value.trim();
                if (!message) return;
                
                addMessage('user', message);
                input.value = '';
                
                try {
                    const response = await fetch('/chat', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ message, thread_id: threadId })
                    });
                    
                    const data = await response.json();
                    threadId = data.thread_id;
                    addMessage('assistant', data.response);
                    refreshStats();
                } catch (error) {
                    addMessage('assistant', 'Error: ' + error.message);
                }
            }
            
            function addMessage(role, content) {
                const container = document.getElementById('chatContainer');
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${role}`;
                messageDiv.textContent = content;
                container.appendChild(messageDiv);
                container.scrollTop = container.scrollHeight;
            }
            
            async function clearMemory() {
                try {
                    await fetch('/memory/clear', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
                    document.getElementById('chatContainer').innerHTML = '';
                    threadId = null;
                    refreshStats();
                } catch (error) {
                    console.error('Error clearing memory:', error);
                }
            }
            
            async function refreshStats() {
                try {
                    const response = await fetch('/stats');
                    const stats = await response.json();
                    document.getElementById('stats').innerHTML = `
                        <strong>Agent Stats:</strong> 
                        Short-term: ${stats.short_term_messages} messages | 
                        Long-term: ${stats.long_term_facts} facts | 
                        Episodes: ${stats.episodic_memories} | 
                        Tool calls: ${stats.total_tool_calls}
                    `;
                } catch (error) {
                    document.getElementById('stats').textContent = 'Error loading stats';
                }
            }
            
            function handleKeyPress(event) {
                if (event.key === 'Enter') {
                    sendMessage();
                }
            }
            
            // Load stats on page load
            refreshStats();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)