import gradio as gr
import json
import re
import math
import os
import getpass
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
import uuid
from enum import Enum

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

# Setup Google API Key
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google API Key: ")

# Initialize Gemini model
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Tool Results
class ToolResult:
    def __init__(self, success: bool, result: Any, error: str = None):
        self.success = success
        self.result = result
        self.error = error

# Agent Tools powered by Gemini
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

# Memory Management (same as before)
@dataclass
class Memory:
    short_term: List[Dict] = field(default_factory=list)  # Recent conversation
    long_term: Dict[str, Any] = field(default_factory=dict)  # Persistent facts
    working: Dict[str, Any] = field(default_factory=dict)  # Current task context
    episodic: List[Dict] = field(default_factory=list)  # Past experiences/episodes

class MemoryType(Enum):
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
    WORKING = "working"
    EPISODIC = "episodic"

# Agent State Management
@dataclass
class AgentState:
    current_task: Optional[str] = None
    task_progress: Dict[str, Any] = field(default_factory=dict)
    active_tools: List[str] = field(default_factory=list)
    conversation_context: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

# Main Gemini-Powered Agent Class
class GeminiAgent:
    def __init__(self):
        self.memory = Memory()
        self.state = AgentState()
        self.tools = GeminiTools(model)
        self.model = model
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
    
    def process_message(self, message: str, thread_id: str) -> Tuple[str, Dict]:
        """Main message processing function using Gemini"""
        
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
            
            # Parse JSON response
            import json
            analysis_data = json.loads(analysis_response.content)
            
            return (
                analysis_data.get("intent", "conversation"),
                analysis_data.get("entities", {}),
                analysis_data.get("should_use_tools", False)
            )
        except:
            # Fallback to simple analysis
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
            # Extract mathematical expression
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
            # Fallback to Gemini
            response, tool_calls = self._process_with_gemini(message)
            
        return response, tool_calls
    
    def _process_with_gemini(self, message: str) -> Tuple[str, List[Dict]]:
        """Process message directly with Gemini using conversation context"""
        
        # Build context from memory
        context_messages = []
        
        # Add system message
        context_messages.append(SystemMessage(content=self.system_prompt))
        
        # Add relevant long-term memory
        if self.memory.long_term:
            memory_context = f"User information I remember: {json.dumps(self.memory.long_term, indent=2)}"
            context_messages.append(SystemMessage(content=memory_context))
        
        # Add recent conversation history
        for msg in self.memory.short_term[-10:]:  # Last 10 messages
            if msg["role"] == "user":
                context_messages.append(HumanMessage(content=msg["content"]))
            else:
                context_messages.append(AIMessage(content=msg["content"]))
        
        # Add current message
        context_messages.append(HumanMessage(content=message))
        
        try:
            response = self.model.invoke(context_messages)
            
            # Check if this is personal information to store
            self._extract_personal_info(message)
            
            return response.content, []
            
        except Exception as e:
            return f"I apologize, but I'm having trouble processing your request: {str(e)}", []
    
    def _extract_math_expression(self, message: str) -> str:
        """Extract mathematical expression from message"""
        # Look for mathematical patterns
        math_patterns = [
            r'[\d+\-*/().\s]+',
            r'\d+\s*[\+\-\*/]\s*\d+',
        ]
        
        for pattern in math_patterns:
            matches = re.findall(pattern, message)
            if matches:
                return max(matches, key=len).strip()
        
        return message  # Return original if no clear pattern found
    
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
        # Simple extraction - can be enhanced
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
        
        # Extract name
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
        
        # Update episodic memory with significant interactions
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
        
        # Manage memory sizes
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

# Initialize the Gemini agent
agent = GeminiAgent()

# Gradio Interface Functions
def chat_with_agent(message, history, thread_id):
    """Main chat function powered by Gemini"""
    if not thread_id:
        thread_id = "thread_" + str(uuid.uuid4())[:8]
    
    try:
        response, context_info = agent.process_message(message, thread_id)
        history.append([message, response])
        return history, "", thread_id, json.dumps(context_info, indent=2)
    except Exception as e:
        error_response = f"I apologize, but I encountered an error: {str(e)}"
        history.append([message, error_response])
        return history, "", thread_id, json.dumps({"error": str(e)}, indent=2)

def clear_conversation():
    """Clear conversation and memory"""
    global agent
    agent.clear_memory()
    return [], "", "", "{}"

def clear_specific_memory(memory_type):
    """Clear specific memory type"""
    global agent
    memory_type_enum = MemoryType(memory_type)
    agent.clear_memory(memory_type_enum)
    return f"Cleared {memory_type} memory"

def get_memory_dump():
    """Get memory dump for display"""
    return json.dumps(agent.get_memory_dump(), indent=2)

def get_agent_stats():
    """Get agent statistics"""
    stats = agent._get_memory_stats()
    return f"""Gemini Agent Statistics:
    
Short-term Memory: {stats['short_term_messages']} messages
Long-term Memory: {stats['long_term_facts']} facts stored
Working Memory: {stats['working_memory_items']} active items
Episodic Memory: {stats['episodic_memories']} episodes
Total Tool Calls: {stats['total_tool_calls']}

Current State: {agent.state.current_task or 'Ready'}
Model: Gemini 1.5 Flash
Thread ID: {agent.conversation_id}
"""

# Create Gradio Interface
with gr.Blocks(title="Gemini Agent with Memory", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🤖 Gemini-Powered AI Agent with Memory")
    gr.Markdown("""
    This intelligent agent is powered by **Google's Gemini AI** and features advanced memory management.
    
    **🧠 Memory Types:**
    - **Short-term**: Recent conversation context
    - **Long-term**: Persistent user information and facts
    - **Working**: Current task and processing context
    - **Episodic**: Past experiences and interactions
    
    **🛠️ Enhanced Capabilities:**
    - 🧮 **Smart Calculator**: Math with Gemini verification
    - 🌤️ **Weather Assistant**: Climate info and patterns
    - ⏰ **Intelligent Reminders**: Natural language time parsing
    - 🔍 **Knowledge Search**: Access to Gemini's vast knowledge
    - 💭 **Sentiment Analysis**: Understand emotional context
    - 🧠 **Contextual Memory**: Remember and build on conversations
    """)
    
    with gr.Row():
        # Main chat interface
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(height=500, label="Gemini Agent Chat", show_copy_button=True)
            msg = gr.Textbox(
                label="Message", 
                placeholder="Try: 'Hi, I'm Sarah', 'Calculate 47 * 23', 'Weather in Paris', 'Remind me to exercise at 6pm'"
            )
            
            with gr.Row():
                clear_btn = gr.Button("Clear Conversation", variant="secondary")
                submit_btn = gr.Button("Send", variant="primary")
        
        # Agent state and memory panel
        with gr.Column(scale=1):
            thread_id = gr.Textbox(label="Thread ID", interactive=False)
            
            with gr.Tabs():
                with gr.TabItem("Context Info"):
                    context_info = gr.JSON(label="Last Response Analysis", show_indices=False)
                
                with gr.TabItem("Memory Dump"):
                    memory_dump = gr.JSON(label="Agent Memory State", show_indices=False)
                    refresh_memory_btn = gr.Button("Refresh Memory")
                
                with gr.TabItem("Agent Stats"):
                    agent_stats = gr.Textbox(label="Gemini Agent Statistics", lines=12, interactive=False)
                    refresh_stats_btn = gr.Button("Refresh Stats")
                
                with gr.TabItem("Memory Controls"):
                    memory_type_dropdown = gr.Dropdown(
                        choices=["short_term", "long_term", "working", "episodic"],
                        label="Memory Type to Clear",
                        value="short_term"
                    )
                    clear_memory_btn = gr.Button("Clear Selected Memory")
                    memory_status = gr.Textbox(label="Status", interactive=False)
    
    # Enhanced example prompts for Gemini
    with gr.Row():
        gr.Examples(
            examples=[
                ["Hi, my name is Alex and I'm a software developer"],
                ["Calculate the compound interest on $5000 at 3.5% for 10 years"],
                ["What's the weather typically like in Singapore in December?"],
                ["Set a reminder to review my budget every Friday at 2pm"],
                ["Tell me about machine learning and its applications"],
                ["Analyze the sentiment of this text: 'I'm really excited about this new project!'"],
                ["What's my name and what do I do for work?"],
                ["What did we just discuss about compound interest?"],
                ["How do you remember information across our conversations?"]
            ],
            inputs=msg,
            label="Try These Gemini-Powered Examples"
        )
    
    # Event handlers
    msg.submit(chat_with_agent, [msg, chatbot, thread_id], [chatbot, msg, thread_id, context_info])
    submit_btn.click(chat_with_agent, [msg, chatbot, thread_id], [chatbot, msg, thread_id, context_info])
    clear_btn.click(clear_conversation, outputs=[chatbot, msg, thread_id, context_info])
    
    refresh_memory_btn.click(get_memory_dump, outputs=[memory_dump])
    refresh_stats_btn.click(get_agent_stats, outputs=[agent_stats])
    clear_memory_btn.click(clear_specific_memory, [memory_type_dropdown], [memory_status])
    
    # Auto-refresh stats on startup
    demo.load(get_agent_stats, outputs=[agent_stats])

    # Add API key info
    with gr.Accordion("Setup Instructions", open=False):
        gr.Markdown("""
        ### 🔑 Google API Key Setup
        
        To use this Gemini-powered agent, you need a Google API key:
        
        1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
        2. Create a new API key
        3. Enter it when prompted, or set the `GOOGLE_API_KEY` environment variable
        
        ### 🚀 Features Powered by Gemini
        
        - **Intelligent Intent Recognition**: Gemini analyzes your messages to understand what you need
        - **Contextual Responses**: Uses conversation history for more relevant replies  
        - **Enhanced Tool Selection**: Decides when to use tools vs. direct AI responses
        - **Smart Memory Management**: Learns and remembers important information about you
        - **Natural Language Processing**: Understands complex requests and nuanced language
        
        ### 🎯 Advanced Examples to Try
        
        - **Complex Math**: "What's 15% of $2,847, and if I save that amount monthly at 4% annual interest, how much will I have in 5 years?"
        - **Weather + Memory**: "I live in Tokyo, what's the weather like?" (then later: "How's the weather where I live?")
        - **Contextual Reminders**: "Remind me to call my doctor next Tuesday at 2 PM about my appointment"
        - **Knowledge + Memory**: "I'm learning Python, tell me about list comprehensions" (it will remember you're learning Python)
        """)

if __name__ == "__main__":
    demo.launch()