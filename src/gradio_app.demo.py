import gradio as gr
import requests
import json
from datetime import datetime
import uuid
import os

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Global state management
class ChatState:
    def __init__(self):
        self.user_id = str(uuid.uuid4())[:8]
        self.conversation_count = 0
        self.memories_display = ""
    
    def new_session(self):
        self.user_id = str(uuid.uuid4())[:8]
        self.conversation_count = 0
        self.memories_display = ""

# Initialize global state
chat_state = ChatState()

# Helper functions
def send_message(message: str, user_id: str):
    """Send message to backend API"""
    try:
        response = requests.post(
            f"{BACKEND_URL}/chat",
            json={"message": message, "user_id": user_id},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error communicating with backend: {e}"}

def get_user_memories(user_id: str):
    """Get all memories for current user"""
    try:
        response = requests.get(f"{BACKEND_URL}/memories/{user_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error retrieving memories: {e}"}

def clear_user_memories(user_id: str):
    """Clear all memories for current user"""
    try:
        response = requests.delete(f"{BACKEND_URL}/memories/{user_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Error clearing memories: {e}"}

def check_backend_health():
    """Check if backend is healthy"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, None
    except:
        return False, None

def format_chat_message(role, message, metadata=None):
    """Format chat message with metadata"""
    formatted_msg = f"**{role}:** {message}"
    
    if metadata and role == "Assistant":
        if metadata.get('memories_used'):
            formatted_msg += f"\n\n🧠 *Used {len(metadata['memories_used'])} memories | Decision: {metadata.get('memory_decision', 'N/A')}*"
        
        if metadata.get('model_info'):
            model_info = metadata['model_info']
            formatted_msg += f"\n🤖 *{model_info.get('llm_model', 'Unknown')} + {model_info.get('embedding_model', 'Unknown')}*"
    
    return formatted_msg

def chat_function(message, history):
    """Main chat function"""
    if not message.strip():
        return history, ""
    
    # Send message to backend
    response_data = send_message(message.strip(), chat_state.user_id)
    
    if "error" in response_data:
        # Add error message to history
        history.append([message, f"❌ {response_data['error']}"])
        return history, ""
    
    # Format assistant response with metadata
    assistant_response = response_data.get('response', 'No response')
    metadata = {
        'memories_used': response_data.get('memories_used', []),
        'memory_decision': response_data.get('memory_retrieval_decision', ''),
        'model_info': response_data.get('model_info', {})
    }
    
    # Add memory and model info to response if available
    if metadata['memories_used']:
        assistant_response += f"\n\n🧠 *Used {len(metadata['memories_used'])} memories | Decision: {metadata['memory_decision']}*"
    
    if metadata['model_info']:
        model_info = metadata['model_info']
        assistant_response += f"\n🤖 *{model_info.get('llm_model', 'Unknown')} + {model_info.get('embedding_model', 'Unknown')}*"
    
    # Update conversation count
    chat_state.conversation_count += 1
    
    # Add to history
    history.append([message, assistant_response])
    
    return history, ""

def view_memories():
    """View user memories"""
    memories_data = get_user_memories(chat_state.user_id)
    
    if "error" in memories_data:
        return f"❌ {memories_data['error']}"
    
    if not memories_data.get('memories'):
        return "ℹ️ No memories found for this session."
    
    memories_text = f"**🧠 Your Memories (Total: {memories_data.get('total', 0)})**\n\n"
    
    for i, memory in enumerate(memories_data['memories'][:10], 1):  # Show first 10
        memories_text += f"**Memory {i} ({memory['id'][:8]}...):**\n"
        memories_text += f"*Time: {memory['timestamp']}*\n"
        memories_text += f"{memory['content']}\n\n---\n\n"
    
    return memories_text

def clear_memories():
    """Clear user memories"""
    result = clear_user_memories(chat_state.user_id)
    
    if "error" in result:
        return f"❌ {result['error']}"
    
    return f"✅ {result.get('message', 'Memories cleared successfully!')}"

def new_session():
    """Start a new session"""
    chat_state.new_session()
    return [], f"🆕 **New session started!**\n**User ID:** {chat_state.user_id}\n**Messages:** 0", ""

def get_session_info():
    """Get current session information"""
    is_healthy, health_data = check_backend_health()
    
    status = "✅ Backend Connected" if is_healthy else "❌ Backend Disconnected"
    
    info = f"**Session Information**\n\n"
    info += f"**Status:** {status}\n"
    info += f"**User ID:** {chat_state.user_id}\n"
    info += f"**Messages:** {chat_state.conversation_count}\n\n"
    
    if is_healthy and health_data and 'models' in health_data:
        info += f"**Model Info:**\n"
        info += f"```json\n{json.dumps(health_data['models'], indent=2)}\n```"
    elif not is_healthy:
        info += "⚠️ *Make sure the backend server is running!*"
    
    return info

def example_chat(example_text, history):
    """Handle example button clicks"""
    return chat_function(example_text, history)

# Custom CSS for better styling
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

.chat-message {
    padding: 10px;
    margin: 5px 0;
    border-radius: 10px;
}

.memory-info {
    background-color: #f0f8ff;
    padding: 10px;
    border-radius: 5px;
    border-left: 4px solid #007bff;
    margin: 10px 0;
}

.status-connected {
    color: #28a745;
}

.status-disconnected {
    color: #dc3545;
}
"""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="AI Chatbot with Memory", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 AI Chatbot with Memory
    *Powered by Google Gemini Flash & Hugging Face Embeddings*
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            # Main chat interface
            chatbot = gr.Chatbot(
                label="💬 Conversation",
                height=500,
                show_label=True,
                avatar_images=("👤", "🤖")
            )
            
            msg = gr.Textbox(
                label="Message",
                placeholder="Type your message here...",
                lines=1,
                max_lines=3
            )
            
            with gr.Row():
                send_btn = gr.Button("Send 📤", variant="primary")
                clear_btn = gr.Button("Clear Chat 🗑️", variant="secondary")
            
            # Example buttons
            gr.Markdown("### 💡 Try these examples:")
            with gr.Row():
                example1_btn = gr.Button("👋 Hello, I'm Alex and I love pizza")
                example2_btn = gr.Button("🍕 What's my favorite food?")
                example3_btn = gr.Button("📚 Remember our last conversation")
        
        with gr.Column(scale=1):
            # Control panel
            gr.Markdown("## 🎛️ Controls")
            
            # Session info
            session_info = gr.Markdown(get_session_info())
            refresh_info_btn = gr.Button("🔄 Refresh Info", size="sm")
            
            gr.Markdown("---")
            
            # Memory management
            gr.Markdown("### 🧠 Memory Management")
            
            memories_display = gr.Markdown("Click 'View Memories' to see your stored memories.")
            
            with gr.Row():
                view_memories_btn = gr.Button("🔍 View Memories", size="sm")
                clear_memories_btn = gr.Button("🗑️ Clear Memories", size="sm", variant="secondary")
            
            memory_result = gr.Markdown("")
            
            gr.Markdown("---")
            
            # Session management
            new_session_btn = gr.Button("🆕 New Session", variant="stop")
    
    # Event handlers
    def clear_chat():
        return [], ""
    
    # Chat functionality
    msg.submit(chat_function, [msg, chatbot], [chatbot, msg])
    send_btn.click(chat_function, [msg, chatbot], [chatbot, msg])
    clear_btn.click(clear_chat, outputs=[chatbot, msg])
    
    # Example buttons
    example1_btn.click(lambda h: chat_function("Hello, I'm Alex and I love pizza", h), [chatbot], [chatbot, msg])
    example2_btn.click(lambda h: chat_function("What's my favorite food?", h), [chatbot], [chatbot, msg])
    example3_btn.click(lambda h: chat_function("Can you remember what we talked about before?", h), [chatbot], [chatbot, msg])
    
    # Memory management
    view_memories_btn.click(view_memories, outputs=[memories_display])
    clear_memories_btn.click(clear_memories, outputs=[memory_result])
    
    # Session management
    new_session_btn.click(new_session, outputs=[chatbot, session_info, msg])
    refresh_info_btn.click(get_session_info, outputs=[session_info])

# Launch configuration
if __name__ == "__main__":
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        show_api=False
    )