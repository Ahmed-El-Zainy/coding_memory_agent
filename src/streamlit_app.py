import streamlit as st
import requests
import json
from datetime import datetime
import uuid
import os

# Page configuration
st.set_page_config(
    page_title="AI Chatbot with Memory",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# Custom CSS
st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.8rem;
        margin: 0.5rem 0;
        max-width: 70%;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        margin-left: auto;
        text-align: right;
    }
    .assistant-message {
        background-color: #f1f3f4;
        color: #333;
        border-left: 4px solid #007bff;
    }
    .memory-indicator {
        font-size: 0.8rem;
        color: #666;
        font-style: italic;
        margin-top: 0.5rem;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'user_id' not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())[:8]

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'conversation_count' not in st.session_state:
    st.session_state.conversation_count = 0

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
        st.error(f"Error communicating with backend: {e}")
        return None

def get_user_memories(user_id: str):
    """Get all memories for current user"""
    try:
        response = requests.get(f"{BACKEND_URL}/memories/{user_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error retrieving memories: {e}")
        return None

def clear_user_memories(user_id: str):
    """Clear all memories for current user"""
    try:
        response = requests.delete(f"{BACKEND_URL}/memories/{user_id}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error clearing memories: {e}")
        return None

def get_model_info():
    """Get model information from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/models")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting model info: {e}")
        return None
    """Check if backend is healthy"""
    try:
        response = requests.get(f"{BACKEND_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

# Main app
def main():
    st.title("🤖 AI Chatbot with Memory")
    st.markdown("*Powered by Google Gemini Flash & Hugging Face Embeddings*")
    
    # Sidebar
    with st.sidebar:
        st.header("🎛️ Controls")
        
        # Backend status
        is_healthy, health_data = check_backend_health()
        if is_healthy:
            st.success("✅ Backend Connected")
            if health_data and 'models' in health_data:
                with st.expander("🔧 Model Info"):
                    st.json(health_data['models'])
        else:
            st.error("❌ Backend Disconnected")
            st.warning("Make sure the backend server is running!")
        
        # User ID display
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("👤 Session Info")
        st.text(f"User ID: {st.session_state.user_id}")
        st.text(f"Messages: {st.session_state.conversation_count}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Memory management
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.subheader("🧠 Memory Management")
        
        if st.button("🔍 View My Memories", use_container_width=True):
            memories_data = get_user_memories(st.session_state.user_id)
            if memories_data:
                st.session_state.show_memories = True
        
        if st.button("🗑️ Clear My Memories", use_container_width=True, type="secondary"):
            if st.session_state.get('confirm_clear', False):
                result = clear_user_memories(st.session_state.user_id)
                if result:
                    st.success(f"Cleared {result.get('message', 'memories')}")
                    st.session_state.confirm_clear = False
                    st.rerun()
            else:
                st.session_state.confirm_clear = True
                st.warning("Click again to confirm!")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # New session
        if st.button("🆕 New Session", use_container_width=True):
            st.session_state.user_id = str(uuid.uuid4())[:8]
            st.session_state.chat_history = []
            st.session_state.conversation_count = 0
            st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Chat history display
        st.subheader("💬 Conversation")
        
        chat_container = st.container()
        with chat_container:
            for i, chat in enumerate(st.session_state.chat_history):
                # User message
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>You:</strong> {chat['user_message']}
                </div>
                """, unsafe_allow_html=True)
                
                # Assistant message
                memories_info = ""
                model_info = ""
                
                if chat.get('memories_used'):
                    memories_info = f"""
                    <div class="memory-indicator">
                        🧠 Used {len(chat['memories_used'])} memories | Decision: {chat.get('memory_decision', 'N/A')}
                    </div>
                    """
                
                if chat.get('model_info'):
                    model_info = f"""
                    <div class="memory-indicator">
                        🤖 {chat['model_info'].get('llm_model', 'Unknown')} + {chat['model_info'].get('embedding_model', 'Unknown')}
                    </div>
                    """
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>Assistant:</strong> {chat['assistant_response']}
                    {memories_info}
                    {model_info}
                </div>
                """, unsafe_allow_html=True)
        
        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            col_input, col_send = st.columns([4, 1])
            with col_input:
                user_input = st.text_input(
                    "Message",
                    placeholder="Type your message here...",
                    label_visibility="collapsed"
                )
            with col_send:
                send_button = st.form_submit_button("Send 📤", use_container_width=True)
        
        if send_button and user_input.strip():
            with st.spinner("🤔 Thinking..."):
                response_data = send_message(user_input.strip(), st.session_state.user_id)
                
                if response_data:
                    # Add to chat history
                    chat_entry = {
                        'user_message': user_input.strip(),
                        'assistant_response': response_data.get('response', 'No response'),
                        'memories_used': response_data.get('memories_used', []),
                        'memory_decision': response_data.get('memory_retrieval_decision', ''),
                        'model_info': response_data.get('model_info', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.chat_history.append(chat_entry)
                    st.session_state.conversation_count += 1
                    st.rerun()
    
    with col2:
        # Memory display
        if st.session_state.get('show_memories', False):
            st.subheader("🧠 Your Memories")
            
            memories_data = get_user_memories(st.session_state.user_id)
            if memories_data and memories_data.get('memories'):
                st.text(f"Total memories: {memories_data.get('total', 0)}")
                
                for memory in memories_data['memories'][:5]:  # Show first 5
                    with st.expander(f"Memory {memory['id'][:8]}..."):
                        st.text(f"Time: {memory['timestamp']}")
                        st.text_area(
                            "Content:",
                            memory['content'],
                            height=100,
                            disabled=True,
                            key=f"memory_{memory['id']}"
                        )
            else:
                st.info("No memories found for this session.")
            
            if st.button("❌ Close Memories"):
                st.session_state.show_memories = False
                st.rerun()
    
    # Example prompts
    if not st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💡 Try these examples:")
        
        example_col1, example_col2, example_col3 = st.columns(3)
        
        with example_col1:
            if st.button("👋 Hello, I'm Alex and I love pizza"):
                st.session_state.example_input = "Hello, I'm Alex and I love pizza"
                st.rerun()
        
        with example_col2:
            if st.button("🍕 What's my favorite food?"):
                st.session_state.example_input = "What's my favorite food?"
                st.rerun()
        
        with example_col3:
            if st.button("📚 Remember our last conversation"):
                st.session_state.example_input = "Can you remember what we talked about before?"
                st.rerun()
        
        # Handle example input
        if hasattr(st.session_state, 'example_input'):
            with st.spinner("🤔 Thinking..."):
                response_data = send_message(st.session_state.example_input, st.session_state.user_id)
                
                if response_data:
                    chat_entry = {
                        'user_message': st.session_state.example_input,
                        'assistant_response': response_data.get('response', 'No response'),
                        'memories_used': response_data.get('memories_used', []),
                        'memory_decision': response_data.get('memory_retrieval_decision', ''),
                        'model_info': response_data.get('model_info', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    st.session_state.chat_history.append(chat_entry)
                    st.session_state.conversation_count += 1
                    delattr(st.session_state, 'example_input')
                    st.rerun()

if __name__ == "__main__":
    main()