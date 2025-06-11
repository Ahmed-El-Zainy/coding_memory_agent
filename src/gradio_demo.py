# import gradio as gr
# import requests
# import os
# import json
# import logging
# import argparse
# from typing import List, Dict, Any, Optional
# from datetime import datetime
# import yaml
# from pathlib import Path

# # # Configure logging
# # logging.basicConfig(level=logging.INFO)
# # logger = logging.getLogger(__name__)

# from logs.custom_logger import CustomLoggerTracker

# logging = CustomLoggerTracker()
# logger = logging.get_logger(__name__)



# class GradioInterface:
#     def __init__(self, backend_url: str):
#         self.backend_url = backend_url
#         self.session_id = None
#         self.user_id = "default"
#         self.chat_history = []
#         self.config = self.load_config()
        
#     def load_config(self) -> Dict[str, Any]:
#         config_path = Path("src/config.yaml")
#         if config_path.exists():
#             with open(config_path, 'r') as file:
#                 return yaml.safe_load(file)
#         return {}
    
#     def create_session(self):
#         """Create a new chat session"""
#         try:
#             response = requests.post(
#                 f"{self.backend_url}/sessions",
#                 json={"user_id": self.user_id}
#             )
#             if response.status_code == 200:
#                 self.session_id = response.json()["session_id"]
#                 logger.info(f"Created new session: {self.session_id}")
#             else:
#                 logger.error(f"Failed to create session: {response.text}")
#         except Exception as e:
#             logger.error(f"Error creating session: {str(e)}")
    
#     def chat(self, message: str, history: List[List[str]], image: Optional[str] = None):
#         """Process chat message and return response"""
#         if not self.session_id:
#             self.create_session()
        
#         try:
#             if image:
#                 # Handle image chat
#                 files = {
#                     'image': ('image.png', open(image, 'rb'), 'image/png'),
#                     'message': (None, message),
#                     'user_id': (None, self.user_id),
#                     'session_id': (None, self.session_id)
#                 }
#                 response = requests.post(
#                     f"{self.backend_url}/chat/image",
#                     files=files
#                 )
#             else:
#                 # Handle text chat
#                 response = requests.post(
#                     f"{self.backend_url}/chat",
#                     json={
#                         "message": message,
#                         "user_id": self.user_id,
#                         "session_id": self.session_id,
#                         "include_recent": True
#                     }
#                 )
            
#             if response.status_code == 200:
#                 data = response.json()
#                 history.append([message, data["response"]])
#                 return history, ""
#             else:
#                 error_msg = f"Error: {response.text}"
#                 history.append([message, error_msg])
#                 return history, ""
                
#         except Exception as e:
#             error_msg = f"Error: {str(e)}"
#             history.append([message, error_msg])
#             return history, ""
    
#     def clear_chat(self):
#         """Clear chat history and create new session"""
#         self.create_session()
#         return [], None
    
#     def create_interface(self):
#         """Create and return the Gradio interface"""
#         with gr.Blocks(
#             title="AI Chat with Memory",
#             theme=gr.themes.Soft(),
#             css="footer {visibility: hidden}"
#         ) as interface:
#             gr.Markdown("# 🤖 AI Chat with Memory")
#             gr.Markdown("Chat with an AI that remembers your conversations!")
            
#             with gr.Row():
#                 with gr.Column(scale=4):
#                     chatbot = gr.Chatbot(
#                         height=600,
#                         show_copy_button=True,
#                         show_share_button=True,
#                         avatar_images=(
#                             "👤",  # User avatar
#                             "🤖"   # Bot avatar
#                         )
#                     )
                    
#                     with gr.Row():
#                         msg = gr.Textbox(
#                             placeholder="Type your message here...",
#                             show_label=False,
#                             container=False,
#                             scale=8
#                         )
#                         submit = gr.Button("Send", variant="primary", scale=1)
                    
#                     with gr.Row():
#                         image_input = gr.Image(
#                             type="filepath",
#                             label="Upload Image (Optional)",
#                             show_label=True
#                         )
#                         clear = gr.Button("Clear Chat", variant="secondary")
                
#                 with gr.Column(scale=1):
#                     gr.Markdown("### 📝 Features")
#                     gr.Markdown("""
#                     - 💬 Text chat with memory
#                     - 🖼️ Image analysis
#                     - 🔄 Conversation history
#                     - 🧠 Context-aware responses
#                     """)
                    
#                     gr.Markdown("### ⚙️ Settings")
#                     with gr.Group():
#                         user_id = gr.Textbox(
#                             label="User ID",
#                             value=self.user_id,
#                             interactive=True
#                         )
                        
#                         def update_user_id(new_id):
#                             self.user_id = new_id
#                             self.create_session()
#                             return new_id
                        
#                         user_id.change(
#                             fn=update_user_id,
#                             inputs=[user_id],
#                             outputs=[user_id]
#                         )
            
#             # Event handlers
#             submit.click(
#                 fn=self.chat,
#                 inputs=[msg, chatbot, image_input],
#                 outputs=[chatbot, msg]
#             )
            
#             msg.submit(
#                 fn=self.chat,
#                 inputs=[msg, chatbot, image_input],
#                 outputs=[chatbot, msg]
#             )
            
#             clear.click(
#                 fn=self.clear_chat,
#                 inputs=[],
#                 outputs=[chatbot, image_input]
#             )
            
#             # Initialize session
#             interface.load(self.create_session)
        
#         return interface

# def main():
#     parser = argparse.ArgumentParser(description="Gradio Interface for AI Chat")
#     parser.add_argument("--backend-url", default="http://localhost:8000",
#                        help="Backend API URL")
#     parser.add_argument("--host", default="0.0.0.0",
#                        help="Host to run the Gradio interface")
#     parser.add_argument("--port", type=int, default=7860,
#                        help="Port to run the Gradio interface")
#     parser.add_argument("--share", action="store_true",
#                        help="Share the interface publicly")
    
#     args = parser.parse_args()
    
#     interface = GradioInterface(args.backend_url)
#     demo = interface.create_interface()
    
#     demo.launch(
#         server_name=args.host,
#         server_port=args.port,
#         share=args.share
#     )

# if __name__ == "__main__":
#     main()


import gradio as gr
import requests
import json
import os
import time
from typing import List, Dict, Any, Optional, Tuple
import base64
from datetime import datetime
import uuid
import yaml

class GradioClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session_id = str(uuid.uuid4())
        self.user_id = "gradio_user"
        self.conversation_history = []
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from config.yaml"""
        try:
            with open("src/config.yaml", 'r') as file:
                return yaml.safe_load(file)
        except Exception as e:
            print(f"Could not load config: {e}")
            return {}
    
    def check_server_health(self) -> Tuple[bool, str]:
        """Check if the FastAPI server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                return True, f"Server healthy - {data.get('memory_stats', {}).get('total', 0)} memories stored"
            return False, f"Server returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to server. Make sure FastAPI server is running on localhost:8000"
        except Exception as e:
            return False, f"Health check failed: {str(e)}"
    
    def send_message(self, message: str, include_recent: bool = True) -> Tuple[str, List[Dict], float]:
        """Send a text message to the chatbot"""
        try:
            payload = {
                "message": message,
                "user_id": self.user_id,
                "session_id": self.session_id,
                "include_recent": include_recent
            }
            
            response = requests.post(
                f"{self.base_url}/chat",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["response"], data.get("memories_used", []), data.get("processing_time", 0)
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                return error_msg, [], 0
                
        except requests.exceptions.Timeout:
            return "Request timed out. Please try again.", [], 0
        except Exception as e:
            return f"Error sending message: {str(e)}", [], 0
    
    def send_message_with_image(self, message: str, image_path: str) -> Tuple[str, str, List[Dict], float]:
        """Send a message with an image to the chatbot"""
        try:
            with open(image_path, 'rb') as image_file:
                files = {
                    'image': (os.path.basename(image_path), image_file, 'image/jpeg')
                }
                data = {
                    'message': message,
                    'user_id': self.user_id,
                    'session_id': self.session_id
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/image",
                    files=files,
                    data=data,
                    timeout=60
                )
            
            if response.status_code == 200:
                result = response.json()
                return (
                    result["response"], 
                    result.get("image_analysis", ""), 
                    result.get("memories_used", []), 
                    result.get("processing_time", 0)
                )
            else:
                error_msg = f"Error {response.status_code}: {response.text}"
                return error_msg, "", [], 0
                
        except Exception as e:
            return f"Error sending image: {str(e)}", "", [], 0
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get user statistics"""
        try:
            response = requests.get(f"{self.base_url}/users/{self.user_id}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            return {"error": f"Status {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def get_memories(self, limit: int = 10) -> List[Dict]:
        """Get stored memories"""
        try:
            response = requests.get(
                f"{self.base_url}/memories",
                params={"user_id": self.user_id, "limit": limit},
                timeout=10
            )
            if response.status_code == 200:
                return response.json().get("memories", [])
            return []
        except Exception as e:
            print(f"Error getting memories: {e}")
            return []
    
    def clear_cache(self) -> str:
        """Clear the server cache"""
        try:
            response = requests.post(f"{self.base_url}/cache/clear", timeout=10)
            if response.status_code == 200:
                return "Cache cleared successfully"
            return f"Error clearing cache: {response.status_code}"
        except Exception as e:
            return f"Error clearing cache: {str(e)}"

# Initialize client
client = GradioClient()

def format_memories(memories: List[Dict]) -> str:
    """Format memories for display"""
    if not memories:
        return "No relevant memories found"
    
    formatted = "**Relevant Memories Used:**\n\n"
    for i, memory in enumerate(memories, 1):
        score = memory.get('similarity_score', 0)
        text = memory.get('text', '')[:200] + "..." if len(memory.get('text', '')) > 200 else memory.get('text', '')
        formatted += f"{i}. **Relevance: {score:.2f}**\n{text}\n\n"
    
    return formatted

def format_conversation_history(history: List[Tuple[str, str]]) -> str:
    """Format conversation history"""
    if not history:
        return "No conversation history"
    
    formatted = "**Recent Conversation:**\n\n"
    for i, (user_msg, bot_msg) in enumerate(history[-5:], 1):
        formatted += f"**Turn {i}:**\n"
        formatted += f"👤 **You:** {user_msg}\n"
        formatted += f"🤖 **Bot:** {bot_msg}\n\n"
    
    return formatted

def chat_with_bot(message: str, history: List[Tuple[str, str]], include_recent: bool = True) -> Tuple[List[Tuple[str, str]], str, str, str]:
    """Handle chat messages"""
    if not message.strip():
        return history, "", "Please enter a message", ""
    
    # Check server health first
    is_healthy, health_msg = client.check_server_health()
    if not is_healthy:
        return history, "", f"⚠️ Server Error: {health_msg}", ""
    
    # Send message
    response, memories_used, processing_time = client.send_message(message, include_recent)
    
    # Update history
    history.append((message, response))
    client.conversation_history = history
    
    # Format additional info
    memories_info = format_memories(memories_used)
    processing_info = f"⏱️ **Processing Time:** {processing_time:.2f}s\n\n📊 **Memories Retrieved:** {len(memories_used)}"
    
    return history, "", memories_info, processing_info

def chat_with_image(message: str, image, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str, str, str, str]:
    """Handle chat with image"""
    if not message.strip():
        return history, "", "", "Please enter a message", ""
    
    if image is None:
        return history, "", "", "Please upload an image", ""
    
    # Check server health
    is_healthy, health_msg = client.check_server_health()
    if not is_healthy:
        return history, "", "", f"⚠️ Server Error: {health_msg}", ""
    
    try:
        # Send message with image
        response, image_analysis, memories_used, processing_time = client.send_message_with_image(message, image)
        
        # Update history
        history.append((f"{message} [Image uploaded]", response))
        client.conversation_history = history
        
        # Format info
        memories_info = format_memories(memories_used)
        processing_info = f"⏱️ **Processing Time:** {processing_time:.2f}s\n\n📊 **Memories Retrieved:** {len(memories_used)}"
        
        return history, "", memories_info, processing_info, image_analysis
        
    except Exception as e:
        return history, "", "", f"Error processing image: {str(e)}", ""

def get_stats() -> Tuple[str, str]:
    """Get user and server statistics"""
    try:
        # Get user stats
        user_stats = client.get_user_stats()
        
        # Get server health
        is_healthy, health_msg = client.check_server_health()
        
        user_info = f"""
        **User Statistics:**
        - User ID: {user_stats.get('user_id', 'Unknown')}
        - Total Messages: {user_stats.get('total_messages', 0)}
        - Stored Memories: {user_stats.get('memory_count', 0)}
        - Last Activity: {user_stats.get('last_activity', 'Never')}
        - Session ID: {client.session_id}
        """
        
        server_info = f"""
        **Server Status:**
        - Health: {'✅ Healthy' if is_healthy else '❌ Unhealthy'}
        - Details: {health_msg}
        - Base URL: {client.base_url}
        """
        
        return user_info, server_info
        
    except Exception as e:
        return f"Error getting stats: {str(e)}", "Error connecting to server"

def view_memories() -> str:
    """View stored memories"""
    try:
        memories = client.get_memories(limit=20)
        
        if not memories:
            return "No memories stored yet. Start chatting to build memory!"
        
        formatted = "**Stored Memories:**\n\n"
        for i, memory in enumerate(memories, 1):
            timestamp = memory.get('timestamp', 'Unknown')
            text = memory.get('text', '')
            access_count = memory.get('access_count', 0)
            
            # Truncate long memories
            display_text = text[:300] + "..." if len(text) > 300 else text
            
            formatted += f"**{i}. Memory from {timestamp}**\n"
            formatted += f"📝 {display_text}\n"
            formatted += f"🔍 Accessed {access_count} times\n\n"
        
        return formatted
        
    except Exception as e:
        return f"Error retrieving memories: {str(e)}"

def clear_conversation() -> Tuple[List, str]:
    """Clear conversation history"""
    client.conversation_history = []
    client.session_id = str(uuid.uuid4())  # New session
    return [], "Conversation cleared! Started new session."

def clear_server_cache() -> str:
    """Clear server cache"""
    return client.clear_cache()

# Create Gradio interface
with gr.Blocks(
    title="Advanced Chatbot with Memory",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .chat-container {
        height: 500px !important;
    }
    """
) as app:
    
    gr.Markdown("""
    # 🤖 Advanced Chatbot with Memory & Image Processing
    
    This chatbot features:
    - 🧠 **Long-term Memory**: Remembers past conversations
    - 🖼️ **Image Analysis**: Upload and discuss images
    - 📊 **Smart Context**: Uses relevant memories for better responses
    - ⚡ **Real-time Stats**: Monitor performance and usage
    
    **Make sure your FastAPI server is running on localhost:8000**
    """)
    
    with gr.Tabs():
        # Main Chat Tab
        with gr.TabItem("💬 Chat", id="chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=500,
                        show_copy_button=True,
                        container=True
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your Message",
                            placeholder="Type your message here...",
                            lines=2,
                            scale=4
                        )
                        send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                    
                    with gr.Row():
                        include_recent = gr.Checkbox(
                            label="Include Recent Messages in Context",
                            value=True
                        )
                        clear_btn = gr.Button("Clear Conversation 🗑️", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📊 Chat Info")
                    processing_info = gr.Markdown("Ready to chat!")
                    
                    gr.Markdown("### 🧠 Memory Usage")
                    memories_info = gr.Markdown("No memories used yet")
        
        # Image Chat Tab
        with gr.TabItem("🖼️ Image Chat", id="image_chat"):
            with gr.Row():
                with gr.Column(scale=2):
                    image_chatbot = gr.Chatbot(
                        label="Image Conversation",
                        height=400
                    )
                    
                    with gr.Row():
                        image_msg = gr.Textbox(
                            label="Message about the image",
                            placeholder="Describe what you want to know about the image...",
                            lines=2,
                            scale=3
                        )
                        image_send_btn = gr.Button("Send 📤", variant="primary", scale=1)
                    
                    image_input = gr.Image(
                        label="Upload Image",
                        type="filepath",
                        height=200
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Image Analysis")
                    image_analysis = gr.Markdown("Upload an image to see analysis")
                    
                    gr.Markdown("### 📊 Processing Info")
                    image_processing_info = gr.Markdown("Ready for image chat!")
                    
                    gr.Markdown("### 🧠 Memory Usage")
                    image_memories_info = gr.Markdown("No memories used yet")
        
        # Statistics Tab
        with gr.TabItem("📈 Statistics", id="stats"):
            with gr.Row():
                stats_btn = gr.Button("Refresh Stats 🔄", variant="primary")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 👤 User Statistics")
                    user_stats = gr.Markdown("Click 'Refresh Stats' to load")
                
                with gr.Column():
                    gr.Markdown("### 🖥️ Server Status")
                    server_stats = gr.Markdown("Click 'Refresh Stats' to load")
        
        # Memory Management Tab
        with gr.TabItem("🧠 Memory", id="memory"):
            with gr.Row():
                with gr.Column():
                    view_memories_btn = gr.Button("View Stored Memories 👁️", variant="primary")
                    clear_cache_btn = gr.Button("Clear Server Cache 🗑️", variant="secondary")
                
            memories_display = gr.Markdown("Click 'View Stored Memories' to see what the bot remembers")
            cache_status = gr.Markdown("")
        
        # Settings Tab
        with gr.TabItem("⚙️ Settings", id="settings"):
            gr.Markdown("### 🔧 Configuration")
            
            # Load and display config
            config = client.load_config()
            
            gr.Markdown(f"""
            **Current Configuration:**
            
            - **Memory Threshold:** {config.get('memory', {}).get('memory_relevance_threshold', 0.75)}
            - **Max Memories per Query:** {config.get('memory', {}).get('max_memories_per_query', 5)}
            - **Max Total Memories:** {config.get('memory', {}).get('max_total_memories', 10000)}
            - **Image Upload:** {'✅ Enabled' if config.get('ui', {}).get('features', {}).get('image_upload', True) else '❌ Disabled'}
            - **Caching:** {'✅ Enabled' if config.get('performance', {}).get('caching', {}).get('enabled', True) else '❌ Disabled'}
            
            **Server Configuration:**
            - **API Host:** {config.get('api', {}).get('host', '0.0.0.0')}
            - **API Port:** {config.get('api', {}).get('port', 8000)}
            - **Primary Model:** {config.get('models', {}).get('language', {}).get('primary', 'gemini-1.5-flash')}
            """)
    
    # Event handlers for main chat
    def send_message(message, history, include_recent):
        return chat_with_bot(message, history, include_recent)
    
    send_btn.click(
        send_message,
        inputs=[msg, chatbot, include_recent],
        outputs=[chatbot, msg, memories_info, processing_info]
    )
    
    msg.submit(
        send_message,
        inputs=[msg, chatbot, include_recent],
        outputs=[chatbot, msg, memories_info, processing_info]
    )
    
    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, processing_info]
    )
    
    # Event handlers for image chat
    image_send_btn.click(
        chat_with_image,
        inputs=[image_msg, image_input, image_chatbot],
        outputs=[image_chatbot, image_msg, image_memories_info, image_processing_info, image_analysis]
    )
    
    # Event handlers for stats
    stats_btn.click(
        get_stats,
        outputs=[user_stats, server_stats]
    )
    
    # Event handlers for memory management
    view_memories_btn.click(
        view_memories,
        outputs=[memories_display]
    )
    
    clear_cache_btn.click(
        clear_server_cache,
        outputs=[cache_status]
    )

if __name__ == "__main__":
    # Check server health on startup
    is_healthy, health_msg = client.check_server_health()
    
    if is_healthy:
        print("✅ FastAPI server is running and healthy!")
        print(f"📊 {health_msg}")
    else:
        print("⚠️  Warning: FastAPI server is not responding")
        print(f"❌ {health_msg}")
        print("\n🚀 Make sure to start your FastAPI server first:")
        print("   python src/fastapi_server.py")
    
    print(f"\n🌐 Starting Gradio interface...")
    print(f"🔗 Connecting to FastAPI server at: {client.base_url}")
    
    # Launch Gradio app
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True,
        show_error=True
    )