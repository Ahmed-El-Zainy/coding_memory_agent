import gradio as gr
import requests
import os
import json
import logging
import argparse
from typing import List, Dict, Any, Optional
from datetime import datetime
import yaml
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GradioInterface:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url
        self.session_id = None
        self.user_id = "default"
        self.chat_history = []
        self.config = self.load_config()
        
    def load_config(self) -> Dict[str, Any]:
        config_path = Path("src/config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as file:
                return yaml.safe_load(file)
        return {}
    
    def create_session(self):
        """Create a new chat session"""
        try:
            response = requests.post(
                f"{self.backend_url}/sessions",
                json={"user_id": self.user_id}
            )
            if response.status_code == 200:
                self.session_id = response.json()["session_id"]
                logger.info(f"Created new session: {self.session_id}")
            else:
                logger.error(f"Failed to create session: {response.text}")
        except Exception as e:
            logger.error(f"Error creating session: {str(e)}")
    
    def chat(self, message: str, history: List[List[str]], image: Optional[str] = None):
        """Process chat message and return response"""
        if not self.session_id:
            self.create_session()
        
        try:
            if image:
                # Handle image chat
                files = {
                    'image': ('image.png', open(image, 'rb'), 'image/png'),
                    'message': (None, message),
                    'user_id': (None, self.user_id),
                    'session_id': (None, self.session_id)
                }
                response = requests.post(
                    f"{self.backend_url}/chat/image",
                    files=files
                )
            else:
                # Handle text chat
                response = requests.post(
                    f"{self.backend_url}/chat",
                    json={
                        "message": message,
                        "user_id": self.user_id,
                        "session_id": self.session_id,
                        "include_recent": True
                    }
                )
            
            if response.status_code == 200:
                data = response.json()
                history.append([message, data["response"]])
                return history, ""
            else:
                error_msg = f"Error: {response.text}"
                history.append([message, error_msg])
                return history, ""
                
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            history.append([message, error_msg])
            return history, ""
    
    def clear_chat(self):
        """Clear chat history and create new session"""
        self.create_session()
        return [], None
    
    def create_interface(self):
        """Create and return the Gradio interface"""
        with gr.Blocks(
            title="AI Chat with Memory",
            theme=gr.themes.Soft(),
            css="footer {visibility: hidden}"
        ) as interface:
            gr.Markdown("# 🤖 AI Chat with Memory")
            gr.Markdown("Chat with an AI that remembers your conversations!")
            
            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        height=600,
                        show_copy_button=True,
                        show_share_button=True,
                        avatar_images=(
                            "👤",  # User avatar
                            "🤖"   # Bot avatar
                        )
                    )
                    
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            show_label=False,
                            container=False,
                            scale=8
                        )
                        submit = gr.Button("Send", variant="primary", scale=1)
                    
                    with gr.Row():
                        image_input = gr.Image(
                            type="filepath",
                            label="Upload Image (Optional)",
                            show_label=True
                        )
                        clear = gr.Button("Clear Chat", variant="secondary")
                
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 Features")
                    gr.Markdown("""
                    - 💬 Text chat with memory
                    - 🖼️ Image analysis
                    - 🔄 Conversation history
                    - 🧠 Context-aware responses
                    """)
                    
                    gr.Markdown("### ⚙️ Settings")
                    with gr.Group():
                        user_id = gr.Textbox(
                            label="User ID",
                            value=self.user_id,
                            interactive=True
                        )
                        
                        def update_user_id(new_id):
                            self.user_id = new_id
                            self.create_session()
                            return new_id
                        
                        user_id.change(
                            fn=update_user_id,
                            inputs=[user_id],
                            outputs=[user_id]
                        )
            
            # Event handlers
            submit.click(
                fn=self.chat,
                inputs=[msg, chatbot, image_input],
                outputs=[chatbot, msg]
            )
            
            msg.submit(
                fn=self.chat,
                inputs=[msg, chatbot, image_input],
                outputs=[chatbot, msg]
            )
            
            clear.click(
                fn=self.clear_chat,
                inputs=[],
                outputs=[chatbot, image_input]
            )
            
            # Initialize session
            interface.load(self.create_session)
        
        return interface

def main():
    parser = argparse.ArgumentParser(description="Gradio Interface for AI Chat")
    parser.add_argument("--backend-url", default="http://localhost:8000",
                       help="Backend API URL")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to run the Gradio interface")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the Gradio interface")
    parser.add_argument("--share", action="store_true",
                       help="Share the interface publicly")
    
    args = parser.parse_args()
    
    interface = GradioInterface(args.backend_url)
    demo = interface.create_interface()
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main()