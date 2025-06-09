# import gradio as gr
# import requests
# import json
# import argparse
# import os

# class ChatbotUI:
#     def __init__(self, backend_url: str):
#         self.backend_url = backend_url.rstrip('/')
        
#     def chat_with_bot(self, message, history):
#         try:
#             response = requests.post(
#                 f"{self.backend_url}/chat",
#                 json={"message": message},
#                 headers={"Content-Type": "application/json"}
#             )
            
#             if response.status_code == 200:
#                 data = response.json()
#                 bot_response = data["response"]
#                 memories_count = len(data["memories_used"])
                
#                 if memories_count > 0:
#                     bot_response += f"\n\n💭 *Used {memories_count} memories*"
                
#                 history.append((message, bot_response))
#                 return history, ""
#             else:
#                 history.append((message, f"Error: {response.status_code}"))
#                 return history, ""
                
#         except Exception as e:
#             history.append((message, f"Connection error: {str(e)}"))
#             return history, ""
    
#     def get_memory_stats(self):
#         try:
#             response = requests.get(f"{self.backend_url}/memories")
#             if response.status_code == 200:
#                 data = response.json()
#                 return f"Total memories: {data['count']}"
#             return "Failed to fetch memory stats"
#         except:
#             return "Backend not available"
    
#     def create_interface(self):
#         with gr.Blocks(title="Chatbot with Memory") as demo:
#             gr.Markdown("# 🤖 Chatbot with Memory")
#             gr.Markdown("This chatbot remembers your conversations and uses relevant memories to provide better responses.")
            
#             chatbot = gr.Chatbot(height=500)
            
#             with gr.Row():
#                 msg = gr.Textbox(
#                     placeholder="Type your message here...",
#                     container=False,
#                     scale=4
#                 )
#                 submit = gr.Button("Send", scale=1)
            
#             with gr.Row():
#                 clear = gr.Button("Clear Chat")
#                 memory_stats = gr.Textbox(
#                     label="Memory Stats",
#                     value=self.get_memory_stats(),
#                     interactive=False
#                 )
            
#             def update_stats():
#                 return self.get_memory_stats()
            
#             submit.click(
#                 self.chat_with_bot,
#                 inputs=[msg, chatbot],
#                 outputs=[chatbot, msg]
#             ).then(
#                 update_stats,
#                 outputs=[memory_stats]
#             )
            
#             msg.submit(
#                 self.chat_with_bot,
#                 inputs=[msg, chatbot],
#                 outputs=[chatbot, msg]
#             ).then(
#                 update_stats,
#                 outputs=[memory_stats]
#             )
            
#             clear.click(lambda: ([], ""), outputs=[chatbot, msg])
        
#         return demo

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--backend-url", default="http://localhost:8000")
#     parser.add_argument("--host", default="0.0.0.0")
#     parser.add_argument("--port", type=int, default=7860)
#     parser.add_argument("--share", action="store_true")
#     args = parser.parse_args()
    
#     ui = ChatbotUI(args.backend_url)
#     demo = ui.create_interface()
    
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
import argparse
import os

class ChatbotUI:
    def __init__(self, backend_url: str):
        self.backend_url = backend_url.rstrip('/')
        
    def chat_with_bot(self, message, history):
        try:
            response = requests.post(
                f"{self.backend_url}/chat",
                json={"message": message},
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                bot_response = data["response"]
                memories_count = len(data["memories_used"])
                
                if memories_count > 0:
                    bot_response += f"\n\n💭 *Used {memories_count} memories*"
                
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": bot_response})
                return history, ""
            else:
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": f"Error: {response.status_code}"})
                return history, ""
                
        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"Connection error: {str(e)}"})
            return history, ""
    
    def get_memory_stats(self):
        try:
            response = requests.get(f"{self.backend_url}/memories")
            if response.status_code == 200:
                data = response.json()
                return f"Total memories: {data['count']}"
            return "Failed to fetch memory stats"
        except:
            return "Backend not available"
    
    def create_interface(self):
        with gr.Blocks(title="Chatbot with Memory") as demo:
            gr.Markdown("# 🤖 Chatbot with Memory")
            gr.Markdown("This chatbot remembers your conversations and uses relevant memories to provide better responses.")
            
            chatbot = gr.Chatbot(height=500, type="messages")
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    container=False,
                    scale=4
                )
                submit = gr.Button("Send", scale=1)
            
            with gr.Row():
                clear = gr.Button("Clear Chat")
                memory_stats = gr.Textbox(
                    label="Memory Stats",
                    value=self.get_memory_stats(),
                    interactive=False
                )
            
            def update_stats():
                return self.get_memory_stats()
            
            submit.click(
                self.chat_with_bot,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            ).then(
                update_stats,
                outputs=[memory_stats]
            )
            
            msg.submit(
                self.chat_with_bot,
                inputs=[msg, chatbot],
                outputs=[chatbot, msg]
            ).then(
                update_stats,
                outputs=[memory_stats]
            )
            
            clear.click(lambda: ([], ""), outputs=[chatbot, msg])
        
        return demo

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend-url", default="http://localhost:8000")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    ui = ChatbotUI(args.backend_url)
    demo = ui.create_interface()
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )

if __name__ == "__main__":
    main()