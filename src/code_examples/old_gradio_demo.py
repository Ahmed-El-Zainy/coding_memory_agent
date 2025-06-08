# # -*- coding: utf-8 -*-
# """
# Gradio Memory Chatbot with Caching and Enhanced Session Handling
# """

# import gradio as gr
# import hashlib
# import json
# import time
# import os
# from datetime import datetime, timedelta
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass
# from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_core.runnables.history import RunnableWithMessageHistory
# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_google_genai import ChatGoogleGenerativeAI

# @dataclass
# class CacheEntry:
#     response: str
#     timestamp: datetime
#     session_id: str
#     memory_type: str

# class ResponseCache:
#     def __init__(self, ttl_minutes: int = 60):
#         self.cache = {}
#         self.ttl = timedelta(minutes=ttl_minutes)

#     def _generate_key(self, message: str, session_id: str, memory_type: str) -> str:
#         content = f"{message}_{session_id}_{memory_type}"
#         return hashlib.md5(content.encode()).hexdigest()

#     def get(self, message: str, session_id: str, memory_type: str) -> Optional[str]:
#         key = self._generate_key(message, session_id, memory_type)
#         entry = self.cache.get(key)
#         if entry and datetime.now() - entry.timestamp < self.ttl:
#             return entry.response
#         elif entry:
#             del self.cache[key]
#         return None

#     def set(self, message: str, session_id: str, memory_type: str, response: str):
#         key = self._generate_key(message, session_id, memory_type)
#         self.cache[key] = CacheEntry(response, datetime.now(), session_id, memory_type)

#     def clear(self):
#         self.cache.clear()

#     def get_stats(self) -> Dict[str, Any]:
#         now = datetime.now()
#         active = sum(1 for v in self.cache.values() if now - v.timestamp < self.ttl)
#         return {
#             "total_entries": len(self.cache),
#             "active_entries": active,
#             "expired_entries": len(self.cache) - active,
#         }

# class EnhancedChatHistory(ChatMessageHistory):
#     def __init__(self, session_id: str):
#         super().__init__()
#         self.session_id = session_id
#         self.created_at = datetime.now()
#         self.message_count = 0

#     def add_message(self, message: BaseMessage) -> None:
#         super().add_message(message)
#         self.message_count += 1

# class TrimmedChatHistory(EnhancedChatHistory):
#     def __init__(self, session_id: str, max_messages: int = 10):
#         super().__init__(session_id)
#         self.max_messages = max_messages

#     def add_message(self, message: BaseMessage) -> None:
#         super().add_message(message)
#         if len(self.messages) > self.max_messages:
#             self.messages = self.messages[-self.max_messages:]

# class SummaryChatHistory(EnhancedChatHistory):
#     def __init__(self, session_id: str, model, summary_threshold: int = 12):
#         super().__init__(session_id)
#         self.model = model
#         self.summary_threshold = summary_threshold
#         self.summary = None
#         self.last_summary_count = 0

#     def add_message(self, message: BaseMessage) -> None:
#         super().add_message(message)
#         if (len(self.messages) - self.last_summary_count) > self.summary_threshold:
#             self._create_summary()

#     def _create_summary(self):
#         try:
#             msgs_to_summarize = self.messages[:-4]
#             if not msgs_to_summarize:
#                 return

#             content = "Conversation Summary Request:\n\n"
#             for msg in msgs_to_summarize:
#                 role = "User" if isinstance(msg, HumanMessage) else "Assistant"
#                 content += f"{role}: {msg.content}\n"
#             content += "\nPlease summarize this conversation:"

#             summary_response = self.model.invoke([HumanMessage(content=content)])
#             self.summary = summary_response.content

#             recent = self.messages[-4:]
#             self.messages = [SystemMessage(content=f"Summary: {self.summary}")] + recent
#             self.last_summary_count = len(self.messages)

#         except Exception as e:
#             print(f"Summary error: {e}")

# class MemoryManager:
#     def __init__(self, model):
#         self.model = model
#         self.stores = {"basic": {}, "trimmed": {}, "summary": {}}

#     def get_session_history(self, session_id: str, memory_type: str) -> EnhancedChatHistory:
#         if session_id not in self.stores[memory_type]:
#             if memory_type == "basic":
#                 self.stores[memory_type][session_id] = EnhancedChatHistory(session_id)
#             elif memory_type == "trimmed":
#                 self.stores[memory_type][session_id] = TrimmedChatHistory(session_id)
#             elif memory_type == "summary":
#                 self.stores[memory_type][session_id] = SummaryChatHistory(session_id, self.model)
#         return self.stores[memory_type][session_id]

#     def clear_session(self, session_id: str, memory_type: str = None):
#         if memory_type:
#             self.stores[memory_type].pop(session_id, None)
#         else:
#             for store in self.stores.values():
#                 store.pop(session_id, None)

#     def get_session_stats(self, session_id: str, memory_type: str) -> Dict[str, Any]:
#         history = self.stores[memory_type].get(session_id)
#         if history:
#             return {
#                 "message_count": history.message_count,
#                 "session_duration": str(datetime.now() - history.created_at),
#                 "current_messages": len(history.messages),
#                 "has_summary": getattr(history, 'summary', None) is not None
#             }
#         return {}

# class GradioMemoryChatbot:
#     def __init__(self):
#         self.setup_environment()
#         self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.7)
#         self.cache = ResponseCache(ttl_minutes=30)
#         self.memory_manager = MemoryManager(self.model)
#         self.chatbots = self._create_chatbots()
#         self._total_requests = 0
#         self._cache_hits = 0
#         self._start_time = datetime.now()

#     def setup_environment(self):
#         if not os.environ.get("GOOGLE_API_KEY"):
#             pass

#     def _ensure_session_id(self, session_id: str) -> str:
#         return session_id.strip() if session_id and session_id.strip() else f"session_{int(time.time())}"

#     def _create_chatbots(self):
#         prompt = ChatPromptTemplate.from_messages([
#             SystemMessage(content="You are a helpful AI with memory."),
#             MessagesPlaceholder(variable_name="history"),
#             ("human", "{input}"),
#         ])
#         runnable = prompt | self.model
#         return {
#             mt: RunnableWithMessageHistory(
#                 runnable,
#                 lambda sid, mt=mt: self.memory_manager.get_session_history(sid, mt),
#                 input_messages_key="input",
#                 history_messages_key="history"
#             ) for mt in ["basic", "trimmed", "summary"]
#         }

#     def chat_with_memory(self, message, history, session_id, memory_type, use_cache, api_key):
#         session_id = self._ensure_session_id(session_id)
#         if api_key:
#             os.environ["GOOGLE_API_KEY"] = api_key.strip()
#         if not os.environ.get("GOOGLE_API_KEY"):
#             return "Please provide your Google API key.", history, "❌ API Key Required"
        
#         self._total_requests += 1

#         if use_cache:
#             cached = self.cache.get(message, session_id, memory_type)
#             if cached:
#                 self._cache_hits += 1
#                 history.append((message, f"[CACHED] {cached}"))
#                 return "", history, self._get_status_info()

#         try:
#             chatbot = self.chatbots[memory_type]
#             response = chatbot.invoke({"input": message}, config={"configurable": {"session_id": session_id}})
#             response_text = response.content

#             if use_cache:
#                 self.cache.set(message, session_id, memory_type, response_text)
#             history.append((message, response_text))
#             return "", history, self._get_status_info()
#         except Exception as e:
#             error = f"Error: {e}"
#             history.append((message, error))
#             return "", history, error

#     def _get_status_info(self):
#         stats = self.cache.get_stats()
#         hit_rate = (self._cache_hits / self._total_requests * 100) if self._total_requests else 0
#         uptime = str(datetime.now() - self._start_time).split(".")[0]
#         return f"""📊 **Status:**
# • Requests: {self._total_requests}
# • Cache Hit Rate: {hit_rate:.1f}%
# • Active Cache Entries: {stats['active_entries']}
# • Uptime: {uptime}"""

#     def clear_conversation(self, session_id, memory_type):
#         self.memory_manager.clear_session(session_id, memory_type)
#         return [], "✅ Conversation cleared!"

#     def clear_cache(self):
#         self.cache.clear()
#         return "✅ Cache cleared!"

#     def get_session_info(self, session_id, memory_type):
#         stats = self.memory_manager.get_session_stats(session_id, memory_type)
#         if stats:
#             return f"""📋 **Session Info:**
# • Messages: {stats['message_count']}
# • Duration: {stats['session_duration']}
# • History: {stats['current_messages']} messages
# • Summary: {'Yes' if stats['has_summary'] else 'No'}"""
#         return "No session info available."

#     def generate_new_session_id(self):
#         return f"session_{int(time.time())}", [], "🆕 New session started!"

#     def create_interface(self):
#         with gr.Blocks(title="Memory Chatbot with Cache", theme=gr.themes.Soft()) as ui:
#             gr.Markdown("# 🤖 Memory Chatbot with Caching")

#             with gr.Row():
#                 with gr.Column(scale=3):
#                     chatbot_display = gr.Chatbot(label="Chat", height=400, show_copy_button=True)
#                     with gr.Row():
#                         message_input = gr.Textbox(label="Your Message", scale=4)
#                         send_button = gr.Button("Send", variant="primary")
#                     with gr.Row():
#                         clear_button = gr.Button("Clear Conversation")
#                         session_info_button = gr.Button("Session Info")

#                 with gr.Column(scale=1):
#                     gr.Markdown("### ⚙️ Settings")
#                     api_key_input = gr.Textbox(label="Google API Key", type="password")
#                     session_id_input = gr.Textbox(label="Session ID", value=f"session_{int(time.time())}")
#                     new_session_button = gr.Button("🔄 New Session")
#                     memory_type = gr.Radio(["basic", "trimmed", "summary"], value="basic", label="Memory Type")
#                     use_cache = gr.Checkbox(label="Enable Caching", value=True)

#                     gr.Markdown("### 📊 Status")
#                     status_display = gr.Markdown("Ready.")
#                     gr.Markdown("### 🧹 Actions")
#                     cache_clear_button = gr.Button("Clear Cache")
#                     session_info_display = gr.Markdown("")

#             send_button.click(self.chat_with_memory,
#                 inputs=[message_input, chatbot_display, session_id_input, memory_type, use_cache, api_key_input],
#                 outputs=[message_input, chatbot_display, status_display]
#             )

#             message_input.submit(self.chat_with_memory,
#                 inputs=[message_input, chatbot_display, session_id_input, memory_type, use_cache, api_key_input],
#                 outputs=[message_input, chatbot_display, status_display]
#             )

#             clear_button.click(self.clear_conversation,
#                 inputs=[session_id_input, memory_type],
#                 outputs=[chatbot_display, status_display]
#             )

#             session_info_button.click(self.get_session_info,
#                 inputs=[session_id_input, memory_type],
#                 outputs=[session_info_display]
#             )

#             cache_clear_button.click(self.clear_cache, outputs=[status_display])

#             new_session_button.click(self.generate_new_session_id,
#                 outputs=[session_id_input, chatbot_display, status_display]
#             )

#         return ui

# def main():
#     chatbot = GradioMemoryChatbot()
#     ui = chatbot.create_interface()
#     ui.launch(server_name="0.0.0.0", server_port=7860, debug=True, show_error=True)

# if __name__ == "__main__":
#     main()