# -*- coding: utf-8 -*-
"""
LangChain Chatbot Memory Implementation (Compatible Version)
A comprehensive guide to adding memory to chatbots using stable LangChain components
"""

import getpass
import os
from typing import List, Dict, Any
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI

class InMemoryHistory:
    """Simple in-memory chat history storage"""
    
    def __init__(self):
        self.store = {}
    
    def get_session_history(self, session_id: str) -> ChatMessageHistory:
        if session_id not in self.store:
            self.store[session_id] = ChatMessageHistory()
        return self.store[session_id]

class ChatbotMemoryManager:
    """
    A comprehensive chatbot memory manager that demonstrates different memory strategies
    """
    
    def __init__(self):
        """Initialize the chatbot with Google's Gemini model"""
        self.setup_environment()
        self.model = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")
        self.history_store = InMemoryHistory()
        
    def setup_environment(self):
        """Setup environment variables for API keys"""
        if not os.environ.get("GOOGLE_API_KEY"):
            try:
                api_key = getpass.getpass("Enter your Google API Key: ")
                os.environ["GOOGLE_API_KEY"] = api_key
            except KeyboardInterrupt:
                print("\nAPI key setup cancelled. Please set GOOGLE_API_KEY environment variable.")
                exit(1)
    
    def basic_message_passing(self):
        """
        Demonstrates basic memory through explicit message passing
        """
        print("=== Basic Message Passing Example ===")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        chain = prompt | self.model
        
        # Example conversation with manual message management
        conversation_history = [
            HumanMessage(content="Translate from English to French: I love programming."),
            AIMessage(content="J'adore la programmation."),
            HumanMessage(content="What did you just say?"),
        ]
        
        try:
            response = chain.invoke({"messages": conversation_history})
            print(f"Response: {response.content}")
            return response
        except Exception as e:
            print(f"Error in basic message passing: {e}")
            return None
    
    def create_persistent_chatbot(self):
        """
        Creates a chatbot with automatic history management using RunnableWithMessageHistory
        """
        print("=== Creating Persistent Chatbot ===")
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        runnable = prompt | self.model
        
        # Create chatbot with message history
        chatbot = RunnableWithMessageHistory(
            runnable,
            self.history_store.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        return chatbot
    
    def demo_persistent_chat(self, chatbot):
        """
        Demonstrates the persistent chatbot functionality
        """
        print("=== Persistent Chat Demo ===")
        
        try:
            # First interaction
            response1 = chatbot.invoke(
                {"input": "Translate to French: I love programming."},
                config={"configurable": {"session_id": "demo_session_1"}},
            )
            print(f"First response: {response1.content}")
            
            # Second interaction - should remember context
            response2 = chatbot.invoke(
                {"input": "What did I just ask you?"},
                config={"configurable": {"session_id": "demo_session_1"}},
            )
            print(f"Second response: {response2.content}")
            
            return response1, response2
        except Exception as e:
            print(f"Error in persistent chat demo: {e}")
            return None, None
    
    def create_trimmed_memory_chatbot(self, max_messages=4):
        """
        Creates a chatbot that trims message history to manage context window
        """
        print(f"=== Creating Trimmed Memory Chatbot (max {max_messages} messages) ===")
        
        class TrimmedChatHistory(ChatMessageHistory):
            def __init__(self, max_messages=4):
                super().__init__()
                self.max_messages = max_messages
            
            def add_message(self, message: BaseMessage) -> None:
                super().add_message(message)
                # Keep only the last max_messages
                if len(self.messages) > self.max_messages:
                    self.messages = self.messages[-self.max_messages:]
        
        class TrimmedHistoryStore:
            def __init__(self, max_messages=4):
                self.store = {}
                self.max_messages = max_messages
            
            def get_session_history(self, session_id: str) -> TrimmedChatHistory:
                if session_id not in self.store:
                    self.store[session_id] = TrimmedChatHistory(self.max_messages)
                return self.store[session_id]
        
        trimmed_store = TrimmedHistoryStore(max_messages)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        runnable = prompt | self.model
        
        chatbot = RunnableWithMessageHistory(
            runnable,
            trimmed_store.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        return chatbot
    
    def demo_trimmed_memory(self, chatbot):
        """
        Demonstrates trimmed memory functionality
        """
        print("=== Trimmed Memory Demo ===")
        
        try:
            session_id = "trimmed_demo"
            
            # Build up conversation history
            chatbot.invoke(
                {"input": "Hey there! I'm Nemo."},
                config={"configurable": {"session_id": session_id}},
            )
            
            chatbot.invoke(
                {"input": "How are you today?"},
                config={"configurable": {"session_id": session_id}},
            )
            
            chatbot.invoke(
                {"input": "I like programming in Python."},
                config={"configurable": {"session_id": session_id}},
            )
            
            # This should trigger trimming, potentially losing the name
            response = chatbot.invoke(
                {"input": "What is my name?"},
                config={"configurable": {"session_id": session_id}},
            )
            
            print(f"Response: {response.content}")
            return response
        except Exception as e:
            print(f"Error in trimmed memory demo: {e}")
            return None
    
    def create_summary_memory_chatbot(self):
        """
        Creates a chatbot that summarizes old conversations when they get too long
        """
        print("=== Creating Summary Memory Chatbot ===")
        
        class SummaryChatHistory(ChatMessageHistory):
            def __init__(self, model, summary_threshold=6):
                super().__init__()
                self.model = model
                self.summary_threshold = summary_threshold
                self.summary = None
                self.session_id = None  # Add session_id field
            
            def add_message(self, message: BaseMessage) -> None:
                super().add_message(message)
                
                # Summarize if we have too many messages
                if len(self.messages) > self.summary_threshold:
                    self.create_summary()
            
            def create_summary(self):
                """Create a summary of the conversation and trim messages"""
                try:
                    # Create summary prompt
                    messages_to_summarize = self.messages[:-2]  # Keep last 2 messages
                    
                    summary_prompt = (
                        "Please create a concise summary of the following conversation, "
                        "including important details and context:\n\n"
                    )
                    
                    # Add messages to prompt
                    for msg in messages_to_summarize:
                        role = "Human" if isinstance(msg, HumanMessage) else "Assistant"
                        summary_prompt += f"{role}: {msg.content}\n"
                    
                    summary_prompt += "\nSummary:"
                    
                    # Generate summary
                    summary_response = self.model.invoke([HumanMessage(content=summary_prompt)])
                    self.summary = summary_response.content
                    
                    # Keep only recent messages and add summary
                    recent_messages = self.messages[-2:]
                    self.messages = [SystemMessage(content=f"Previous conversation summary: {self.summary}")] + recent_messages
                    
                except Exception as e:
                    print(f"Error creating summary: {e}")
        
        class SummaryHistoryStore:
            def __init__(self, model, summary_threshold=6):
                self.store = {}
                self.model = model
                self.summary_threshold = summary_threshold
            
            def get_session_history(self, session_id: str) -> SummaryChatHistory:
                if session_id not in self.store:
                    self.store[session_id] = SummaryChatHistory(self.model, self.summary_threshold)
                return self.store[session_id]
        
        summary_store = SummaryHistoryStore(self.model)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        runnable = prompt | self.model
        
        chatbot = RunnableWithMessageHistory(
            runnable,
            summary_store.get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )
        
        return chatbot
    
    def demo_summary_memory(self, chatbot):
        """
        Demonstrates summary memory functionality
        """
        print("=== Summary Memory Demo ===")
        
        try:
            session_id = "summary_demo"
            
            # Build up a longer conversation
            inputs = [
                "Hey there! I'm Nemo.",
                "I'm a software engineer.",
                "I love working with Python and AI.",
                "How are you today?",
                "What's the weather like?",
                "Can you help me with coding?",
                "What did I say my name was?"
            ]
            
            for i, input_text in enumerate(inputs):
                response = chatbot.invoke(
                    {"input": input_text},
                    config={"configurable": {"session_id": session_id}},
                )
                print(f"Input {i+1}: {input_text}")
                print(f"Response {i+1}: {response.content}\n")
            
            return response
        except Exception as e:
            print(f"Error in summary memory demo: {e}")
            return None
    
    def run_all_demos(self):
        """
        Run all memory demonstration examples
        """
        print("Starting LangChain Chatbot Memory Demonstrations")
        print("=" * 60)
        
        try:
            # Basic message passing
            print("1. Testing Basic Message Passing...")
            self.basic_message_passing()
            print("\n" + "-" * 40 + "\n")
            
            # Persistent chatbot
            print("2. Testing Persistent Memory...")
            persistent_app = self.create_persistent_chatbot()
            self.demo_persistent_chat(persistent_app)
            print("\n" + "-" * 40 + "\n")
            
            # Trimmed memory chatbot
            print("3. Testing Trimmed Memory...")
            trimmed_app = self.create_trimmed_memory_chatbot(max_messages=4)
            self.demo_trimmed_memory(trimmed_app)
            print("\n" + "-" * 40 + "\n")
            
            # Summary memory chatbot
            print("4. Testing Summary Memory...")
            summary_app = self.create_summary_memory_chatbot()
            self.demo_summary_memory(summary_app)
            
            print("\n" + "=" * 60)
            print("All demonstrations completed successfully!")
            
        except Exception as e:
            print(f"Error during demonstration: {str(e)}")
            print("Make sure you have set up your Google API key correctly.")
            print("You can set it as an environment variable: export GOOGLE_API_KEY='your-key-here'")

def main():
    """
    Main function to run the chatbot memory demonstrations
    """
    print("LangChain Chatbot Memory Implementation")
    print("This script demonstrates different memory strategies for chatbots")
    print("=" * 60)
    
    # Create and run the memory manager
    memory_manager = ChatbotMemoryManager()
    memory_manager.run_all_demos()

if __name__ == "__main__":
    main()