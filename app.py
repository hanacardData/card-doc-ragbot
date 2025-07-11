# app.py
import gradio as gr
import asyncio
import os
import json
from datetime import datetime
import uuid
from utils.session_config import SessionConfigManager, ChatMessage
from utils.logging_config import setup_logging
from utils.graph_state import (
    GraphState, classify_intent, decide_path, generate_from_history,
    retrieve, grade_documents, generate, transform_query,
    decide_to_generate, grade_generation_v_documents_and_question
)
from langgraph.graph import StateGraph, END, START
from langgraph.errors import GraphRecursionError

class ChatbotApp:
    _workflow_semaphore = asyncio.Semaphore(20)

    def __init__(self):
        """Initialize the application"""
        self.session_manager = SessionConfigManager()
        self.logger = setup_logging()
        self.workflow = self._initialize_workflow()

    def _initialize_workflow(self):
        """Initialize workflow graph"""
        workflow = StateGraph(GraphState)

        # Define nodes
        workflow.add_node("classify_intent", classify_intent)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("grade_documents", grade_documents)
        workflow.add_node("generate", generate)
        workflow.add_node("generate_from_history", generate_from_history)
        workflow.add_node("transform_query", transform_query)

        # Build graph
        workflow.add_edge(START, "classify_intent")

        # Add conditional edges from intent classifier
        workflow.add_conditional_edges(
            "classify_intent",
            decide_path,
            {
                "generate_from_history": "generate_from_history",
                "transform_query": "transform_query",
                "retrieve": "retrieve",
            },
        )

        workflow.add_edge("retrieve", "grade_documents")
        workflow.add_conditional_edges(
            "grade_documents",
            decide_to_generate,
            {
                "transform_query": "transform_query",
                "generate": "generate",
            },
        )
        workflow.add_edge("transform_query", "retrieve")
        workflow.add_edge("generate_from_history", END)

        workflow.add_conditional_edges(
            "generate",
            grade_generation_v_documents_and_question,
            {
                "not supported": "generate",
                "useful": END,
                "not useful": "transform_query",
            },
        )

        return workflow.compile()

    async def save_chat_history(self, session_id, messages, is_append=False):
        """Save chat history to file"""
        try:
            os.makedirs('history', exist_ok=True)
            current_date = datetime.now().strftime("%Y-%m-%d")
            filename = f"{session_id}_chat_history_{current_date}.json"
            file_path = os.path.join('history', filename)

            if is_append and os.path.exists(file_path):
                with open(file_path, "r", encoding="utf-8") as f:
                    existing_messages = json.load(f)
                existing_messages.extend(messages)
                messages_to_save = existing_messages
            else:
                messages_to_save = messages

            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(messages_to_save, f, ensure_ascii=False, indent=4)

            self.logger.info(f"Chat history saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving chat history: {str(e)}")

    async def process_message(self, message, history, session_id):
        """Process user message and generate response"""
        try:
            self.session_manager.append_message(
                session_id,
                ChatMessage(role="user", content=message)
            )

            final_response = None
            async with self._workflow_semaphore:
                self.logger.info(f"Session {session_id} acquired semaphore")
                try:
                    graph_config = self.session_manager.get_graph_config(session_id)
                    async for chunk in self.workflow.astream(
                            {"question": message},
                            graph_config
                    ):
                        if 'generate_from_history' in chunk:
                            final_response = chunk['generate_from_history'].get('generation', '')
                        elif 'generate' in chunk:
                            final_response = chunk['generate'].get('generation', '')

                        if chunk.get('end'):
                            break

                except GraphRecursionError:
                    final_response = "정확한 정보가 부족해, 답변을 생성하지 못했습니다. 카드 상품명을 포함해 재질의 해주시기 바랍니다."

            if final_response:
                self.session_manager.append_message(
                    session_id,
                    ChatMessage(role="assistant", content=final_response)
                )

                messages = self.session_manager.get_messages(session_id)
                messages_list = [
                    {"role": msg.role, "content": msg.content}
                    for msg in messages
                ]

                is_append = len(messages_list) > 2
                await self.save_chat_history(session_id, messages_list[-2:], is_append)

                return final_response

        except Exception as e:
            self.logger.error(f"Error in process_message: {str(e)}")
            return "처리 중 오류가 발생했습니다. 다시 시도해 주세요."

def create_chatbot():
    """Create and configure the Gradio interface"""
    app = ChatbotApp()

    # Store session IDs
    sessions = {}

    async def respond(message, history):
        """Gradio chatbot response handler"""
        # Get or create session ID for this conversation
        conversation_id = history[0][0] if history else None
        if conversation_id not in sessions:
            sessions[conversation_id] = str(uuid.uuid4())
        session_id = sessions[conversation_id]

        response = await app.process_message(message, history, session_id)

        accumulated_response = ""
        for char in response:
          accumulated_response += char
          await asyncio.sleep(0.05)
          yield accumulated_response

        yield accumulated_response

    # Create Gradio interface
    chat_interface = gr.ChatInterface(
        respond,
        chatbot=gr.Chatbot(height=600),
        textbox=gr.Textbox(
            placeholder="트래블로그 관련 질의를 입력하세요...",
            container=True,
            submit_btn = True,
            stop_btn = True
        ),
        show_progress = 'full',
        title="Hana Travlog AI ChatBot 💳",
        description="""
        💡 2024.07 기준 트래블로그 카드별 사용약관을 기반으로 답변을 제공합니다. 약관은 변경될 수 있으니 최신 정보를 확인하세요.

        📌 트래블로그 상품명을 입력해주셔야 답변의 성능이 올라갑니다. (예: 트래블로그 PRESTIGE 신용카드의 연회비에 대해 알려줘)
        """,
        theme="soft",
        examples=[
            "트래블로그 PRESTIGE 신용카드의 연회비가 얼마인가요?",
            "미성년자도 트래블로그 발급 받을 수 있어?",
            "해외에서 ATM 이용 시 인출한도는 얼마인가요?"
        ],
        concurrency_limit=20

    )

    return chat_interface

if __name__ == "__main__":
    chat_interface = create_chatbot()
    chat_interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True, #로컬에서는 False로 변경
        debug=True,

    )