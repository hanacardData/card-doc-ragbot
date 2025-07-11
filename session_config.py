### session_config.py
from dataclasses import dataclass, field
from typing import List, Dict, Union, Optional
from langgraph.checkpoint.memory import MemorySaver
import uuid

@dataclass
class ChatMessage:
    """
    채팅 메시지를 위한 데이터 클래스

    Attributes:
        role (str): 메시지 발신자 역할 (예: 'user', 'assistant')
        content (str): 메시지 내용
    """
    role: str
    content: str

@dataclass
class SessionConfig:
    """
    통합된 세션 및 설정 관리를 위한 클래스

    Attributes:
        session_id (str): 세션 고유 식별자
        memory_saver (MemorySaver): LangGraph 메모리 저장소
        messages (List[ChatMessage]): 세션 메시지 목록
        stop_flag (bool): 세션 중단 플래그
        recursion_limit (int): 그래프 재귀 제한
    """
    session_id: str
    memory_saver: MemorySaver
    messages: List[ChatMessage] = field(default_factory=lambda: [
        ChatMessage(role="assistant", content="무엇을 도와드릴까요?")
    ])
    stop_flag: bool = False
    recursion_limit: int = 10

    @classmethod
    def create_new(cls, session_id: str) -> 'SessionConfig':
        """
        새로운 세션 설정을 생성합니다.

        Args:
            session_id (str): 생성할 세션의 고유 식별자

        Returns:
            SessionConfig: 새로 생성된 세션 설정
        """
        return cls(
            session_id=session_id,
            memory_saver=MemorySaver(),
            messages=[ChatMessage(role="assistant", content="무엇을 도와드릴까요?")],
            stop_flag=False,
            recursion_limit=10
        )

    def get_graph_config(self) -> dict:
        """
        LangGraph용 설정 딕셔너리를 반환합니다.

        Returns:
            dict: LangGraph 설정 딕셔너리
        """
        return {
            "configurable": {"thread_id": self.session_id},
            "recursion_limit": self.recursion_limit
        }

class SessionConfigManager:
    """
    Gradio용 세션 관리자
    각 채팅 인스턴스의 상태를 관리합니다.
    """
    def __init__(self):
        """
        세션 관리자를 초기화합니다.
        """
        self.sessions = {}

    def get_or_create_config(self, session_id: Optional[str] = None) -> SessionConfig:
        """
        세션 ID에 해당하는 설정을 가져오거나 새로 생성합니다.

        Args:
            session_id (str): 세션 ID. None인 경우 새로 생성

        Returns:
            SessionConfig: 기존 또는 새로 생성된 세션 설정
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self.sessions:
            self.sessions[session_id] = SessionConfig.create_new(session_id)
        return self.sessions[session_id]

    def get_graph_config(self, session_id: str) -> dict:
        """
        LangGraph용 설정 딕셔너리를 반환합니다.

        Args:
            session_id (str): 세션 ID

        Returns:
            dict: LangGraph 설정 딕셔너리
        """
        config = self.get_or_create_config(session_id)
        return config.get_graph_config()

    def get_messages(self, session_id: str) -> List[ChatMessage]:
        """
        특정 세션의 메시지 목록을 반환합니다.

        Args:
            session_id (str): 세션 ID

        Returns:
            List[ChatMessage]: 세션 메시지 목록
        """
        return self.get_or_create_config(session_id).messages

    def append_message(self, session_id: str, message: Union[ChatMessage, Dict[str, str]]):
        """
        세션에 새 메시지를 추가합니다.

        Args:
            session_id (str): 세션 ID
            message (Union[ChatMessage, Dict[str, str]]): 추가할 메시지

        Raises:
            ValueError: 잘못된 메시지 형식일 경우
        """
        config = self.get_or_create_config(session_id)

        if isinstance(message, dict):
            message = ChatMessage(role=message['role'], content=message['content'])

        config.messages.append(message)

    def clear_session(self, session_id: str):
        """
        특정 세션의 대화 기록을 초기화합니다.

        Args:
            session_id (str): 초기화할 세션 ID
        """
        if session_id in self.sessions:
            self.sessions[session_id] = SessionConfig.create_new(session_id)