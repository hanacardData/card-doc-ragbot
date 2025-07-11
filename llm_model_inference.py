### llm_model_inference.py
import os
import multiprocessing
from langchain_community.chat_models import ChatLlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from utils.llm_prompts_templates import *
import streamlit as st

# LLM 모델 경로 설정
model_path = os.path.join('models', 'llm_model', 'Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf')
#model_path = os.getenv('MODEL_PATH')

# 콜백 설정
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# LLM Model instance
@st.cache_resource
def load_llm_model():
    return ChatLlamaCpp(
        model_path=model_path,
        n_ctx=2048,
        n_gpu_layers=10,
        n_batch=128,
        max_tokens=512,
        callback_manager=callback_manager,
        n_threads=max(1, multiprocessing.cpu_count() // 2),
        repeat_penalty=1.1,
        temperature=0.1,
        verbose=False,
    )

# 모델 인스턴스 가져오기
llm = load_llm_model()

# 각 프롬프트와 LLM 연결
chat_vs_docs_grader = chat_vs_docs_prompt | llm | JsonOutputParser()
chat_type_grader = chat_type_prompt | llm | JsonOutputParser()
retrieval_grader = retrieval_prompt | llm | StrOutputParser()
rag_chain = generate_prompt | llm | StrOutputParser()
chat_generator = chat_generate_prompt | llm | StrOutputParser()
hallucination_grader = hallucination_prompt | llm | JsonOutputParser()
answer_grader = answer_prompt | llm | JsonOutputParser()
question_rewriter = re_write_prompt | llm | StrOutputParser()
