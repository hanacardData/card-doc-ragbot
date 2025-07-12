# 🏦 Hana Travlog AI ChatBot 

하나카드 트래블로그 카드 상담을 위한 RAG 기반 AI 챗봇입니다. LangGraph와 LLaMA 3.1을 활용하여 사용자 질문에 대한 답변을 제공합니다.

## 🌟 주요 기능

- **RAG (Retrieval-Augmented Generation)**: 최신 카드 약관 문서를 기반으로 한 답변
- **LangGraph 워크플로우**: 사용자 의도 분류 및 맞춤형 응답 생성
- **다중 검색 엔진**: FAISS와 BM25를 결합한 앙상블 리트리버
- **대화 기록 관리**: 세션별 대화 기록 저장 및 컨텍스트 유지
- **실시간 스트리밍**: 점진적 답변 생성으로 향상된 사용자 경험

## 🚀 설치 및 설정

#### LLM 모델
- **모델**: Meta-Llama-3.1-8B-Instruct-Q6_K_L.gguf
- **경로**: `models/llm_model/`
- **다운로드**: [Hugging Face](https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF)

#### 임베딩 모델
- **모델**: bge-m3
- **경로**: `models/embedding_model/`
- **다운로드**: [Hugging Face](https://huggingface.co/BAAI/bge-m3)


### 디렉토리 구조
```
card-doc-ragbot/
├── app.py                    # 메인 애플리케이션
├── utils/
│   ├── graph_state.py        # LangGraph 상태 관리
│   ├── llm_model_inference.py # LLM 모델 설정
│   ├── vector_db_retrievers.py # 벡터 검색 엔진
│   ├── session_config.py     # 세션 관리
│   ├── logging_config.py     # 로깅 설정
│   └── llm_prompts_templates.py # 프롬프트 템플릿
```


## 🧠 시스템 아키텍처

### LangGraph 워크플로우
1. **의도 분류**: 사용자 질문을 분석하여 적절한 응답 경로 결정
2. **문서 검색**: 관련 문서를 FAISS와 BM25로 검색
3. **문서 평가**: 검색된 문서의 관련성 평가
4. **답변 생성**: RAG 기반 답변 생성
5. **품질 검증**: 환각 현상 및 답변 품질 검증

### 핵심 컴포넌트
- **ChatbotApp**: 메인 애플리케이션 클래스
- **SessionConfigManager**: 세션 및 대화 관리
- **EnsembleRetriever**: FAISS + BM25 하이브리드 검색
- **GraphState**: LangGraph 상태 관리

## 🔧 설정 및 커스터마이징

### 모델 파라미터 조정
`llm_model_inference.py`에서 다음 파라미터들을 조정할 수 있습니다:
- `n_ctx`: 컨텍스트 길이 (기본값: 2048)
- `n_gpu_layers`: GPU 레이어 수 (기본값: 10)
- `max_tokens`: 최대 토큰 수 (기본값: 512)
- `temperature`: 창의성 제어 (기본값: 0.1)

### 검색 설정
`vector_db_retrievers.py`에서 검색 파라미터를 조정:
- `k`: 검색할 문서 수
- `weights`: 앙상블 가중치 [FAISS, BM25]

## 📊 사용 예시

### 기본 질문
```
사용자: "트래블로그 PRESTIGE 신용카드의 연회비가 얼마인가요?"
봇: "트래블로그 PRESTIGE 신용카드의 연회비는 150,000원입니다..."
```

### 후속 질문
```
사용자: "그럼 skypass는?"
봇: "트래블로그 skypass 카드의 연회비는 80,000원입니다..."
```

### 일반 대화
```
사용자: "안녕하세요"
봇: "안녕하세요! 제 이름은 트래블로거입니다. 무엇을 도와드릴까요?"
```


## 📧 문의사항

프로젝트에 대한 질문이나 제안사항이 있으시면 이슈를 생성해주세요.

---

**⚠️ 주의사항**: 이 챗봇은 2024.07 기준 트래블로그 카드 약관을 기반으로 합니다. 최신 정보는 공식 웹사이트에서 확인하세요.
