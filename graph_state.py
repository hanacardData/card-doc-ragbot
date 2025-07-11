### graph_state.py
import json
from typing import List
from langgraph.graph import MessagesState
from utils.vector_db_retrievers import ensemble_retriever
from utils.llm_model_inference import (
    chat_vs_docs_grader, chat_type_grader, retrieval_grader,
    rag_chain, chat_generator, hallucination_grader, answer_grader,
    question_rewriter
)

class GraphState(MessagesState):
    """
    Represents the state of our graph, using SessionConfig's message state.

    Attributes:
        question: the current question
        generation: LLM generation
        documents: list of documents
    """
    question: str
    generation: str
    documents: List[str]


def format_chat_history(messages):
    """
    Format chat history for prompt input.

    Args:
        messages (list): List of message dictionaries

    Returns:
        str: Formatted chat history
    """
    if not messages:  # messages가 None이거나 빈 리스트인 경우
        return ""

    formatted_history = []
    for msg in messages:
        # Check if msg is a ChatMessage object
        if hasattr(msg, 'role') and hasattr(msg, 'content'):
            role = msg.role.capitalize()
            content = msg.content
        # If msg is a dictionary
        elif isinstance(msg, dict):
            role = msg["role"].capitalize()
            content = msg["content"]
        else:
            continue

        formatted_history.append(f"{role}: {content}")
    return "\n".join(formatted_history)

async def classify_intent(state):
    """
    Classify the intent of the question using LLM-based graders.

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated state with intent classification results
    """
    print("---CLASSIFY INTENT---")
    question = state["question"]

    try:
        history = state["messages"]
    except KeyError:
        print("No chat history found, starting fresh conversation")
        history = []

    # First grader: Chat vs Docs
    chat_vs_docs_result = await chat_vs_docs_grader.ainvoke({
        "question": question,
        "history": format_chat_history(history)
    })

    if chat_vs_docs_result["score"] == "yes":  # Should use chat
        # Second grader: Chat only vs Chat+Docs
        chat_type_result = await chat_type_grader.ainvoke({
            "question": question,
            "history": format_chat_history(history)
        })

        if chat_type_result["score"] == "yes":
            intent = "chat_and_docs"
        else:
            intent = "chat_only"
    else:
        intent = "docs_only"

    return {
        "intent": intent,
        "question": question,
        "messages": history
    }


def decide_path(state):
    """
    Decide which path to take based on LLM-graded intent classification.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to process
    """
    intent = state["intent"]

    if intent == "chat_only":
        return "generate_from_history"
    elif intent == "chat_and_docs":
        return "transform_query"
    else:  # docs_only
        return "retrieve"

async def retrieve(state):
    """
    Retrieve documents based on the current question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = await ensemble_retriever.ainvoke(question)

    # Update chat history with the user's question
    state["messages"].append({"role": "user", "content": question})

    return {"documents": documents, "question": question}


async def generate_from_history(state):
    """
    Generate response based only on chat history without document retrieval.

    Args:
        state (dict): The current graph state

    Returns:
        dict: Updated state with generation
    """
    print("---GENERATE FROM HISTORY---")
    question = state["question"]
    history = state["messages"]

    # Generate response using only chat history
    generation = await chat_generator.ainvoke({
        "question": question,
        "history": format_chat_history(history)
    })

    # Update chat history
    state["messages"].append({"role": "user", "content": question})
    state["messages"].append({"role": "assistant", "content": generation})

    return {
        "generation": generation,
        "question": question,
        "messages": state["messages"]
    }

async def generate(state):
    """
    Generate an answer based on the question and retrieved documents.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = await rag_chain.ainvoke({"context": documents, "question": question})

    # Update chat history with the generated answer
    state["messages"].append({"role": "assistant", "content": generation})

    return {"documents": documents, "question": question, "generation": generation}


async def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question.

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Updates documents key with only filtered relevant documents
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")  # print -> return
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    for d in documents:
        score = await retrieval_grader.ainvoke(
            {"question": question, "document": d.page_content}
        )

        # 문자열로 반환된 경우 JSON으로 파싱
        if isinstance(score, str):
            score = json.loads(score)

        grade = score["score"]
        if grade == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            continue

    return {"documents": filtered_docs, "question": question}


async def transform_query(state):
    """
    Re-write the query using the question rewriter and consider chat history.

    Args:
        state (GraphState): The current graph state

    Returns:
        state (GraphState): Updates the question key with a re-phrased question
    """
    print("---TRANSFORM QUERY---")

    # Access question and documents from state
    question = state["question"]
    documents = state["documents"]
    history = state["messages"]

    # 각 문서에서 카드구분과 상품명 추출
    card_type = '정보없음'
    product_name = '정보없음'

    # documents가 있는 경우 첫 번째 문서의 메타데이터 사용
    if documents:
        # 모든 문서의 메타데이터 수집
        card_types = [doc.metadata.get('카드구분', '') for doc in documents if doc.metadata.get('카드구분')]
        product_names = [doc.metadata.get('상품명', '') for doc in documents if doc.metadata.get('상품명')]

        for type_name in card_types:
            if type_name.lower() in question.lower():
                card_type = type_name
                break

        for prod_name in product_names:
            if prod_name.lower() in question.lower():
                product_name = prod_name
                break

    # Re-write the query considering the chat history
    better_question = await question_rewriter.ainvoke({
        "question": question,
        "card_type": card_type,
        "product_name": product_name,
        "history": format_chat_history(history)
    })
    print(f"---{better_question}---")

    # Update chat history with the re-written question
    state["messages"].append({"role": "user", "content": better_question})

    return {
        "documents": documents,
        "question": better_question,
    }


### Edges


async def decide_to_generate(state):
    """
    Determines whether to generate an answer, or re-generate a question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")  # print -> return
    state["question"]
    filtered_documents = state["documents"]

    if not filtered_documents:
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, TRANSFORM QUERY---"
        )
        return "transform_query"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"


async def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document, and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("\n---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    history = state["messages"]

    score = await hallucination_grader.ainvoke(
        {"documents": documents,
         "generation": generation, "history": format_chat_history(history) }
    )

    # Check hallucination
    grade = score["score"]
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = await answer_grader.ainvoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"