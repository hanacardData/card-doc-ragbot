### llm_prompts_templates.py
from langchain.prompts import PromptTemplate


chat_vs_docs_prompt = PromptTemplate(
    template="""You are a grader determining whether a question should be answered using chat history or document-based information.

Question: {question}
Chat History:
{history}

Consider these specific scenarios carefully:

1. ALWAYS use chat history (score: 'yes') for:
   - Basic greetings and conversation ("안녕", "고마워", "이름이 머야")
   - Questions about previous conversations:
     * "내가 맨 처음 질문한게 뭐지?"
     * "방금 전에 뭐라고 했지?"
     * "아까 답변이 뭐였어?"
   - Meta-questions about the conversation itself
   - Questions about the assistant itself
   - Simple clarifications about previous answers

2. ALWAYS use documents (score: 'no') for:
   - Brand new topics not mentioned in chat history
   - First-time specific product queries (e.g., "Prestige card 연회비는 얼마야?")
   - Requests for factual information not previously discussed

3. Consider the context carefully:
   - For follow-up questions like "그럼 skypass는?", check if it's part of an ongoing topic
   - If the previous context provides enough information, use chat
   - If new information is needed even with context, use documents

Determine if this question can be answered sufficiently using only the chat history.
If yes, we should use chat history. If no, we need to search documents.

Provide the response as a JSON object with a single key 'score' and the value 'yes' or 'no' without any explanation.""",
    input_variables=["question", "history"],
)

chat_type_prompt = PromptTemplate(
    template="""You are a grader determining whether a chat-based question requires additional document context.

Question: {question}
Chat History:
{history}

Evaluate the following scenarios carefully:

1. Questions that need ONLY chat history (score: 'no'):
   - Basic greetings and conversation ("안녕", "고마워", "이름이 머야")
   - Requests to recall previous conversation:
     * "내가 맨 처음 질문한게 뭐지?"
     * "이전에 어떤 내용을 물어봤었지?"
   - Questions about what was just discussed
   - Clarification requests about previous answers

2. Questions that need BOTH chat history AND documents (score: 'yes'):
   - Follow-up questions that need new information:
     * Previous: "Prestige card 연회비는 얼마야?"
     * Current: "그럼 skypass는?" (needs both previous context AND new card info)
   - Comparative questions about previously discussed topics
   - Questions that build upon previous answers but need additional facts

3. Key decision factors:
   - Does the question reference previous conversation AND need new information?
   - Is this a pure recall of previous conversation (chat only)?
   - Does the follow-up question require new factual information?

Determine if this question requires both chat history AND document context to provide a complete answer.
If yes, we need both chat and documents. If no, chat history alone is sufficient.

Provide the response as a JSON object with a single key 'score' and the value 'yes' or 'no' without any explanation.""",
    input_variables=["question", "history"],
)

retrieval_prompt = PromptTemplate(
    template="""You are an evaluator assessing the relevance of retrieved documents to a given question.

Question: {question}
Retrieved document: {document}

Consider the following criteria:
1. Direct relevance to the specific question topic
2. For product-related queries (e.g., "연회비", "카드 혜택"), ensure information is specific and current
3. For comparative questions (e.g., "그럼 skypass는?"), verify the document contains comparable information
4. For follow-up questions, consider both the current question and previous context

Determine if this document is relevant and useful for answering the question.
Return your assessment as a JSON object with a single key 'score' and value 'yes' or 'no' without any explanation.""",
    input_variables=["question", "document"],
)

generate_prompt = PromptTemplate(
    template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are TravelLogger, a helpful assistant for question-answering tasks. Please answer in Korean. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use five sentences maximum and keep the answer concise. Let's think step-by-step.

When dealing with:
1. Product information: Provide clear, specific details
2. Comparative questions: Ensure balanced comparison using provided information
3. Follow-up questions: Maintain context while focusing on new information
4. Tables: Accurately interpret data, never confuse columns and rows
<|eot_id|><|start_header_id|>user<|end_header_id|>
Question: {question}
Context: {context}
Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
input_variables=["question", "context"],
)

chat_generate_prompt = PromptTemplate(
    template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are TravelLogger, a friendly and helpful AI assistant specialized in travel-related conversations. Always respond in Korean. For your first interaction or when greeting users, introduce yourself as "트래블로거" and offer to help.

Guidelines for your responses:
- For greetings or first interactions, always respond with: "안녕하세요! 제 이름은 트래블로거입니다. 무엇을 도와드릴까요?"
- When answering questions about previous conversations, be direct and accurate
- Keep responses concise, using no more than five sentences
- Maintain conversation context and reference relevant previous exchanges when appropriate
- Focus on providing clear, helpful information based on the chat history
<|eot_id|><|start_header_id|>user<|end_header_id|>
Previous conversation:
{history}

Current question: {question}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
    input_variables=["history", "question"],
)

hallucination_prompt = PromptTemplate(
    template="""You are a grader tasked with determining whether an answer is grounded in or supported by a set of factual documents or relevant conversation history.

    Here are the provided factual documents:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the relevant conversation history:
    \n ------- \n
    {history}
    \n ------- \n
    Here is the generated answer: {generation}

    Based on the documents and conversation history, assess whether the generated answer is factually accurate or logically supported by these inputs.

    Provide a binary score 'yes' or 'no' as to whether the answer is grounded in or supported by the documents or conversation history.
    Return your assessment as a JSON object with the key 'score' and a value of 'yes' or 'no' without additional explanation.
    """,
    input_variables=["documents", "generation", "history"],
)

answer_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n
    Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

re_write_prompt = PromptTemplate(
    template="""You are helping to rewrite questions to improve document retrieval results.

Original Question: {question}
Chat History: {history}
Card Type: {card_type}
Product Name: {product_name}

Consider these cases when rewriting:
1. Follow-up questions (e.g., "그럼 skypass는?"):
   - Include relevant context from previous questions
   - Expand to a complete question (e.g., "skypass 카드의 연회비는 얼마인가요?")
   - Use provided card type and product name for specificity
2. Contextual queries:
   - Maintain important context from chat history
   - Create self-contained, searchable questions
   - Include card type and product details when relevant
3. Vague or broad questions:
   - Make more specific while keeping original intent
   - Include key terms for better document matching
   - Reference specific card information when applicable

Rewrite the question to be clear, specific, and self-contained while preserving the original meaning.""",
    input_variables=["question", "history", "card_type", "product_name"]
)

