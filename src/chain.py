from langchain.prompts import PromptTemplate

QA_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant for medical billing.

Here is the conversation so far:
{chat_history}

Use the following context to answer the question:
{context}

Question: {question}
Answer:
""")

conversation_history = []

def chat(user_query, llm, retriever):
    docs = retriever.invoke(user_query)

    # Format chat history for the prompt
    print(conversation_history)
    chat_history_str = "\n".join([f"User: {q}\nAI: {a}" for q, a in conversation_history])

    if not docs:
        answer = llm.invoke(user_query).content
        source_docs = []
    else:
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = QA_PROMPT.format(
            context=context,
            question=user_query,
            chat_history=chat_history_str
        )
        answer = llm.invoke(prompt).content
        source_docs = docs
        print(prompt)

    conversation_history.append((user_query, answer))
    return answer, source_docs
