from langchain.chains import ConversationalRetrievalChain
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

def get_conv_chain(llm, retriever):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
    )

conversation_history = []

def chat(conv_chain, user_query, llm, retriever):
    docs = retriever.invoke(user_query)
    if not docs:
        # No docs, just fallback to general knowledge
        answer = llm.invoke(user_query).content
        source_docs = []
    else:
        # Use retrieved docs explicitly
        context = "\n\n".join([doc.page_content for doc in docs])
        prompt = QA_PROMPT.format(context=context, question=user_query)
        answer = llm.invoke(prompt).content
        source_docs = docs

    conversation_history.append((user_query, answer))
    return answer, source_docs

