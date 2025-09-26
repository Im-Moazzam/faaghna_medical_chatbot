from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

def get_chat_prompt():
    system_prompt = SystemMessagePromptTemplate.from_template("""
You are an assistant that primarily answers questions based on the provided documents.

Instructions:
- Always check if the provided context has directly relevant or related information.
- If the context has *partially relevant* info, use it and expand with your general knowledge.
- If the context has nothing useful at all, then say:
  "I don’t know. The document does not provide this info."
- Be concise, factual, and clear.

Context:
{context}
""")

    human_prompt = HumanMessagePromptTemplate.from_template("{question}")
    return ChatPromptTemplate.from_messages([system_prompt, human_prompt])
