import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.callbacks.base import BaseCallbackHandler
from src.loaders import load_and_split_docx
from src.embeddings import get_embeddings
from src.vectorstore_utils import save_vectorstore, load_vectorstore
from src.chain import chat

load_dotenv()

st.title("Fa'Aghna Chatbot")

# Load and split document
DOC_PATH = os.path.join("data", "Medical Billing Info Doc.docx")
INDEX_PATH = os.path.join("vectorstore", "Medical_Billing_FAISS")

if not os.path.exists(INDEX_PATH):
    split_docs = load_and_split_docx(DOC_PATH)
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    save_vectorstore(vector_store, INDEX_PATH)
else:
    embeddings = get_embeddings()
    vector_store = load_vectorstore(INDEX_PATH, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k":5})

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.streamed_text = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.streamed_text += token
        self.container.markdown(f"**AI 🤖:** {self.streamed_text}")
        
# Enable streaming
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
    streaming=True,
)

user_query = st.text_input("Ask a question:")
if user_query:
    # Create a container for streaming output
    output_container = st.empty()
    llm.callbacks = [StreamHandler(output_container)]

    # Run your chat
    answer, source_docs = chat(user_query, llm, retriever)

    st.markdown("---")
    st.markdown("**Source Documents:**")
    for i, doc in enumerate(source_docs, 1):
        st.markdown(f"**Document {i}:**\n{doc.page_content[:300]}...")
