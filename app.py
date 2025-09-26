import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from src.loaders import load_and_split_docx
from src.embeddings import get_embeddings
from src.vectorstore_utils import save_vectorstore, load_vectorstore
from src.chain import get_conv_chain, chat

load_dotenv()

st.title("Medical Billing Chatbot")

# Load and split document
DOC_PATH = os.path.join("data", "Medical Billing Info Doc.docx")
INDEX_PATH = os.path.join("vectorstore", "Medical_Billing_FAISS.index")

if not os.path.exists(INDEX_PATH):
    split_docs = load_and_split_docx(DOC_PATH)
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)
    save_vectorstore(vector_store, INDEX_PATH)
else:
    embeddings = get_embeddings()
    vector_store = load_vectorstore(INDEX_PATH, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k":5})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=os.getenv("OPENAI_API_KEY"))
conv_chain = get_conv_chain(llm, retriever)

user_query = st.text_input("Ask a question:")
if user_query:
    answer, source_docs = chat(conv_chain, user_query, llm, retriever)
    st.markdown(f"**AI 🤖:** {answer}")
    st.markdown("---")
    st.markdown("**Source Documents:**")
    for i, doc in enumerate(source_docs, 1):
        st.markdown(f"**Document {i}:**\n{doc.page_content[:300]}...")
