from langchain_community.embeddings import SentenceTransformerEmbeddings

def get_embeddings(model_name="all-mpnet-base-v2"):
    return SentenceTransformerEmbeddings(model_name=model_name)
