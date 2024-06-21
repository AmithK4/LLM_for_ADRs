from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
from sentence_transformers import SentenceTransformer

class SentenceTransformerWrapper:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_query(self, text: str):
        return self.model.encode(text).tolist()

    def embed_documents(self, texts: list):
        return self.model.encode(texts, convert_to_tensor=False).tolist()

def get_embedding_function():
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    # embeddings = OllamaEmbeddings(model="nomic-embed-text")

    embeddings = SentenceTransformerWrapper("multi-qa-mpnet-base-cos-v1")
    return embeddings
