
import chromadb
import config
from pathlib import Path
from llama_index.core import VectorStoreIndex, Settings, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from data_loader import load_documents
from llm import setup_llm



def create_embed_model():
    return HuggingFaceEmbedding(model_name=config.EMBED_MODEL_NAME)

def create_index(documents):
    embed_model = create_embed_model()
    Settings.embed_model = embed_model 
    # Settings.chunk_size = config.CHUNK_SIZE
    # Settings.llm = llm

    db = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db.get_or_create_collection("quickstart")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    index = VectorStoreIndex.from_documents(
        documents, storage_context=storage_context, embed_model=embed_model
    )
    return index

def load_index():
    embed_model = create_embed_model()
    db2 = chromadb.PersistentClient(path="./chroma_db")
    chroma_collection = db2.get_or_create_collection("quickstart")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    index = VectorStoreIndex.from_vector_store(
        vector_store,
        embed_model=embed_model,
    )
    return index

def setup_index():
    if Path("./chroma_db").exists(): # Check if the index exists on disk
        index = load_index()
    else:
        documents = load_documents()
        index = create_index(documents)
    llm = setup_llm()
    embed_model = create_embed_model()

    # #global context     
    # Settings.chunk_size = config.CHUNK_SIZE
    # Settings.llm = llm
    # Settings.embed_model = embed_model 

    return index    

    