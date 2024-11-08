from llama_index.core import SimpleDirectoryReader
import config

def load_documents():
    documents = SimpleDirectoryReader(config.DATA_PATH).load_data()
    return documents