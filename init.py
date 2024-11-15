import sys
import config
from vector_index import setup_index, create_embed_model, create_index
from llama_index.core import Settings
from query_engine import create_query_engine
from data_loader import load_documents
from llm import setup_llm, download_save_llm


print("Initializing the chatbot...")

# Load documents
print("Loading documents...")
documents = load_documents()
print("Documents loaded.")

# Load embedding model
print("Loading embedding model...")
embed_model = create_embed_model()
print("Done")
# Set up LLM
print("Setting up LLM...")
llm = download_save_llm()
print("LLM setup complete.")

# Global context  (IMPORTANT)   
Settings.chunk_size = config.CHUNK_SIZE
Settings.llm = llm
Settings.embed_model = embed_model

# Set up index
print("Setting up vector index...")
index = create_index(documents)
print("Done")

# Create query engine
print("Creating query engine...")
# query_engine = create_query_engine(index,llm)
query_engine = index.as_query_engine(streaming=True)
print("Done")

#global context  (IMPORTANT)   
Settings.chunk_size = config.CHUNK_SIZE
Settings.llm = llm
Settings.embed_model = embed_model

def generate_response(user_query):      #used for gradio
    streaming_response = query_engine.query(user_query)
    final = ''
    for text in streaming_response.response_gen:
        final += text
        yield final

def generate_response2(query_engine, user_query):   #for terminal output (main.py)
    print("Answer: ")
    response_stream = query_engine.query(user_query)
    final_response = ""
    try:
        for token in response_stream.response_gen:
            final_response += token
            sys.stdout.write("\r" + final_response)
            sys.stdout.flush()
    except Exception as e:
        return f"An error occurred: {e}"        


while True:
    user_query = input("Ask Lord Krishna: ")
    if user_query.lower() == "exit":
        break
    response = generate_response2(query_engine, user_query)
    print(response)
