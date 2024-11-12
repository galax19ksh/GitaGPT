import sys
import config
from vector_index import setup_index, create_embed_model
from llama_index.core import Settings
from llm import load_llm

def initialize():
    
    # Load embedding model
    print("Loading embedding model...")
    embed_model = create_embed_model()
    print("Done")
    # Set up LLM
    print("Setting up LLM...")
    llm = load_llm()
    print("LLM setup complete.")
    
    # Set up index
    print("Setting up vector index...")
    index = setup_index()
    print("Done")

    #global context  (IMPORTANT)   
    Settings.chunk_size = config.CHUNK_SIZE
    Settings.llm = llm
    Settings.embed_model = embed_model
    
    # Create query engine
    print("Creating query engine...")
    query_engine = index.as_query_engine(streaming=True)
    print("Done")
    
    
           

    return query_engine        
    
if __name__ == "__main__":
    query_engine = initialize()

    def generate_response(query_engine, user_query):  
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
        response = generate_response(query_engine, user_query)
        print(response)
