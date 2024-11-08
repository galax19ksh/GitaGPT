## Needs fix 

import streamlit as st
from vector_index import setup_index
from query_engine import create_query_engine

# Initialize LLM and index
#llm = setup_llm()

index = setup_index()

query_engine = create_query_engine(index)

def generate_response(user_query):
    streaming_response = query_engine.query(user_query)
    final = ''
    for text in streaming_response.response_gen:
        final += text
        yield final

# Streamlit interface
st.title('GitaGPT')
st.write("Ask questions and receive wisdom based on the Bhagavad Gita.")

user_query = st.text_input("Ask your question:")

if user_query:
    response = generate_response(user_query)
    st.write(response)