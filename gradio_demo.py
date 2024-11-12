import gradio as gr
from vector_index import setup_index
from llm import load_llm
from vector_index import create_embed_model
#from llama_index.core import Settings

llm = load_llm()
embed_model = create_embed_model()
index = setup_index()

query_engine = index.as_query_engine(streaming=True)

def generate_response(user_query):
    streaming_response = query_engine.query(user_query)
    final = ''
    for text in streaming_response.response_gen:
        final += text
        yield final

# Create Gradio interface
iface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask Lord Krishna..."),
    outputs="text",
    title="GitaGPT",
    description="Ask questions and receive wisdom based on the Bhagavad Gita."
)

iface.launch(share=True)
