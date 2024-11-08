import gradio as gr
from vector_index import setup_index
from query_engine import create_query_engine


# Check if the index exists on disk
index = setup_index()

query_engine = create_query_engine(index)

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

if __name__ == "__main__":
    iface.launch(share=True)