def create_query_engine(index):
    return index.as_query_engine(streaming=True) #other parameters like llm= also passable

def create_chat_engine(index):
    return index.as_chat_engine()
