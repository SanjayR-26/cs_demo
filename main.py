from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import StorageContext, load_index_from_storage

import os
import streamlit as st

GOOGLE_API_KEY = "AIzaSyA8uO3_RA3jSOMWHIhYj0Lvv6TUKqKqZHQ" 
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

llm = Gemini(model="models/gemini-pro")
embed_model = HuggingFaceEmbedding(model_name ="sentence-transformers/all-mpnet-base-v2")


st.header("JACKGROUPS - Customer support bot")

if "messages" not in st.session_state.keys(): # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there! I'm an AI powered customer service bot. Plese do ask me any questions you have bout jackgroups"}
    ]

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Jackgroup docs – hang tight! This should take 1-2 minutes."):
        if os.path.exists("data/stored"):   
            storage_context = StorageContext.from_defaults(persist_dir="data/stored", embed_model=embed_model)
            index = load_index_from_storage(storage_context)
        else:  
            os.mkdir("data/stored")  
            data = SimpleDirectoryReader(input_dir="data/", recursive=True).load_data()
            index = VectorStoreIndex.from_documents(data, embed_model=embed_model)
            index.storage_context.persist(persist_dir="data/stored")
            
        return index

index = load_data()
chat_engine = index.as_chat_engine(llm=llm, chat_mode="react",system_prompt="You re a customer care executive bot working for Jack groups, and you are here to help customers with their query in a conversational manner. Be friendly and co-operative. Always answer a query with what information you have. And all answers should be descriptive.",  verbose=True)


if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: 
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
