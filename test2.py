#Import Library
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda

from langchain_postgres.vectorstores import PGVector
from database import COLLECTION_NAME, CONNECTION_STRING
from langchain_community.utilities.redis import get_client
from langchain_community.storage import RedisStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from pathlib import Path
from IPython.display import display, HTML
from base64 import b64decode
import hashlib
import chromadb
import tempfile
import shutil
import streamlit as st
import logging
import uuid
import json
import time
import torch
import redis
import os
from dotenv import load_dotenv
load_dotenv()


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

PERSIST_DIRECTORY = os.path.join("data", "vectors")

FILE_PATH = Path("data/hbspapers_48__1.pdf") 

# FILE_PATH = Path("data/layout-parser-paper.pdf") 



logging.basicConfig(level=logging.INFO)

client = redis.Redis(host="localhost", port=6379, db=0)




###Data Loading
def load_pdf_data(file_path):
    logging.info(f"Data ready to be partitioned and loaded ")
    raw_pdf_elements = partition_pdf(
        filename=file_path,
      
        infer_table_structure=True,
        strategy = "hi_res",
        
        extract_image_block_types = ["Image"],
        extract_image_block_to_payload  = True,

        chunking_strategy="by_title",     
        mode='elements',
        max_characters=10000,
        new_after_n_chars=5000,
        combine_text_under_n_chars=2000,
        image_output_dir_path="data/",
    )
    logging.info(f"Pdf data finish loading, chunks now available!")
    return raw_pdf_elements


def get_pdf_hash(pdf_path):
    """Generate a SHA-256 hash of the PDF file content."""
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()
    return hashlib.sha256(pdf_bytes).hexdigest()


#summarize the data
def summarize_text_and_tables(text, tables):
    logging.info("Ready to summarize data with LLM")
    prompt_text = """You are an assistant tasked with summarizing text and tables. \
    
                    You are to give a concise summary of the table or text and do nothing else. 
                    Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")
    summarize_chain = {"element": RunnablePassthrough()}| prompt | model | StrOutputParser()
    logging.info(f"{model} done with summarization")
    return {
        "text": summarize_chain.batch(text, {"max_concurrency": 5}),
        "table": summarize_chain.batch(tables, {"max_concurrency": 5})
    }
  

def initialize_retriever():

    store = RedisStore(client=client)
    id_key = "doc_id"
    vectorstore = PGVector(
            embeddings=OpenAIEmbeddings(),
            collection_name=COLLECTION_NAME,
            connection=CONNECTION_STRING,
            use_jsonb=True,
            )
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")
    return retriever


###Multivector Retriever

def store_docs_in_retriever(text, text_summary, table, table_summary, retriever):
    """Store text and table documents along with their summaries in the retriever."""

    def add_documents_to_retriever(documents, summaries, retriever, id_key = "doc_id"):
        """Helper function to add documents and their summaries to the retriever."""
        if not summaries:
            return None, []

        doc_ids = [str(uuid.uuid4()) for _ in documents]
        summary_docs = [
            Document(page_content=summary, metadata={id_key: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]

        retriever.vectorstore.add_documents(summary_docs, ids=doc_ids)
        retriever.docstore.mset(list(zip(doc_ids, documents)))     

# Add text, table, and image summaries to the retriever
    add_documents_to_retriever(text, text_summary, retriever)
    add_documents_to_retriever(table, table_summary, retriever)
    return retriever



###RAG pipeline
def parse_retriver_output(data):
    parsed_elements = []
    for element in data:
        if 'CompositeElement' in str(type(element)):
            parsed_elements.append(element.text)
        else:
            parsed_elements.append(element)
            
    return parsed_elements


###Chat with LLM

def chat_with_llm(retriever):

    logging.info(f"Context ready to send to LLM ")
    prompt_text = """
                You are an AI Assistant tasked with understanding detailed
                information from text and tables. You are to answer the question based on the 
                context provided to you. You must not go beyond the context given to you.
                
                Context:
                {context}

                Question:
                {question}
                """

    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")
 
    rag_chain = ({
       "context": retriever | RunnableLambda(parse_retriver_output), "question": RunnablePassthrough(),
        } 
        | prompt 
        | model 
        | StrOutputParser()
        )
    # rag_chain = {
    #    "context": retriever | RunnableLambda(parse_retriver_output), "question": RunnablePassthrough(),
    #     } | RunnablePassthrough().assign(
    #     response=(
    #     prompt 
    #     | model 
    #     | StrOutputParser()
    #     )
    #     )
        
    logging.info(f"Completed! ")

    return rag_chain

def _get_file_path(file_upload):

    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists

    if isinstance(file_upload, str):
        file_path = file_upload  # Already a string path
    else:
        file_path = os.path.join(temp_dir, file_upload.name)
        with open(file_path, "wb") as f:
            f.write(file_upload.getbuffer())
            st.success(f"File saved to {file_path}")
        return file_path
    

### process PDF
def process_pdf(file_upload):
    print('Processing PDF hash info...')
    
    file_path =  _get_file_path(file_upload)
    pdf_hash = get_pdf_hash(file_path)

    retriever = initialize_retriever()
    existing = client.exists(f"pdf:{pdf_hash}")
    print(f"Checking Redis for hash {pdf_hash}: {'Exists' if existing else 'Not found'}")

    if existing:
        print(f"PDF already exists with hash {pdf_hash}. Skipping upload.")
        return retriever

    print(f"New PDF detected. Processing... {pdf_hash}")
 

    pdf_elements = load_pdf_data(file_path)
    
    tables = [element.metadata.text_as_html for element in
               pdf_elements if 'Table' in str(type(element))]
    
    text = [element.text for element in pdf_elements if 
            'CompositeElement' in str(type(element))]
   
    summaries = summarize_text_and_tables(text, tables)
    enriched_retriever = store_docs_in_retriever(text, summaries['text'], tables,  summaries['table'], retriever)
    
    # Store the PDF hash in Redis
    client.set(f"pdf:{pdf_hash}", json.dumps({"text": "PDF processed"}))  

    # Debug: Check if Redis stored the key
    stored = client.exists(f"pdf:{pdf_hash}")
    print(f"Stored PDF hash in Redis: {'Success' if stored else 'Failed'}")
    return enriched_retriever





def invoke_chat(file_upload, message):

    retriever =process_pdf(file_upload)
    rag_chain = chat_with_llm(retriever)
    response = rag_chain.invoke(message)
    response_placeholder = st.empty()
    response_placeholder.write(response)
    return response




def main():
    # streamlit_initialize()
    st.title("PDF Chat Assistant ")
    logging.info("App started")

    if 'messages' not in st.session_state:
        st.session_state.messages = []

 
    file_upload = st.sidebar.file_uploader(
    label="Upload", type=["pdf"], 
    accept_multiple_files=False,
    key="pdf_uploader"
    )

    if file_upload:
        st.success("File uploaded successfully! Processing...")
  
        
        st.success("File processed successfully! You can now ask your question.")

    # Prompt for user input
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Generate response if last message is not from assistant
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            start_time = time.time()
            logging.info("Generating response...")
            with st.spinner("Writing..."):
                user_message = " ".join([msg["content"] for msg in st.session_state.messages if msg])
                
                response_message = invoke_chat(file_upload, user_message)
                #response_message = invoke_chat(FILE_PATH,user_message)

                duration = time.time() - start_time
                response_msg_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"

                st.session_state.messages.append({"role": "assistant", "content": response_msg_with_duration})
                st.write(f"Duration: {duration:.2f} seconds")
                logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")


    



if __name__ == "__main__":
    main()