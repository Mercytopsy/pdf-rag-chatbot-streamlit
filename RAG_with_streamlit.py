#Import Library
from unstructured.partition.pdf import partition_pdf
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough,RunnableLambda

# from langchain.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from pathlib import Path
from IPython.display import display, HTML
from base64 import b64decode
import chromadb
import tempfile
import shutil
import streamlit as st
import logging
import uuid
import time
import torch
import os
from dotenv import load_dotenv
load_dotenv()


torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

PERSIST_DIRECTORY = os.path.join("data", "vectors")

FILE_PATH = Path("data/hbspapers_48__1.pdf") 


logging.basicConfig(level=logging.INFO)





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


#summarize the data
def summarize_text_and_tables(text, tables):
    logging.info("Ready to summarize data with LLM")
    llm_summary= {}
    prompt_text = """You are an assistant tasked with summarizing text and tables. \
    
                    You are to give a concise summary of the table or text and do nothing else. 
                    Table or text chunk: {element} """
    prompt = ChatPromptTemplate.from_template(prompt_text)
    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")
    summarize_chain = {"element": RunnablePassthrough()}| prompt | model | StrOutputParser()
    # table_summary = summarize_chain.batch(tables, {"max_concurrency": 5})
    # text_summary = summarize_chain.batch(text, {"max_concurrency": 5})
    # llm_summary['text'] = text_summary
    # llm_summary['table'] = table_summary
    #return llm_summary
    logging.info(f"{model} done with summarization")
    return {
        "text": summarize_chain.batch(text, {"max_concurrency": 5}),
        "table": summarize_chain.batch(tables, {"max_concurrency": 5})
    }

    
   


##Multi-vector Retriever
def create_retriever(text, text_summary, table, tables_summary):
    vectorstore = Chroma(collection_name="rag",
                         persist_directory=PERSIST_DIRECTORY,
                         embedding_function=OpenAIEmbeddings() )
    store = InMemoryStore()
    id_key = "doc_id"


    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        docstore=store,
        id_key=id_key
    )

    def add_documents_to_retriver(documents, summaries, retriever):
        if summaries:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            summary_docs = [
                Document(page_content=summary, metadata={id_key: doc_ids[i]})
                for i, summary in enumerate(summaries)
            ]
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, documents)))
    
    #add text and tables to retriever
    add_documents_to_retriver(text, text_summary, retriever)
    add_documents_to_retriver(table, tables_summary, retriever)
  
  
    logging.info(f"Data added to vectorstore ")
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

    rag_chain = {
       "context": retriever | RunnableLambda(parse_retriver_output), "question": RunnablePassthrough(),
        } | RunnablePassthrough().assign(
        response=(
        prompt 
        | model 
        | StrOutputParser()
        )
        )
 
#     rag_chain = (
#     {
#         "context": retriever | RunnableLambda(parse_retriver_output), 
#         "question": RunnablePassthrough()
#     }
#     | prompt
#     | model
#     | StrOutputParser()
# )
    logging.info(f"Completed! ")
    return rag_chain

### extract tables and text

# def pdf_to_retriever(file_path):
#     pdf_elements = load_pdf_data(file_path)
    
#     tables = [element.metadata.text_as_html for element in
#                pdf_elements if 'Table' in str(type(element))]
    
#     text = [element for element in pdf_elements if 
#             'CompositeElement' in str(type(element))]
   
    
#     summaries = summarize_text_and_tables(text, tables)

    
#     retriever = create_retriever(text, summaries['text'], tables, summaries['table'])
    
#     # rag_chain = chat_with_llm(retriever)
    
#     return retriever





# def invoke_chat(message):
#     rag_chain = process_pdf_elements()
#     response = rag_chain.invoke(message)
#     response_placeholder = st.empty()
#     response_placeholder.write(response)
#     return response

def create_vector_db(file_upload):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)  # Ensure the directory exists

    if isinstance(file_upload, str):
        file_path = file_upload  # Already a string path
    else:
        file_path = os.path.join(temp_dir, file_upload.name)
        with open(file_path, "wb") as f:
            f.write(file_upload.getbuffer())
    st.success(f"File saved to {file_path}")

    #send pdf to unstructured to load data
    pdf_elements = load_pdf_data(file_path)
    
    tables = [element.metadata.text_as_html for element in
               pdf_elements if 'Table' in str(type(element))]
    text = [element for element in pdf_elements if 
            'CompositeElement' in str(type(element))]
   
    #summarize text and tables with llm
    summaries = summarize_text_and_tables(text, tables)
    retriever = create_retriever(text, summaries['text'], tables, summaries['table'])
    shutil.rmtree(temp_dir)

    return retriever


def invoke_chat(retriever, message):
    # if file_upload is not None:
        # retriever = create_vector_db(file_upload)   
    rag_chain = chat_with_llm(retriever)
    print(rag_chain)
    response = rag_chain.invoke(message)
    response_placeholder = st.empty()
    response_placeholder.write(response)
    return response


import streamlit as st
import logging
import time

def main():
    st.title("PDF Chat Assistant")
    logging.info("App started")

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None

    # Sidebar for file upload
    file_upload = st.sidebar.file_uploader(
        label="Upload", type=["pdf"],
        accept_multiple_files=False,
        key="pdf_uploader"
    )

    # Process uploaded file only when it's new
    if file_upload:
        if "file_processed" not in st.session_state or st.session_state.get("last_uploaded_file") != file_upload.name:
            st.success("File uploaded successfully! You can now ask your question.")

            with st.spinner("Processing uploaded PDF..."):
                st.session_state["vector_db"] = create_vector_db(file_upload)  
                st.session_state["file_processed"] = True  
                st.session_state["last_uploaded_file"] = file_upload.name  # Track last uploaded file

            st.success("Done processing PDF! You can send in your prompt...") 

        else:
            st.success("PDF already processed. You can ask your questions now.")

    # # Display chat history
    # for message in st.session_state.messages:
    #     with st.chat_message(message["role"]):
    #         st.write(message["content"])

    # Capture user input
    if prompt := st.chat_input("Enter a prompt here...", key="chat_input"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        logging.info(f"User message: {prompt}")

        user_message = {"role": "user", "content": prompt}
        st.session_state.messages.append(user_message)
        logging.info(f"User message: {prompt}")

        # 🔹 **Display the user message immediately**
        with st.chat_message("user"):
            st.write(prompt)

        # Generate response immediately after user input
        if st.session_state["vector_db"] is not None:
            with st.chat_message("assistant"):
                start_time = time.time()
                logging.info("Generating response...")
                with st.spinner("Writing..."):
                    user_message = " ".join([msg["content"] for msg in st.session_state.messages if msg])
                    response_message = invoke_chat(st.session_state["vector_db"], user_message)

                    duration = time.time() - start_time
                    response_msg_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"

                    # Append response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response_msg_with_duration})

                    logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

                    # Force re-render to show new messages
                       


    # if file_upload:
    #     # if st.session_state["vector_db"] is None:
    #     # with st.spinner("Processing uploaded PDF..."):
    #         # st.session_state["vector_db"] = create_vector_db(file_upload)
    #     st.success("File uploaded and processed successfully! You can now ask your question.")


    #     # Prompt for user input and save to chat history
    #     if prompt:= st.chat_input("Your question"):
    #         st.session_state.messages.append({"role": "user", "content": prompt})
        
    #     #Display the user's query
    #     for message in st.session_state.messages:
    #         with st.chat_message(message["role"]):
    #             st.write(message["content"])

    #     # Generate a new response if the last message is not from the assistant
    #     # if st.session_state.messages[-1]["role"] != "assistant":
    #     if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
    #         with st.chat_message("assistant"):
    #             start_time = time.time() #timing the response generation
    #             logging.info("Generating response")
    #             with st.spinner("Writing..."):
                    
    
    #                 message= " ".join([msg["content"] for msg in st.session_state.messages if msg])
    #                 # response_message = invoke_chat(message)
    #                 response_message = invoke_chat(message, file_upload if file_upload else None)
    #                 duration = time.time()-start_time #calculate the duration
    #                 response_msg_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
    #                 st.session_state.messages.append({"role": "assistant", "content": response_msg_with_duration})
    #                 st.write(f"Duration: {duration:.2f} seconds")
    #                 logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")


    #                 # if st.session_state["vector_db"] is not None:
                    
    #                 # response_message = invoke_chat(st.session_state["vector_db"], message)
    #                 # duration = time.time()-start_time #calculate the duration
    #                 # response_msg_with_duration = f"{response_message}\n\nDuration: {duration:.2f} seconds"
    #                 # st.session_state.messages.append({"role": "assistant", "content": response_msg_with_duration})
    #                 # st.write(f"Duration: {duration:.2f} seconds")
    #                 # logging.info(f"Response: {response_message}, Duration: {duration:.2f} s")

    



if __name__ == "__main__":
    main()