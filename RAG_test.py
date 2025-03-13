import os
import uuid
import time
import torch
import logging
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.schema.document import Document
from langchain.storage import InMemoryStore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from unstructured.partition.pdf import partition_pdf

# Load environment variables
load_dotenv()

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)] 

# Configurations
PERSIST_DIRECTORY = "data/vectors"
FILE_PATH = Path("data/hbspapers_48__1.pdf")
logging.basicConfig(level=logging.INFO)


def load_pdf_data():
    """Loads and partitions a PDF file into structured elements."""
    logging.info("Loading PDF data...")

    try:
        elements = partition_pdf(
            filename=FILE_PATH,
            infer_table_structure=True,
            strategy="hi_res",
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
            chunking_strategy="by_title",
            mode="elements",
            max_characters=10000,
            new_after_n_chars=5000,
            combine_text_under_n_chars=2000,
            image_output_dir_path="data/",
        )
        logging.info("PDF loaded successfully.")
        return elements
    except Exception as e:
        logging.error(f"Failed to load PDF: {e}")
        return []


def summarize_text_and_tables(text, tables):
    """Summarizes text and tables using an LLM model."""
    logging.info("Summarizing text and tables...")
    
    prompt_text = "Summarize the following table or text: {element}"
    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")
    summarize_chain = RunnablePassthrough() | ChatPromptTemplate.from_template(prompt_text) | model | StrOutputParser()

    return {
        "text": summarize_chain.batch(text, {"max_concurrency": 5}),
        "table": summarize_chain.batch(tables, {"max_concurrency": 5})
    }


def create_retriever(text, text_summary, tables, table_summary):
    """Creates a multi-vector retriever and adds documents."""
    vectorstore = Chroma(collection_name="rag", persist_directory=PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings())
    store = InMemoryStore()
    retriever = MultiVectorRetriever(vectorstore=vectorstore, docstore=store, id_key="doc_id")

    def add_documents(documents, summaries):
        """Helper function to add documents and summaries to the retriever."""
        if summaries:
            doc_ids = [str(uuid.uuid4()) for _ in documents]
            summary_docs = [Document(page_content=summaries[i], metadata={"doc_id": doc_ids[i]}) for i in range(len(documents))]
            retriever.vectorstore.add_documents(summary_docs)
            retriever.docstore.mset(list(zip(doc_ids, documents)))

    add_documents(text, text_summary)
    add_documents(tables, table_summary)

    logging.info("Retriever setup complete.")
    return retriever


def process_pdf_elements():
    """Processes the PDF elements, summarizes them, and prepares the retriever."""
    pdf_elements = load_pdf_data()
    
    if not pdf_elements:
        return None  # Exit if no data is loaded

    tables = [e.metadata.text_as_html for e in pdf_elements if "Table" in str(type(e))]
    text = [e for e in pdf_elements if "CompositeElement" in str(type(e))]

    summaries = summarize_text_and_tables(text, tables)
    return create_retriever(text, summaries["text"], tables, summaries["table"])


def chat_with_llm(retriever):
    """Creates a RAG-based LLM pipeline."""
    logging.info("Setting up RAG pipeline...")

    prompt_text = """
    You are an AI Assistant. Answer questions based only on the given context.
    Context: {context}
    Question: {question}
    """

    model = ChatOpenAI(temperature=0.6, model="gpt-4o-mini")
    rag_chain = {
        "context": retriever | RunnableLambda(lambda docs: [d.text for d in docs if hasattr(d, "text")]),
        "question": RunnablePassthrough()
    } | ChatPromptTemplate.from_template(prompt_text) | model | StrOutputParser()

    return rag_chain


def main():
    """Streamlit interface for the chatbot."""
    st.title("PDF Chat Assistant")
    logging.info("App started")

    if "retriever" not in st.session_state:
        retriever = process_pdf_elements()
        if retriever is None:
            st.error("Failed to process PDF. Check logs for details.")
            return
        st.session_state.retriever = retriever
        st.session_state.rag_chain = chat_with_llm(retriever)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # User input
    if prompt := st.chat_input("Your question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Process response
    if st.session_state.messages and st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            start_time = time.time()
            with st.spinner("Generating response..."):
                response = st.session_state.rag_chain.invoke(prompt)
                duration = time.time() - start_time
                st.session_state.messages.append({"role": "assistant", "content": f"{response}\n\nDuration: {duration:.2f}s"})
                st.write(f"Duration: {duration:.2f}s")


if __name__ == "__main__":
    main()
