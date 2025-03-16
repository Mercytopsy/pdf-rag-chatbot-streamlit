# 📌 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for efficient document processing and knowledge retrieval. It extracts text and tables from PDFs using the **Unstructured** library, stores raw PDFs in **Redis**, and indexes extracted embeddings in **PGVector** for semantic search. The system leverages **MultiVector Retriever** for context retrieval before querying an **LLM (GPT model)**.

## 🚀 Features

- **Unstructured Document Processing**: Extracts text and tables from PDFs.  
- **Redis for Raw Storage**: Stores and retrieves raw PDFs efficiently.  
- **PGVector for Vector Storage**: Indexes and retrieves high-dimensional embeddings for similarity search.  
- **MultiVector Retriever**: Optimized for retrieving contextual information from multiple sources.  
- **LLM Integration**: Uses a **GPT model** to generate responses based on retrieved context.  

## 🛠️ Tech Stack

### Programming Language
- Python  

### Libraries
- `unstructured`
- `pgvector`
- `redis-py`
- `langchain`
- `openai`

### Databases
- **Redis**: For raw PDF storage  
- **PostgreSQL + PGVector**: For embeddings storage  

### LLM
- **GPT** (via OpenAI API or local model)
