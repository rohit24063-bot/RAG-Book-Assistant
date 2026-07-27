# 📚 RAG Book Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content using LangChain, ChromaDB, Hugging Face Embeddings, Gemini, and Streamlit.

## Features

- Upload PDF documents
- Automatic text chunking
- Semantic search using embeddings
- MMR-based retrieval
- AI-powered question answering using Gemini
- Displays source page numbers

## Tech Stack

- Python
- Streamlit
- LangChain
- ChromaDB
- Hugging Face Sentence Transformers
- Google Gemini API

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/RAG-Book-Assistant.git
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```
GOOGLE_API_KEY=YOUR_API_KEY
```

Run the application:

```bash
streamlit run app.py
```