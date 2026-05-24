# RAG with LangChain

A retrieval-augmented generation (RAG) system built with LangChain and the Gemini API. This project is part of my learning journey as a self-taught AI developer working toward becoming an AI Agent Developer.

## Background

Before building this, I had already implemented RAG manually from scratch — chunking documents, scoring chunks against the input, and sending the top results to the LLM. That hands-on experience made learning LangChain's abstractions much easier, because I already understood what each piece was doing under the hood. This project is me taking those concepts and building them using Langchain.

## What I learned building this

- How LangChain's built-in classes (PyPDFLoader, RecursiveCharacterTextSplitter, Chroma, as_retriever) map directly to things I had already built manually in a previous project — which made understanding them much faster
- The difference between lexical scoring and semantic search using real embeddings
- How LCEL (LangChain Expression Language) works
- The difference between memory types: Buffer, Window, Token, Summary, and Embedding memory

## How it works

```
ingest.py → loads PDF → splits into chunks → embeds with Gemini → stores in Chroma
rag.py    → loads Chroma → embeds question → retrieves top 3 chunks → Gemini → answer
```

## Stack

- Python
- LangChain
- Gemini API (`langchain-google-genai`)
- Chroma (vector store)
- PyPDF (document loading)

## Setup

1. Clone the repo
2. Install dependencies:
```bash
pip install langchain langchain-google-genai langchain-chroma langchain-text-splitters langchain-community chromadb pypdf python-dotenv
```
3. Create a `.env` file:
```
GOOGLE_API_KEY=your_key_here
```
4. Add a PDF to the project folder and update the path in `ingest.py`
5. Run ingestion:
```bash
python ingest.py
```
6. Ask questions:
```bash
python rag.py
```

## Author

Saad Eddine Bihmid — self-taught AI developer based in Agadir, Morocco  
GitHub: [github.com/Saadbihmid](https://github.com/Saadbihmid)
