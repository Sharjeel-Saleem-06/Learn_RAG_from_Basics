# 🚀 RAG: From Basics to Mastery

Welcome to **Learn RAG from Basics**! This project is a complete, fully functional, from-scratch implementation of a **Retrieval-Augmented Generation (RAG)** pipeline. It serves as an ultimate guide to learning how Large Language Models (LLMs) can be combined with custom datasets to eliminate hallucinations and create highly accurate AI assistants.

---

## 📖 What is RAG?
Large Language Models (like GPT-4, Llama 3) have an inherent problem: they only know what they were trained on, and they make up information (hallucinate) when asked about private, highly specific, or recent data.

**Retrieval-Augmented Generation (RAG)** solves this. Instead of letting the LLM guess, a RAG system:
1. Takes a user's question and mathematically searches a database of your private documents.
2. Extracts the exact paragraphs that contain the answer.
3. Feeds those paragraphs to the LLM, instructing it to answer the question *only* using the provided text.

---

## 🏗️ The Architecture Flow
This project breaks down RAG into isolated, highly commented Python files so you can understand every single step of the journey. The architecture operates like a waterfall:

```text
+-------------------------------------------------------------+
|               [ PHASE 1: DATA INGESTION ]                   |
|                                                             |
|  [📄 Raw PDFs/TXT]                                          |
|         │                                                   |
|         ▼                                                   |
|  [✂️ 1. data_loader.py + 2. chunker.py]                     |
|         │  (Extracts text & splits into 500-word chunks)    |
|         ▼                                                   |
|  [🧠 3. embedding_generator.py]                             |
|         │  (Converts text chunks into numerical vectors)    |
|         ▼                                                   |
|  [🗄️ 4. vector_db.py (ChromaDB)]                           |
|            (Saves the vectors into a database)              |
+-------------------------------------------------------------+
                              │
                 (Data is saved and ready)
                              │
                              ▼
+-------------------------------------------------------------+
|             [ PHASE 2: QUERY & GENERATION ]                 |
|                                                             |
|  [👤 User Asks a Question]                                  |
|         │                                                   |
|         ▼                                                   |
|  [🧠 Embed Question (embedding_generator.py)]               |
|         │  (Converts question into a numerical vector)      |
|         ▼                                                   |
|  [🔍 Vector Search (vector_db.py)] ◀────── (Reads DB)       |
|         │  (Finds the 3 most relevant chunks to question)   |
|         ▼                                                   |
|  [📝 Prompt Builder (rag_pipeline.py)]                      |
|         │  (Combines the logic: Context + Question)         |
|         ▼                                                   |
|  [🤖 Language Model (Groq / OpenAI)]                        |
|         │  (Reads the context chunks to formulate answer)   |
|         ▼                                                   |
|  [✨ Final Accurate Answer (app.py)]                        |
+-------------------------------------------------------------+
```

---

## 📁 File Structure & Breakdown

| File | Purpose | Key Concept Shown |
|------|---------|-------------------|
| `data_loader.py` | Loads PDF/Text documents | Text parsing and extraction. |
| `chunker.py` | Slices text into overlapping chunks | Respecting token limits and context boundaries. |
| `embedding_generator.py` | Converts text to numbers | Word vectors, Semantic meaning representation. |
| `vector_db.py` | Stores vectors in ChromaDB | Mathematical similarity search (Cosine Distance). |
| `rag_pipeline.py` | The main orchestrator | Prompt Engineering, Context Injection, LLM APIs. |
| `app.py` | FastAPI application | Exposing your RAG model via a REST API. |
| `streamlit_app.py` | Visual Frontend UI | User interaction through chat interfaces. |

---

## ⚙️ How to Run the Project Local

### 1. Requirements
Ensure you have Python 3.9+ installed. You will also need an API key from [Groq](https://console.groq.com/keys) (preferred for extremely fast generation) or OpenAI.

### 2. Setup environment
Run the setup script to install dependencies and configure the environment:
```bash
chmod +x setup.sh
./setup.sh
```

### 3. Add API Keys
Rename `.env.example` to `.env` and paste your Groq API key:
```env
GROQ_API_KEY=your_api_key_here
```

### 4. Test the Pipeline in the Terminal
To see the entire "Ingestion to Query" process print out step-by-step in your terminal:
```bash
python test_pipeline.py
```

### 5. Launch the Web UI
To spin up the beautiful Streamlit chat application and test it live:
```bash
streamlit run streamlit_app.py
```

---

## 📚 Further Learning
Be sure to check out **`START_HERE.md`** and **`ROADMAP.md`** included in this repository. They contain extensive explanations and interview preparation material for understanding advanced techniques such as **HyDE**, **Agentic RAG**, **GraphRAG**, and **Semantic Chunking**.

Built to push the boundaries of LLMs. 🚀
