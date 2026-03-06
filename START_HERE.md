# 🚀 START HERE: Your RAG Project Guide

Welcome! You've successfully asked all the right questions. Let me answer them clearly so you have zero confusion moving forward.

## 1️⃣ Do I need to set up SQL or a Database?
**NO!** You do **not** need to set up SQL, PostgreSQL, MySQL, or any external database server. 

**Why?**
We are using **ChromaDB**. Unlike traditional databases that require you to run a server in the background, ChromaDB is an *embedded database*. 
- It runs directly inside your Python code.
- It automatically creates a folder called `chroma_db/` in this project directory to save the data.
- It handles all the complex "vector math" and storage by itself locally on your machine.
- It requires **zero setup** from you!

## 2️⃣ Is the project properly set up from my side?
Yes! Because you added the Groq API key to `.env.example`, I have copied it to `.env` for you. The application is fully ready to talk to the AI.

To install everything, you just need to run the setup script I provided:
```bash
./setup.sh
```
*(This will install Python libraries like Streamlit, ChromaDB, and FastAPI).*

## 3️⃣ Exactly Which File Do I Start With? (The Architecture Path)
To understand how RAG works end-to-end, you should read and run the files *in this exact order*. The architecture is a waterfall—each file feeds into the next.

### 🏗️ Complete RAG Architecture Diagram
Here is the visual flow of the entire project. You can show this to anyone to easily explain how RAG works!

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

1. **`data_loader.py`** (The Reader) 
   - *What it does:* Loads PDFs and Text files and extracts the raw text.
2. **`chunker.py`** (The Slicer)
   - *What it does:* Because you can't feed a 500-page book to an AI all at once, this file slices the text into small, overlapping chunks (e.g., 1000 characters).
3. **`embedding_generator.py`** (The Translator)
   - *What it does:* Converts those text chunks into lists of numbers (Vectors/Embeddings) so the computer can understand the *meaning* of the text.
4. **`vector_db.py`** (The Memory Bank)
   - *What it does:* Takes those numbers and saves them in our embedded ChromaDB. When you ask a question, it mathematicaly finds the most similar chunks.
5. **`rag_pipeline.py`** (The Brain / Orchestrator)
   - *What it does:* Connects everything. Takes your question -> Searches the DB -> Pastes the found text into a prompt -> Gives it to the Groq LLM to get an answer.
6. **`app.py` & `streamlit_app.py`** (The Interface)
   - *What it does:* Makes the project look beautiful on a web page or accessible via an API.

**To see this whole process happening in front of your eyes step-by-step, run:**
```bash
python test_pipeline.py
```

---

## 4️⃣ Interview Prep: I Did a Deep Web Search for "Latest RAG Concepts"
You asked me to check the web for the latest RAG topics to ensure you are highly competitive for GenAI interviews. Our current project covers the fundamental **"Naive RAG"** architecture perfectly. 

However, AI is moving fast! After running a deep search on recent 2024-2025 interview trends and research, here are the **Advanced RAG Concepts** you must know the names and definitions of for senior interviews. Once you master this base project, these are your next steps:

### 🌟 Advanced Interview Buzzwords (Know These!)
1. **Agentic RAG:** Instead of just searching once, the AI acts as a "research agent." It breaks your question down, searches the database, reads the result, realizes it needs more info, and searches *again* using different keywords.
2. **GraphRAG:** Instead of just storing vectors, you store data as a "Knowledge Graph" (like a mind map connecting entities: *Steve Jobs* → *founded* → *Apple*). This helps answer questions that require connecting the dots across an entire document.
3. **Adaptive RAG & Self-RAG:** The AI first looks at your question and *decides* if it even needs to search the database. If it's a simple "Hi", it skips the search. If it gets a bad search result, it throws it away and searches again (Self-reflection).
4. **HyDE (Hypothetical Document Embeddings):** When a user asks a vague question, the LLM first writes a "fake, hypothetical" answer. Then, it searches the database using that *fake answer* instead of the question. This vastly improves search accuracy!
5. **Corrective RAG (CRAG):** Adds an evaluator step. After retrieving documents, a separate lightweight AI scores them. If the documents are irrelevant, it falls back to a web search instead of hallucinating.
6. **Multi-modal RAG:** RAG that doesn't just read text, but also retrieves images, sound bites, and graphs to answer your question.

## 🎯 Next Steps For You
1. Open your terminal in this folder.
2. Run `source venv/bin/activate` (if you are on Mac) to use a virtual environment, or just rely on the setup script.
3. Run `./setup.sh` to install all libraries.
4. Run `python test_pipeline.py` to see your RAG pipeline work step-by-step in the terminal.
5. Then, open `data_loader.py` in your code editor and read my comments like a textbook.

You are perfectly situated to master this!
