"""
============================================
FILE 5 OF 8: RAG_PIPELINE.PY — The Heart of RAG
============================================

📝 PURPOSE:
    This is the MAIN ORCHESTRATOR. It connects all the pieces:
    
    1. Takes the user's question
    2. Converts it to an embedding (embedding_generator.py)
    3. Searches for relevant chunks (vector_db.py)
    4. Builds a prompt with the retrieved context
    5. Sends it to an LLM for answer generation
    6. Returns the answer with source citations
    
    This is the complete RAG (Retrieval-Augmented Generation) pipeline!

🎓 CONCEPTS COVERED:
    1. RAG Architecture — The complete retrieval + generation flow
    2. Prompt Templates — How to structure LLM inputs
    3. Context Injection — Feeding retrieved docs into the LLM
    4. LLM API Integration — Calling Groq/OpenAI APIs
    5. Source Citation — Tracking where answers come from
    6. Temperature & Generation Parameters
    7. System Prompts — Instructing the LLM how to behave
    8. Hallucination Prevention — Using retrieved context to ground answers

🔗 HOW IT CONNECTS:
    data_loader.py → chunker.py → embedding_generator.py → vector_db.py
                                                              ↕
                                                        rag_pipeline.py  ← YOU ARE HERE
                                                              ↕
                                                           app.py (API)
    
    This file USES embedding_generator and vector_db for retrieval,
    then calls the LLM API for generation.

💡 THE RAG FLOW IN DETAIL:
    
    User asks: "What are the side effects of aspirin?"
    
    Step 1: EMBED the query
    ─────────────────────
    "What are the side effects of aspirin?"
    → EmbeddingGenerator → [0.12, -0.34, 0.56, ...]
    
    Step 2: RETRIEVE relevant chunks
    ────────────────────────────────
    VectorDB.search([0.12, -0.34, 0.56, ...], n_results=3)
    → Returns:
      Chunk A: "Aspirin may cause stomach irritation, nausea..."
      Chunk B: "Common side effects include dizziness, headache..."
      Chunk C: "Long-term aspirin use can lead to gastrointestinal bleeding..."
    
    Step 3: BUILD the prompt
    ───────────────────────
    "You are a helpful assistant. Answer based on the following context:
    
    CONTEXT:
    [Chunk A text]
    [Chunk B text]  
    [Chunk C text]
    
    QUESTION: What are the side effects of aspirin?
    
    Answer based ONLY on the context above. If the context doesn't 
    contain the answer, say 'I don't have enough information.'"
    
    Step 4: GENERATE the answer
    ──────────────────────────
    Send prompt to LLM (Groq/OpenAI) → Get response
    
    Step 5: RETURN with citations
    ───────────────────────────
    Answer: "According to the provided documents (medical_guide.pdf, pages 12-14),
    the common side effects of aspirin include stomach irritation, nausea, 
    dizziness, and headache. Long-term use may lead to gastrointestinal bleeding."

💡 INTERVIEW QUESTION: "What is hallucination and how does RAG prevent it?"
    
    ANSWER: "Hallucination is when an LLM generates information that sounds 
    correct but is factually wrong or made up. For example, asking about a 
    company's recent earnings might produce plausible-sounding but incorrect 
    numbers.
    
    RAG prevents hallucination by:
    1. GROUNDING: Providing the LLM with actual source documents
    2. INSTRUCTION: Telling the LLM to answer ONLY from the given context
    3. CITATION: Requiring the LLM to reference which documents it used
    4. LIMITING: Using low temperature (0.0-0.1) for factual responses
    
    RAG doesn't completely eliminate hallucination, but significantly reduces 
    it because the LLM has REAL information to reference instead of relying 
    solely on its training data."

============================================
"""

import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from data_loader import Document, load_document, load_directory
from chunker import chunk_documents
from embedding_generator import EmbeddingGenerator
from vector_db import VectorDB
from config import config


# ============================================
# PROMPT TEMPLATES
# ============================================
# These templates tell the LLM HOW to answer.
# They're CRITICAL — a bad prompt = a bad answer.
#
# 💡 PROMPT ENGINEERING CONCEPTS:
#
# 1. SYSTEM PROMPT: Sets the LLM's "personality" and behavior rules
#    "You are a helpful assistant that answers questions accurately."
#
# 2. CONTEXT INJECTION: Placing retrieved documents in the prompt
#    "Based on the following context: [retrieved chunks]"
#
# 3. INSTRUCTION: Telling the LLM what to do and what NOT to do
#    "Answer based ONLY on the provided context."
#    "If the context doesn't contain the answer, say so."
#
# 4. FEW-SHOT: Providing examples of desired output (not used here 
#    but important to know)
#    "Example: Q: What is X? A: X is..."
#
# 💡 INTERVIEW INSIGHT:
# "Few-shot" = giving the model a few examples in the prompt
# "Zero-shot" = no examples, just instructions
# "Chain-of-Thought" = asking model to "think step by step"
# ============================================

# The system prompt that defines LLM behavior
SYSTEM_PROMPT = """You are an expert AI assistant designed for Retrieval-Augmented Generation (RAG).

Your responsibilities:
1. Answer questions EXCLUSIVELY based on the provided context
2. If the context doesn't contain enough information, clearly state: "The provided documents don't contain enough information to answer this question."
3. NEVER make up information or use knowledge outside the provided context
4. Cite the source document when possible (mention the filename/source)
5. Be comprehensive but concise in your responses
6. If multiple context chunks provide relevant information, synthesize them into a coherent answer
7. Use bullet points or numbered lists when listing multiple items
8. If the context is contradictory, mention both perspectives

Remember: You are GROUNDED in the provided documents. Accuracy is more important than completeness."""

# The user prompt template with placeholders
USER_PROMPT_TEMPLATE = """Based on the following context documents, answer the question below.

=== CONTEXT DOCUMENTS ===
{context}
=== END CONTEXT ===

QUESTION: {question}

Please provide a detailed, accurate answer based ONLY on the context above. 
If relevant, mention which source document(s) the information comes from."""


# ============================================
# RAG RESPONSE DATACLASS
# ============================================
# Wraps the complete response with metadata for transparency.
# ============================================

@dataclass
class RAGResponse:
    """
    Complete response from the RAG pipeline.
    
    Contains not just the answer, but also:
    - The retrieved chunks that were used (for verification)
    - Source information (for citation)
    - Timing information (for performance monitoring)
    - The prompt that was sent to the LLM (for debugging)
    
    💡 WHY ALL THIS METADATA?
    In production, you need to:
    - Debug wrong answers (check the retrieved context)
    - Cite sources (tell users where the info came from)
    - Monitor performance (track latency, token usage)
    - Evaluate quality (compare retrieved vs expected chunks)
    """
    answer: str
    sources: List[Dict] = field(default_factory=list)  # Retrieved chunks
    query: str = ""
    prompt: str = ""  # The actual prompt sent to LLM
    retrieval_time: float = 0.0  # Seconds for retrieval
    generation_time: float = 0.0  # Seconds for LLM generation
    total_time: float = 0.0
    model_used: str = ""
    num_chunks_retrieved: int = 0


# ============================================
# RAG PIPELINE CLASS
# ============================================

class RAGPipeline:
    """
    The complete RAG (Retrieval-Augmented Generation) pipeline.
    
    This class orchestrates the entire process:
    1. Ingesting documents (load → chunk → embed → store)
    2. Answering queries (embed query → retrieve → generate)
    
    🎓 ARCHITECTURE OVERVIEW:
    
    INGESTION PHASE (done once per document set):
    ┌──────┐   ┌────────┐   ┌───────────┐   ┌──────────┐
    │ Load │──▶│ Chunk  │──▶│  Embed    │──▶│  Store   │
    │ PDFs │   │ Text   │   │  Vectors  │   │ in VecDB │
    └──────┘   └────────┘   └───────────┘   └──────────┘
    
    QUERY PHASE (done for each user question):
    ┌────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
    │ Embed  │──▶│ Search   │──▶│  Build   │──▶│ Generate │
    │ Query  │   │ VectorDB │   │  Prompt  │   │  Answer  │
    └────────┘   └──────────┘   └──────────┘   └──────────┘
    """
    
    def __init__(self):
        """Initialize all components of the RAG pipeline."""
        print("\n🚀 Initializing RAG Pipeline...")
        print("=" * 50)
        
        # Initialize the embedding generator
        self.embedding_generator = EmbeddingGenerator()
        
        # Initialize the vector database
        self.vector_db = VectorDB()
        
        # Initialize the LLM client
        self.llm_client = self._setup_llm()
        
        print("=" * 50)
        print("✅ RAG Pipeline ready!\n")
    
    def _setup_llm(self):
        """
        Set up the LLM (Large Language Model) client.
        
        🎓 LLM API OVERVIEW:
        
        LLM APIs follow a simple pattern:
        
        REQUEST:
        {
            "model": "llama-3.3-70b-versatile",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant..."},
                {"role": "user", "content": "What is machine learning?"}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        
        RESPONSE:
        {
            "choices": [{
                "message": {
                    "role": "assistant", 
                    "content": "Machine learning is..."
                }
            }],
            "usage": {
                "prompt_tokens": 42,
                "completion_tokens": 128,
                "total_tokens": 170
            }
        }
        
        The "messages" format is called "Chat Completion" API.
        Every LLM API (OpenAI, Groq, Anthropic) uses this same pattern!
        
        💡 MESSAGE ROLES:
        - "system": Instructions for the AI (personality, rules)
        - "user": The human's message
        - "assistant": The AI's response (used for conversation history)
        """
        if config.llm.provider == "groq":
            try:
                from groq import Groq
            except ImportError:
                raise ImportError(
                    "Groq library not installed!\n"
                    "Run: pip install groq\n\n"
                    "Groq provides FREE access to LLMs with ultra-fast inference!"
                )
            
            if not config.groq_api_key:
                raise ValueError(
                    "❌ Groq API key missing!\n"
                    "   1. Go to https://console.groq.com/keys\n"
                    "   2. Create a FREE API key\n"
                    "   3. Add to .env: GROQ_API_KEY=your_key"
                )
            
            client = Groq(api_key=config.groq_api_key)
            print(f"🤖 LLM: Groq ({config.llm.groq_model})")
            return client
        
        elif config.llm.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError("OpenAI library not installed! Run: pip install openai")
            
            client = OpenAI(api_key=config.openai_api_key)
            print(f"🤖 LLM: OpenAI ({config.llm.openai_model})")
            return client
        
        else:
            raise ValueError(f"❌ Unknown LLM provider: {config.llm.provider}")
    
    # ============================================
    # PHASE 1: DOCUMENT INGESTION
    # ============================================
    
    def ingest_file(
        self,
        file_path: str,
        chunk_size: int = None,
        chunk_overlap: int = None,
        chunk_strategy: str = "recursive"
    ) -> Dict:
        """
        Ingest a single file into the RAG system.
        
        This is a complete pipeline: Load → Chunk → Embed → Store
        
        Args:
            file_path: Path to the document file
            chunk_size: Override default chunk size
            chunk_overlap: Override default overlap
            chunk_strategy: "fixed", "recursive", or "sentence"
            
        Returns:
            Dict with ingestion statistics
            
        Example:
            >>> pipeline = RAGPipeline()
            >>> stats = pipeline.ingest_file("research_paper.pdf")
            >>> print(f"Ingested {stats['chunks_created']} chunks")
        """
        print(f"\n📥 Ingesting file: {file_path}")
        print("─" * 50)
        
        start_time = time.time()
        
        # Step 1: Load the document
        print("\n[Step 1/4] Loading document...")
        documents = load_document(file_path)
        
        # Step 2: Chunk the documents
        print("\n[Step 2/4] Chunking text...")
        chunks = chunk_documents(
            documents,
            chunk_size=chunk_size or config.chunk.chunk_size,
            chunk_overlap=chunk_overlap or config.chunk.chunk_overlap,
            strategy=chunk_strategy
        )
        
        # Step 3: Generate embeddings
        print("\n[Step 3/4] Generating embeddings...")
        texts, embeddings, metadatas, ids = self.embedding_generator.embed_documents(chunks)
        
        # Step 4: Store in vector database
        print("\n[Step 4/4] Storing in vector database...")
        self.vector_db.add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        elapsed = time.time() - start_time
        
        stats = {
            "file": file_path,
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
            "embeddings_generated": len(embeddings),
            "time_seconds": round(elapsed, 2),
            "chunk_strategy": chunk_strategy,
            "chunk_size": chunk_size or config.chunk.chunk_size,
        }
        
        print(f"\n✅ Ingestion complete in {elapsed:.1f}s")
        print(f"   Documents: {stats['documents_loaded']}")
        print(f"   Chunks: {stats['chunks_created']}")
        
        return stats
    
    def ingest_directory(self, dir_path: str, **kwargs) -> Dict:
        """
        Ingest all supported files from a directory.
        
        Args:
            dir_path: Path to directory containing documents
            **kwargs: Additional arguments passed to ingest_file
            
        Returns:
            Dict with overall ingestion statistics
        """
        print(f"\n📂 Ingesting directory: {dir_path}")
        
        documents = load_directory(dir_path)
        
        if not documents:
            return {"error": "No supported documents found", "dir": dir_path}
        
        chunks = chunk_documents(
            documents,
            chunk_size=kwargs.get("chunk_size", config.chunk.chunk_size),
            chunk_overlap=kwargs.get("chunk_overlap", config.chunk.chunk_overlap),
            strategy=kwargs.get("chunk_strategy", "recursive")
        )
        
        texts, embeddings, metadatas, ids = self.embedding_generator.embed_documents(chunks)
        
        self.vector_db.add_documents(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        return {
            "directory": dir_path,
            "documents_loaded": len(documents),
            "chunks_created": len(chunks),
        }
    
    # ============================================
    # PHASE 2: QUERY ANSWERING
    # ============================================
    
    def _build_context(self, retrieved_chunks: List[Dict]) -> str:
        """
        Build the context string from retrieved chunks.
        
        This formats the retrieved documents in a clear way
        that the LLM can understand and reference.
        
        🎓 CONTEXT INJECTION:
        This is where we "inject" the retrieved information into
        the LLM's prompt. The quality of this formatting directly
        affects the quality of the answer.
        
        Tips for good context injection:
        1. Clearly separate different chunks
        2. Include source information for each chunk
        3. Number the chunks for easy reference
        4. Don't exceed the model's context window
        """
        if not retrieved_chunks:
            return "No relevant documents found."
        
        context_parts = []
        for i, chunk in enumerate(retrieved_chunks, 1):
            source = chunk['metadata'].get('source', 'Unknown')
            page = chunk['metadata'].get('page', 'N/A')
            score = chunk.get('relevance_score', 0)
            
            context_parts.append(
                f"[Document {i}] Source: {source} | Page: {page} | Relevance: {score:.2f}\n"
                f"{chunk['text']}"
            )
        
        return "\n\n---\n\n".join(context_parts)
    
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """
        Call the LLM API to generate a response.
        
        🎓 THE CHAT COMPLETION API:
        
        All modern LLMs use a "messages" format:
        
        messages = [
            {"role": "system", "content": "Your instructions"},
            {"role": "user", "content": "The user's question + context"}
        ]
        
        The system message is ALWAYS processed first, setting the
        AI's behavior. Then the user message triggers the response.
        
        💡 TEMPERATURE EXPLAINED:
        
        Temperature controls the "randomness" of token selection:
        
        The LLM predicts probabilities for each possible next token:
        "The capital of France is" → 
            "Paris":   90% probability
            "Lyon":     5% probability  
            "love":     0.1% probability
        
        Temperature 0.0: ALWAYS picks the highest probability token
            → "Paris" every time (deterministic)
        
        Temperature 0.5: Mostly picks high-probability tokens
            → Usually "Paris", occasionally "Lyon"
        
        Temperature 1.0: Picks based on actual probabilities
            → "Paris" 90% of the time, "Lyon" 5%
        
        Temperature 2.0: Flattens probabilities, increases randomness
            → Might even pick something weird
        
        For RAG: Use 0.0-0.1 (we want FACTUAL answers!)
        """
        model = (
            config.llm.groq_model 
            if config.llm.provider == "groq" 
            else config.llm.openai_model
        )
        
        try:
            response = self.llm_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=config.llm.temperature,
                max_tokens=config.llm.max_tokens
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"❌ LLM Error: {str(e)}\n\nPlease check your API key and internet connection."
    
    def query(
        self,
        question: str,
        n_results: int = None,
        where: Optional[Dict] = None
    ) -> RAGResponse:
        """
        🎯 MAIN FUNCTION — Ask a question and get an AI-powered answer.
        
        This is the complete RAG flow:
        1. Embed the question
        2. Retrieve relevant chunks
        3. Build the prompt with context
        4. Generate the answer
        5. Return answer + sources
        
        Args:
            question: The user's question in natural language
            n_results: Number of chunks to retrieve (default: config setting)
            where: Optional metadata filter (e.g., {"source": "specific_file.pdf"})
            
        Returns:
            RAGResponse object with answer, sources, and metadata
            
        Example:
            >>> pipeline = RAGPipeline()
            >>> pipeline.ingest_file("medical_guide.pdf")
            >>> response = pipeline.query("What are the side effects of aspirin?")
            >>> print(response.answer)
            >>> for source in response.sources:
            ...     print(f"  Source: {source['metadata']['source']}")
        """
        print(f"\n❓ Query: {question}")
        print("─" * 50)
        
        total_start = time.time()
        
        # ============================================
        # STEP 1: RETRIEVE relevant documents
        # ============================================
        retrieval_start = time.time()
        
        retrieved_chunks = self.vector_db.search_with_text(
            query_text=question,
            embedding_generator=self.embedding_generator,
            n_results=n_results or config.vector_db.n_results,
            where=where
        )
        
        retrieval_time = time.time() - retrieval_start
        
        print(f"🔍 Retrieved {len(retrieved_chunks)} relevant chunks ({retrieval_time:.2f}s)")
        
        for i, chunk in enumerate(retrieved_chunks):
            source = chunk['metadata'].get('source', 'unknown')
            score = chunk.get('relevance_score', 0)
            print(f"   [{i+1}] {source} (relevance: {score:.3f})")
        
        # ============================================
        # STEP 2: BUILD the prompt
        # ============================================
        context = self._build_context(retrieved_chunks)
        
        user_prompt = USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question
        )
        
        # ============================================
        # STEP 3: GENERATE the answer
        # ============================================
        generation_start = time.time()
        
        answer = self._call_llm(SYSTEM_PROMPT, user_prompt)
        
        generation_time = time.time() - generation_start
        total_time = time.time() - total_start
        
        print(f"🤖 Generated answer ({generation_time:.2f}s)")
        
        # ============================================
        # STEP 4: Package the response
        # ============================================
        model_used = (
            config.llm.groq_model 
            if config.llm.provider == "groq" 
            else config.llm.openai_model
        )
        
        response = RAGResponse(
            answer=answer,
            sources=retrieved_chunks,
            query=question,
            prompt=user_prompt,
            retrieval_time=round(retrieval_time, 3),
            generation_time=round(generation_time, 3),
            total_time=round(total_time, 3),
            model_used=model_used,
            num_chunks_retrieved=len(retrieved_chunks)
        )
        
        return response
    
    def chat(self, question: str) -> str:
        """
        Simple chat interface — just returns the answer text.
        
        For a simpler interaction when you don't need all the metadata.
        
        Args:
            question: Your question
            
        Returns:
            The answer as a string
        """
        response = self.query(question)
        return response.answer
    
    def get_db_stats(self) -> Dict:
        """Get current database statistics."""
        return self.vector_db.get_stats()
    
    def reset(self) -> None:
        """Delete all documents and start fresh."""
        self.vector_db.delete_collection()
        print("🔄 Pipeline reset — all documents removed")


# ============================================
# TEST & DEMO SECTION
# ============================================

if __name__ == "__main__":
    from rich import print as rprint
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    rprint(Panel.fit(
        "[bold cyan]🔗 RAG PIPELINE — Test & Demo[/bold cyan]\n\n"
        "This is the COMPLETE RAG system!\n"
        "Load documents → Chunk → Embed → Store → Query → Answer",
        title="File 5 of 8",
        border_style="cyan"
    ))
    
    rprint("\n[yellow]⚠️ This demo requires a valid GROQ_API_KEY in your .env file.[/yellow]")
    rprint("[yellow]   Get a FREE key at: https://console.groq.com/keys[/yellow]\n")
    
    # Check if API key is available
    if not config.groq_api_key or config.groq_api_key == "your_groq_api_key_here":
        rprint(Panel.fit(
            "[bold red]❌ API Key Not Set[/bold red]\n\n"
            "To test the RAG pipeline, you need a Groq API key:\n\n"
            "1. Go to [link]https://console.groq.com/keys[/link]\n"
            "2. Sign up (FREE) and create an API key\n"
            "3. Copy .env.example to .env:\n"
            "   [cyan]cp .env.example .env[/cyan]\n"
            "4. Add your key to .env:\n"
            "   [cyan]GROQ_API_KEY=gsk_your_actual_key_here[/cyan]\n"
            "5. Run this file again:\n"
            "   [cyan]python rag_pipeline.py[/cyan]",
            title="Setup Required",
            border_style="red"
        ))
    else:
        # Full demo with API key available
        import os
        
        # Create sample document
        sample_dir = "./sample_docs"
        os.makedirs(sample_dir, exist_ok=True)
        
        sample_text = """
Retrieval-Augmented Generation (RAG) Comprehensive Guide

Chapter 1: Introduction to RAG

Retrieval-Augmented Generation (RAG) is an AI framework that enhances Large Language Models (LLMs) 
by providing them with relevant context from external knowledge sources before generating a response. 
Instead of relying solely on the model's training data, RAG retrieves actual documents and uses them 
as context for generating accurate, grounded answers.

Key benefits of RAG:
- Reduced hallucination: Answers are grounded in real documents
- Up-to-date information: Documents can be updated without retraining the model
- Source attribution: Users can verify answers against original documents
- Cost-effective: No need for expensive model fine-tuning
- Domain adaptation: Works with any specialized document set

Chapter 2: The RAG Architecture

The RAG architecture consists of two main phases:

Ingestion Phase:
1. Document Loading: Read PDFs, text files, web pages, etc.
2. Text Chunking: Split documents into manageable pieces (typically 500-1500 characters)
3. Embedding Generation: Convert text chunks into vector embeddings
4. Vector Storage: Store embeddings in a vector database for efficient retrieval

Query Phase:
1. Query Embedding: Convert the user's question into a vector
2. Similarity Search: Find the most similar document chunks in the vector database
3. Context Assembly: Combine retrieved chunks into a coherent context
4. Prompt Construction: Build a prompt with the context and user's question
5. LLM Generation: Send the prompt to an LLM for answer generation
6. Response Delivery: Return the answer with source citations

Chapter 3: Chunking Strategies

Chunking is the process of splitting documents into smaller pieces for embedding and retrieval.
The choice of chunking strategy significantly impacts RAG system performance.

Fixed-Size Chunking: Splits text at regular character intervals. Simple but may cut mid-sentence. 
Best for uniform content like code or structured data.

Recursive Character Splitting: Tries to split at natural boundaries (paragraphs, sentences, words). 
This is the most commonly used strategy in production systems.

Semantic Chunking: Uses AI to identify topic boundaries. Most sophisticated but computationally expensive.
Best for complex documents with multiple topics.

Key parameters:
- Chunk size: Typically 500-1500 characters. Smaller chunks = more precise retrieval, larger chunks = more context.
- Chunk overlap: Usually 10-20% of chunk size. Prevents losing information at chunk boundaries.

Chapter 4: Embeddings and Vector Search

Embeddings are dense vector representations of text that capture semantic meaning. 
Similar texts produce similar vectors, enabling semantic search that goes beyond keyword matching.

Popular embedding models:
- all-MiniLM-L6-v2: 384 dimensions, fast, good quality, FREE
- text-embedding-3-small: 1536 dimensions, high quality, paid (OpenAI)
- BGE-large-en-v1.5: 1024 dimensions, state-of-the-art, FREE

Vector similarity is measured using:
- Cosine Similarity: Measures angle between vectors (-1 to 1). Most common for text.
- Euclidean Distance: Measures straight-line distance. Less common for text.
- Dot Product: Fast computation, equals cosine similarity for normalized vectors.

Chapter 5: Vector Databases

Vector databases are specialized databases optimized for storing and querying high-dimensional vectors.
They use approximate nearest neighbor (ANN) algorithms for fast search at scale.

Popular vector databases:
- ChromaDB: Open-source, runs locally, easy setup. Perfect for learning and prototyping.
- Pinecone: Cloud-managed, production-ready, scalable. Free tier available.
- Weaviate: Open-source, supports hybrid search (keyword + semantic).
- FAISS: Facebook's library, extremely fast, not a full database.
- Qdrant: Open-source, high performance, rich filtering capabilities.

Choose based on: scale requirements, budget, deployment model (cloud vs local), and features needed.
        """.strip()
        
        sample_file = os.path.join(sample_dir, "rag_guide.txt")
        with open(sample_file, 'w') as f:
            f.write(sample_text)
        
        # Initialize pipeline
        pipeline = RAGPipeline()
        
        # Ingest the document
        rprint("\n[bold green]--- Step 1: Ingesting Document ---[/bold green]")
        stats = pipeline.ingest_file(sample_file)
        rprint(f"\n📊 Ingestion stats: {stats}")
        
        # Ask questions
        rprint("\n[bold green]--- Step 2: Asking Questions ---[/bold green]")
        
        questions = [
            "What is RAG and why is it useful?",
            "What are the different chunking strategies?",
            "Which vector databases are available?",
        ]
        
        for q in questions:
            response = pipeline.query(q)
            
            rprint(f"\n[bold cyan]Question: {q}[/bold cyan]")
            rprint(f"[dim]Retrieval: {response.retrieval_time}s | "
                   f"Generation: {response.generation_time}s | "
                   f"Total: {response.total_time}s[/dim]")
            rprint(f"\n[green]{response.answer}[/green]")
            rprint(f"\n[dim]Sources used: {response.num_chunks_retrieved} chunks[/dim]")
            for s in response.sources[:2]:
                rprint(f"  📄 {s['metadata'].get('source', 'N/A')} "
                       f"(relevance: {s['relevance_score']:.3f})")
            rprint("─" * 60)
        
        # Clean up
        pipeline.reset()
    
    rprint(Panel.fit(
        "[bold green]✅ RAG Pipeline complete![/bold green]\n\n"
        "[yellow]WHAT YOU LEARNED:[/yellow]\n"
        "• Complete RAG flow: Load → Chunk → Embed → Store → Query → Answer\n"
        "• Prompt templates and context injection\n"
        "• System vs User prompts\n"
        "• LLM API integration (Chat Completion format)\n"
        "• Temperature and generation parameters\n"
        "• Source citation and answer grounding\n"
        "• Hallucination prevention strategies\n\n"
        "[yellow]INTERVIEW GOLD:[/yellow]\n"
        "• 'Explain the RAG architecture' → Two phases: ingestion\n"
        "  (load→chunk→embed→store) and query (embed→retrieve→generate)\n"
        "• 'RAG vs Fine-tuning?' → RAG: cheaper, updatable, citable.\n"
        "  Fine-tuning: better for style/behavior changes.\n"
        "• 'How to prevent hallucination?' → Grounding, low temperature,\n"
        "  system prompts, context-only instructions.\n\n"
        "[yellow]NEXT STEP:[/yellow] app.py — creating the API with FastAPI!",
        title="Summary",
        border_style="green"
    ))
