"""
============================================
CONFIG.PY — Centralized Configuration
============================================

📝 PURPOSE:
    This file is the "control center" of your RAG application.
    Instead of hardcoding values everywhere, we define them ONCE here.
    
    Think of it like the settings menu of a video game — one place 
    to change all the knobs and dials.

🎓 CONCEPTS COVERED:
    1. Environment Variables — Secure way to handle secrets (API keys)
    2. Configuration Management — Single source of truth for settings
    3. Dataclasses — Python's clean way to group related settings
    4. Separation of Concerns — Config is separate from logic

🔗 CONNECTS TO:
    - Every other file imports from here
    - .env file provides the actual secret values

💡 INTERVIEW CONCEPT — "Separation of Concerns":
    In software engineering, each module should have ONE responsibility.
    config.py's job is ONLY to manage settings. Nothing else.
    This makes your code:
    - Easier to debug (change one place, not 10)
    - Easier to deploy (different .env for dev vs production)
    - More secure (secrets aren't scattered in code)
============================================
"""

import os
from dataclasses import dataclass
from dotenv import load_dotenv

# ============================================
# WHAT IS load_dotenv()?
# ============================================
# Your .env file contains lines like:
#   GROQ_API_KEY=gsk_abc123...
#
# load_dotenv() reads this file and makes these values
# available via os.getenv("GROQ_API_KEY")
#
# WITHOUT load_dotenv: os.getenv("GROQ_API_KEY") → None
# WITH load_dotenv:    os.getenv("GROQ_API_KEY") → "gsk_abc123..."
#
# 💡 INTERVIEW TIP: Always load env vars at the TOP of your config,
# before any other code tries to access them.
# ============================================
load_dotenv()


# ============================================
# WHAT IS A DATACLASS?
# ============================================
# A dataclass is Python's elegant way to create classes that
# primarily hold data. Instead of writing:
#
#   class ChunkConfig:
#       def __init__(self, chunk_size, chunk_overlap):
#           self.chunk_size = chunk_size
#           self.chunk_overlap = chunk_overlap
#
# You just write:
#
#   @dataclass
#   class ChunkConfig:
#       chunk_size: int = 1000
#       chunk_overlap: int = 200
#
# Same result, less boilerplate!
# ============================================


@dataclass
class ChunkConfig:
    """
    Configuration for text chunking.
    
    🎓 CHUNKING CONCEPTS:
    
    chunk_size (int): Maximum number of characters per chunk.
        - Too small (100): Chunks lose context. "The president" — which president?
        - Too large (5000): Too much noise, irrelevant info mixed in
        - Sweet spot (500-1500): Enough context, focused enough for good retrieval
        
    chunk_overlap (int): How many characters overlap between consecutive chunks.
        - Why overlap? Imagine this text:
          "Machine learning is a subset of artificial intelligence. 
           It uses statistical methods to learn from data."
        - If you split at "intelligence." without overlap, Chunk 2 starts with
          "It uses statistical methods..." — what does "It" refer to?
        - With overlap, Chunk 2 starts with "...subset of artificial intelligence.
          It uses statistical methods..." — now "It" has context!
        
    💡 INTERVIEW QUESTION: "How do you choose chunk size and overlap?"
    ANSWER: It depends on:
        - Your embedding model's token limit (most handle 512 tokens ≈ 2000 chars)
        - Your documents (code needs smaller chunks, essays need larger)
        - Your retrieval goal (precise facts = smaller, broader context = larger)
        - Overlap should typically be 10-20% of chunk_size
    """
    chunk_size: int = 1000       # Characters per chunk
    chunk_overlap: int = 200     # Characters of overlap between chunks
    

@dataclass
class EmbeddingConfig:
    """
    Configuration for embedding generation.
    
    🎓 EMBEDDING CONCEPTS:
    
    model_type: "local" or "openai"
        - "local": Uses sentence-transformers (FREE, runs on YOUR computer)
          Model: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
        - "openai": Uses OpenAI's API (PAID, runs on their servers)
          Model: text-embedding-3-small (1536 dimensions, best quality)
    
    💡 WHAT ARE DIMENSIONS?
        An embedding is just a list of numbers: [0.1, -0.3, 0.7, ...]
        "384 dimensions" means the list has 384 numbers.
        More dimensions = more nuanced understanding of text
        BUT also = more storage space and slower search
        
    💡 INTERVIEW QUESTION: "Local vs API embeddings — tradeoffs?"
    ANSWER:
        Local:  ✅ Free, ✅ Private, ✅ No latency, ❌ Lower quality, ❌ Uses your CPU/GPU
        API:    ✅ Higher quality, ✅ No local resources, ❌ Costs money, ❌ Data sent externally
    """
    model_type: str = os.getenv("EMBEDDING_MODEL", "local")
    local_model_name: str = "all-MiniLM-L6-v2"
    openai_model_name: str = "text-embedding-3-small"
    

@dataclass
class LLMConfig:
    """
    Configuration for the Large Language Model.
    
    🎓 LLM CONCEPTS:
    
    provider: Which LLM service to use
        - "groq": FREE tier available! Uses Groq's ultra-fast inference
        - "openai": Paid, but highest quality (GPT-4, etc.)
    
    model_name: The specific model to use
        - "llama-3.3-70b-versatile": Meta's open-source model, FREE on Groq
        - "gpt-4o-mini": OpenAI's cost-effective model
    
    temperature (float): Controls RANDOMNESS of responses (0.0 to 2.0)
        - 0.0: Deterministic. Same input = same output every time.
                Best for: factual Q&A, code generation, RAG
        - 0.7: Creative but coherent. Good for: general chat
        - 1.0+: Very creative, may go off-topic. Good for: brainstorming
        
        💡 INTERVIEW TIP: For RAG, ALWAYS use low temperature (0.0-0.3)
        because you want FACTUAL answers based on retrieved documents,
        not creative hallucinations.
    
    max_tokens (int): Maximum length of the response
        - 1 token ≈ 4 characters in English 
        - 1024 tokens ≈ ~750 words (enough for most answers)
        
    💡 WHAT IS A "CONTEXT WINDOW"?
        The total amount of text (prompt + response) a model can handle.
        - GPT-4o: 128,000 tokens (~96,000 words!)
        - Llama 3.3 70B: 128,000 tokens
        - This is why RAG works: we can fit retrieved chunks + question
          into this window and get relevant answers
    """
    provider: str = os.getenv("LLM_PROVIDER", "groq")
    groq_model: str = "llama-3.3-70b-versatile"
    openai_model: str = "gpt-4o-mini"
    temperature: float = 0.1  # Low = factual, high = creative
    max_tokens: int = 1024
    

@dataclass
class VectorDBConfig:
    """
    Configuration for the vector database.
    
    🎓 VECTOR DATABASE CONCEPTS:
    
    persist_directory: Where ChromaDB stores its data on disk
        - ChromaDB creates files here to persist your vectors
        - Without this, you'd lose all embeddings when you restart!
    
    collection_name: Like a "table" in a regular database
        - You can have multiple collections for different document sets
        - Example: "medical_docs", "legal_docs", "recipes"
    
    n_results: How many similar chunks to retrieve per query
        - Too few (1-2): Might miss relevant information
        - Too many (20+): Floods the LLM with irrelevant chunks
        - Sweet spot (3-5): Good balance of coverage and precision
        
    💡 INTERVIEW QUESTION: "Why use a vector DB instead of a regular DB?"
    ANSWER: 
        Regular DB (PostgreSQL): Stores data in rows/columns, searches by exact match
            → "Find rows WHERE name = 'John'"
        Vector DB (ChromaDB): Stores data as vectors, searches by SIMILARITY
            → "Find vectors CLOSEST to this query vector"
        You can't do semantic similarity search in a regular DB!
        (Note: pgvector extension adds this to PostgreSQL, which is a hybrid approach)
    """
    persist_directory: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    collection_name: str = "rag_documents"
    n_results: int = 5  # Top-K results to retrieve


# ============================================
# MASTER CONFIG — Combines All Configs
# ============================================
# This is the single object you'll import everywhere:
#   from config import config
#   print(config.chunk.chunk_size)  # 1000
#   print(config.llm.temperature)   # 0.1
# ============================================

@dataclass
class RAGConfig:
    """Master configuration that holds all sub-configs."""
    chunk: ChunkConfig = None
    embedding: EmbeddingConfig = None
    llm: LLMConfig = None
    vector_db: VectorDBConfig = None
    
    # API Keys (loaded from .env)
    groq_api_key: str = None
    openai_api_key: str = None
    
    def __post_init__(self):
        """Initialize sub-configs and load API keys."""
        if self.chunk is None:
            self.chunk = ChunkConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.llm is None:
            self.llm = LLMConfig()
        if self.vector_db is None:
            self.vector_db = VectorDBConfig()
        
        # Load API keys from environment
        self.groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
    
    def validate(self):
        """
        Check that required API keys are present.
        
        💡 WHY VALIDATE?
        Better to fail EARLY with a clear error message than to
        get a confusing error 5 minutes into processing 100 PDFs.
        This is called "Fail Fast" principle in software engineering.
        """
        if self.llm.provider == "groq" and not self.groq_api_key:
            raise ValueError(
                "❌ GROQ_API_KEY not found!\n"
                "   1. Go to https://console.groq.com/keys\n"
                "   2. Create a free API key\n"
                "   3. Add it to your .env file: GROQ_API_KEY=your_key_here"
            )
        if self.llm.provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "❌ OPENAI_API_KEY not found!\n"
                "   1. Go to https://platform.openai.com/api-keys\n"
                "   2. Create an API key\n"
                "   3. Add it to your .env file: OPENAI_API_KEY=your_key_here"
            )
        return True


# ============================================
# SINGLETON PATTERN
# ============================================
# We create ONE instance of RAGConfig that the entire app shares.
# This ensures everyone uses the SAME settings.
#
# 💡 INTERVIEW CONCEPT — "Singleton Pattern":
# A design pattern where only ONE instance of a class exists.
# Useful for configs, database connections, loggers.
# ============================================
config = RAGConfig()


# ============================================
# TEST: Run this file directly to verify config
# ============================================
if __name__ == "__main__":
    from rich import print as rprint
    
    rprint("[bold green]✅ Configuration loaded successfully![/bold green]\n")
    rprint(f"[cyan]Chunk Config:[/cyan]")
    rprint(f"  chunk_size: {config.chunk.chunk_size}")
    rprint(f"  chunk_overlap: {config.chunk.chunk_overlap}")
    rprint(f"\n[cyan]Embedding Config:[/cyan]")
    rprint(f"  model_type: {config.embedding.model_type}")
    rprint(f"  local_model: {config.embedding.local_model_name}")
    rprint(f"\n[cyan]LLM Config:[/cyan]")
    rprint(f"  provider: {config.llm.provider}")
    rprint(f"  temperature: {config.llm.temperature}")
    rprint(f"\n[cyan]Vector DB Config:[/cyan]")
    rprint(f"  persist_dir: {config.vector_db.persist_directory}")
    rprint(f"  collection: {config.vector_db.collection_name}")
    rprint(f"  top_k: {config.vector_db.n_results}")
    rprint(f"\n[cyan]API Keys:[/cyan]")
    rprint(f"  Groq: {'✅ Set' if config.groq_api_key else '❌ Missing'}")
    rprint(f"  OpenAI: {'✅ Set' if config.openai_api_key else '⚠️ Not set (optional)'}")
