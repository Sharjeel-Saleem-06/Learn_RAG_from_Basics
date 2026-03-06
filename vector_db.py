"""
============================================
FILE 4 OF 8: VECTOR_DB.PY — Vector Storage & Retrieval
============================================

📝 PURPOSE:
    We've loaded documents, chunked them, and created embeddings.
    Now we need to STORE these embeddings and SEARCH through them.
    
    That's what a Vector Database does:
    1. STORE: Save embeddings + metadata + original text
    2. SEARCH: Given a query embedding, find the most similar stored embeddings
    
    We're using ChromaDB — an open-source vector database that runs LOCALLY.
    No cloud setup, no account creation, no cost!

🎓 CONCEPTS COVERED:
    1. Vector Database — What it is and why regular databases can't do this
    2. Similarity Search — Finding nearest neighbors in vector space
    3. Collections — Like "tables" in a regular database
    4. Persistence — Saving vectors to disk so they survive restarts
    5. CRUD Operations — Create, Read, Update, Delete for vectors
    6. Distance Metrics — How ChromaDB measures "closeness"
    7. ANN (Approximate Nearest Neighbor) — How search stays fast at scale

🔗 HOW IT CONNECTS:
    data_loader.py  →  chunker.py  →  embedding_generator.py  →  vector_db.py
                                                                  ^^^^^^^^^^^^
                                                                  YOU ARE HERE
    
    Input: Embeddings + Text + Metadata from embedding_generator.py
    Output: Top-K most relevant chunks for a given query
    
    Next: rag_pipeline.py will use this to retrieve context for the LLM

💡 DEEP DIVE — WHY CAN'T WE USE A REGULAR DATABASE?

    Regular Database (e.g., PostgreSQL):
    ────────────────────────────────────
    Table: documents
    | id | text                          | category |
    |----|-------------------------------|----------|
    | 1  | "Machine learning is..."      | AI       |
    | 2  | "Italian cooking recipes..."  | Food     |
    
    Query: SELECT * FROM documents WHERE text LIKE '%AI%'
    → Only finds rows that LITERALLY contain "AI"
    → MISSES "Machine learning is..." even though it's about AI!
    
    Vector Database (e.g., ChromaDB):
    ─────────────────────────────────
    Collection: documents
    | id | embedding            | text                          |
    |----|---------------------|-------------------------------|
    | 1  | [0.4, 0.8, -0.2]   | "Machine learning is..."      |
    | 2  | [-0.8, 0.1, 0.9]   | "Italian cooking recipes..."  |
    
    Query: Find vectors closest to embedding("AI technology")
    → Query embedding: [0.39, 0.81, -0.19]
    → Closest match: [0.4, 0.8, -0.2] = "Machine learning is..."
    → ✅ Found! Even though the text never mentions "AI"
    
    THIS is the power of semantic search!

💡 INTERVIEW QUESTION: "How does vector search scale?"
    
    ANSWER: "Brute force comparison against all vectors is O(n) — too slow
    for millions of vectors. Vector databases use Approximate Nearest Neighbor 
    (ANN) algorithms like:
    
    1. HNSW (Hierarchical Navigable Small World) — ChromaDB uses this!
       Think of it as a multi-layer graph where each layer has fewer nodes.
       Search starts at the top layer and navigates down.
       
    2. IVF (Inverted File Index) — FAISS uses this
       Clusters vectors into groups, searches only the nearest clusters.
       
    3. Product Quantization — Compresses vectors for faster comparison.
    
    These are 'approximate' because they trade a tiny bit of accuracy
    for MASSIVE speed gains. Instead of checking 1M vectors, you check ~1000."

============================================
"""

import os
from typing import List, Optional, Dict, Any
from data_loader import Document
from config import config


class VectorDB:
    """
    Vector Database wrapper using ChromaDB.
    
    ChromaDB is an open-source embedding database designed for
    AI applications. It stores, searches, and manages vector embeddings.
    
    🎓 CHROMADB ARCHITECTURE:
    
    ChromaDB stores data in "collections" (like tables):
    
    Collection: "rag_documents"
    ┌────────────┬──────────────────┬──────────────────────────┬───────────────┐
    │ ID         │ Embedding        │ Document (text)          │ Metadata      │
    ├────────────┼──────────────────┼──────────────────────────┼───────────────┤
    │ chunk_001  │ [0.1, -0.3, ...] │ "ML is a subset of..."  │ {source: ...} │
    │ chunk_002  │ [0.4,  0.2, ...] │ "Deep learning uses..." │ {source: ...} │
    │ chunk_003  │ [-0.8, 0.9, ...] │ "Italian food is..."    │ {source: ...} │
    └────────────┴──────────────────┴──────────────────────────┴───────────────┘
    
    When you search:
    1. Your query is embedded: "What is machine learning?" → [0.12, -0.28, ...]
    2. ChromaDB finds the N closest embeddings (using HNSW algorithm)
    3. Returns the matching documents + metadata + distances
    
    💡 PERSISTENCE:
    By default, ChromaDB stores everything in memory (lost on restart).
    We use PersistentClient to save to disk — your data survives restarts!
    """
    
    def __init__(
        self, 
        persist_directory: str = None,
        collection_name: str = None
    ):
        """
        Initialize the vector database.
        
        Args:
            persist_directory: Where to store database files on disk
            collection_name: Name of the collection (like a table name)
        """
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "ChromaDB is not installed!\n"
                "Run: pip install chromadb\n\n"
                "ChromaDB is a vector database that stores and searches\n"
                "embeddings. It runs locally — no cloud setup needed!"
            )
        
        self.persist_directory = persist_directory or config.vector_db.persist_directory
        self.collection_name = collection_name or config.vector_db.collection_name
        
        # ============================================
        # CREATE THE CHROMADB CLIENT
        # ============================================
        # PersistentClient saves data to disk.
        # Your vectors survive even if you close Python!
        #
        # Alternative: chromadb.Client() — ephemeral (in-memory only)
        # Use for testing, not production.
        # ============================================
        
        # Create directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=self.persist_directory
        )
        
        # ============================================
        # GET OR CREATE COLLECTION
        # ============================================
        # get_or_create_collection: 
        # - If collection exists → returns it
        # - If not → creates a new one
        #
        # distance_fn: How ChromaDB measures similarity
        # - "cosine": Cosine distance (1 - cosine_similarity)
        #   0 = identical, 2 = opposite
        # - "l2": Euclidean distance
        # - "ip": Inner product (dot product)
        #
        # We use cosine because it works best for text embeddings.
        # ============================================
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine distance
        )
        
        print(f"🗄️  Vector DB initialized:")
        print(f"   Directory: {self.persist_directory}")
        print(f"   Collection: {self.collection_name}")
        print(f"   Existing documents: {self.collection.count()}")
    
    def add_documents(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]]
    ) -> None:
        """
        Add documents with their embeddings to the vector database.
        
        This is the WRITE operation — storing your processed chunks.
        
        Args:
            ids: Unique identifier for each document
            embeddings: Vector embeddings (from embedding_generator)
            documents: Original text of each chunk
            metadatas: Metadata for each chunk (source, page, etc.)
            
        Example:
            >>> db.add_documents(
            ...     ids=["chunk_1", "chunk_2"],
            ...     embeddings=[[0.1, 0.2, ...], [0.3, 0.4, ...]],
            ...     documents=["ML is...", "DL uses..."],
            ...     metadatas=[{"source": "ml.pdf"}, {"source": "dl.pdf"}]
            ... )
        
        💡 IMPORTANT:
        - IDs must be unique! Adding a document with an existing ID 
          will raise an error. Use upsert() to update existing docs.
        - Metadata values must be strings, ints, floats, or bools.
          No lists, dicts, or None values!
        """
        if not ids:
            print("⚠️ No documents to add!")
            return
        
        # ============================================
        # CLEAN METADATA
        # ============================================
        # ChromaDB is strict about metadata types.
        # We need to ensure all values are basic types.
        # This is a common gotcha that trips up beginners!
        # ============================================
        clean_metadatas = []
        for meta in metadatas:
            clean_meta = {}
            for key, value in meta.items():
                # ChromaDB only accepts str, int, float, bool
                if isinstance(value, (str, int, float, bool)):
                    clean_meta[key] = value
                elif value is None:
                    clean_meta[key] = "none"
                else:
                    clean_meta[key] = str(value)
            clean_metadatas.append(clean_meta)
        
        # ============================================
        # ADD TO COLLECTION
        # ============================================
        # ChromaDB's add() is like INSERT in SQL.
        # It stores the embedding, document text, and metadata together.
        #
        # Internally, ChromaDB:
        # 1. Adds the embedding to its HNSW index (for fast search)
        # 2. Stores the document text (for retrieval)
        # 3. Stores metadata (for filtering)
        # ============================================
        
        # Add in batches to avoid memory issues with large datasets
        batch_size = 100
        total_added = 0
        
        for i in range(0, len(ids), batch_size):
            batch_ids = ids[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_documents = documents[i:i + batch_size]
            batch_metadatas = clean_metadatas[i:i + batch_size]
            
            try:
                self.collection.add(
                    ids=batch_ids,
                    embeddings=batch_embeddings,
                    documents=batch_documents,
                    metadatas=batch_metadatas
                )
                total_added += len(batch_ids)
            except Exception as e:
                # If IDs already exist, try upserting instead
                if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                    self.collection.upsert(
                        ids=batch_ids,
                        embeddings=batch_embeddings,
                        documents=batch_documents,
                        metadatas=batch_metadatas
                    )
                    total_added += len(batch_ids)
                else:
                    raise e
        
        print(f"   ✅ Added {total_added} documents to vector DB")
        print(f"   📊 Total documents in collection: {self.collection.count()}")
    
    def search(
        self,
        query_embedding: List[float],
        n_results: int = None,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ) -> Dict:
        """
        🎯 CORE FUNCTION — Search for similar documents.
        
        This is the RETRIEVAL part of RAG!
        Given a query embedding, find the N most similar chunks.
        
        Args:
            query_embedding: The vector representation of the user's question
            n_results: How many results to return (default: config setting)
            where: Optional metadata filter (e.g., {"source": "report.pdf"})
            where_document: Optional document text filter
            
        Returns:
            Dict with keys:
                - 'ids': List of matching document IDs
                - 'documents': List of matching text chunks
                - 'metadatas': List of metadata for each match
                - 'distances': List of distances (lower = more similar)
                
        Example:
            >>> results = db.search(
            ...     query_embedding=[0.1, -0.3, ...],
            ...     n_results=3
            ... )
            >>> for doc, score in zip(results['documents'][0], results['distances'][0]):
            ...     print(f"Score: {score:.4f} | {doc[:100]}...")
                
        🎓 HOW THE SEARCH WORKS INTERNALLY:
        
        1. ChromaDB takes your query embedding
        2. Uses HNSW algorithm to find approximate nearest neighbors
        3. Calculates exact cosine distance for the candidates
        4. Returns top-N results sorted by distance (ascending)
        
        HNSW (Hierarchical Navigable Small World) explained simply:
        ──────────────────────────────────────────────────────────
        Imagine a city map with highways, main roads, and alleys:
        
        Layer 3 (highways):    [A] ────────────── [B]
                                ↓                   ↓
        Layer 2 (main roads):  [A] ── [C] ── [D] ─ [B]
                                ↓      ↓      ↓      ↓
        Layer 1 (streets):     [A]-[E]-[C]-[F]-[D]-[G]-[B]
                                ↓  ↓  ↓  ↓  ↓  ↓  ↓  ↓
        Layer 0 (all points):  [A][E][H][C][I][F][J][D][K][G][L][B]
        
        To find a neighbor of query Q:
        1. Start at top layer (highways) — quickly narrow the area
        2. Move down to main roads — get closer
        3. Move down to streets — almost there
        4. Search locally at bottom layer — found it!
        
        This is WAY faster than checking every single point!
        For 1M vectors: brute force = 1M comparisons, HNSW ≈ 1000 comparisons
        
        💡 INTERVIEW: "What is the time complexity of HNSW?"
        ANSWER: O(log n) for search — logarithmic, not linear!
        """
        n = n_results or config.vector_db.n_results
        
        # Build query parameters
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(n, self.collection.count()),  # Can't return more than we have
            "include": ["documents", "metadatas", "distances"]
        }
        
        # Add optional filters
        if where:
            query_params["where"] = where
        if where_document:
            query_params["where_document"] = where_document
        
        # Perform the search!
        results = self.collection.query(**query_params)
        
        return results
    
    def search_with_text(
        self,
        query_text: str,
        embedding_generator,
        n_results: int = None,
        where: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Convenience method: search using text instead of a pre-computed embedding.
        
        This method:
        1. Embeds the query text using the embedding generator
        2. Searches the vector database
        3. Returns results in a cleaner format
        
        Args:
            query_text: The user's question in plain text
            embedding_generator: An EmbeddingGenerator instance
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            List of dicts, each containing:
                - 'text': The matching chunk text
                - 'metadata': The chunk's metadata
                - 'distance': How far from the query (lower = better)
                - 'relevance_score': Converted to 0-1 score (higher = better)
        """
        # Step 1: Embed the query
        query_embedding = embedding_generator.embed_text(query_text)
        
        # Step 2: Search
        raw_results = self.search(
            query_embedding=query_embedding,
            n_results=n_results,
            where=where
        )
        
        # Step 3: Format results nicely
        formatted_results = []
        
        if raw_results['documents'] and raw_results['documents'][0]:
            for i in range(len(raw_results['documents'][0])):
                result = {
                    'text': raw_results['documents'][0][i],
                    'metadata': raw_results['metadatas'][0][i],
                    'distance': raw_results['distances'][0][i],
                    # Convert cosine distance to relevance score
                    # distance 0 → relevance 1.0 (perfect match)
                    # distance 2 → relevance 0.0 (opposite) 
                    'relevance_score': 1 - (raw_results['distances'][0][i] / 2)
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection. ⚠️ This is irreversible!
        
        Use this to start fresh or clean up test data.
        """
        self.client.delete_collection(name=self.collection_name)
        print(f"🗑️  Deleted collection: {self.collection_name}")
        
        # Recreate empty collection
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"   ✅ Recreated empty collection")
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector database.
        
        Returns:
            Dict with collection statistics
        """
        count = self.collection.count()
        return {
            "collection_name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory
        }


# ============================================
# TEST & DEMO SECTION
# ============================================

if __name__ == "__main__":
    from rich import print as rprint
    from rich.panel import Panel
    from rich.table import Table
    from embedding_generator import EmbeddingGenerator
    
    rprint(Panel.fit(
        "[bold cyan]🗄️ VECTOR DATABASE — Test & Demo[/bold cyan]\n\n"
        "This module stores embeddings and performs similarity search.\n"
        "It's the 'memory' of your RAG system!",
        title="File 4 of 8",
        border_style="cyan"
    ))
    
    # ============================================
    # SETUP: Initialize components
    # ============================================
    rprint("\n[yellow]Setting up components...[/yellow]")
    
    # Use a test directory (cleaned up after demo)
    test_db_dir = "./test_chroma_db"
    
    embedding_gen = EmbeddingGenerator(model_type="local")
    db = VectorDB(persist_directory=test_db_dir, collection_name="demo_collection")
    
    # ============================================
    # DEMO 1: Add documents
    # ============================================
    rprint(f"\n[bold green]--- Demo 1: Adding Documents ---[/bold green]")
    
    demo_texts = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
        "Deep learning uses neural networks with multiple layers to process complex patterns.",
        "Natural language processing allows computers to understand and generate human language.",
        "Computer vision enables machines to interpret and analyze visual information from images.",
        "Reinforcement learning trains agents by rewarding desired behaviors and penalizing undesired ones.",
        "Transfer learning allows models trained on one task to be applied to different but related tasks.",
        "RAG (Retrieval-Augmented Generation) combines document retrieval with language model generation.",
        "Vector databases store and search high-dimensional embedding vectors efficiently.",
    ]
    
    # Create embeddings
    embeddings = embedding_gen.embed_texts(demo_texts)
    ids = [f"demo_doc_{i}" for i in range(len(demo_texts))]
    metadatas = [{"source": "demo", "topic": "AI", "index": i} for i in range(len(demo_texts))]
    
    # Add to vector DB
    db.add_documents(
        ids=ids,
        embeddings=embeddings,
        documents=demo_texts,
        metadatas=metadatas
    )
    
    # ============================================
    # DEMO 2: Search for similar documents
    # ============================================
    rprint(f"\n[bold green]--- Demo 2: Similarity Search ---[/bold green]")
    
    queries = [
        "How does AI learn from examples?",
        "What is the process of understanding text?",
        "How do computers see and understand images?",
    ]
    
    for query in queries:
        rprint(f"\n[bold cyan]Query: '{query}'[/bold cyan]")
        results = db.search_with_text(query, embedding_gen, n_results=3)
        
        table = Table(title=f"Top 3 Results")
        table.add_column("Rank", style="yellow", width=5)
        table.add_column("Text", style="green", width=65)
        table.add_column("Score", style="cyan", width=8, justify="right")
        
        for i, result in enumerate(results):
            score = result['relevance_score']
            score_str = f"{score:.4f}"
            table.add_row(f"#{i+1}", result['text'][:80] + "...", score_str)
        
        rprint(table)
    
    # ============================================
    # DEMO 3: Statistics
    # ============================================
    rprint(f"\n[bold green]--- Demo 3: Database Stats ---[/bold green]")
    stats = db.get_stats()
    rprint(f"Collection: {stats['collection_name']}")
    rprint(f"Documents: {stats['document_count']}")
    rprint(f"Storage: {stats['persist_directory']}")
    
    # Clean up test database
    db.delete_collection()
    import shutil
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)
    rprint(f"\n[dim]Cleaned up test database[/dim]")
    
    rprint(Panel.fit(
        "[bold green]✅ Vector Database working correctly![/bold green]\n\n"
        "[yellow]WHAT YOU LEARNED:[/yellow]\n"
        "• Vector DB stores embeddings for fast similarity search\n"
        "• ChromaDB runs locally — no cloud setup needed\n"
        "• Collections are like tables in a regular database\n"
        "• HNSW algorithm enables fast approximate search\n"
        "• Cosine distance: 0 = identical, 2 = opposite\n"
        "• Metadata filtering: search within specific documents\n\n"
        "[yellow]INTERVIEW GOLD:[/yellow]\n"
        "• 'Vector DB vs Regular DB?' → Vector DB uses similarity\n"
        "  search, regular DB uses exact matching.\n"
        "• 'How does HNSW work?' → Multi-layer graph, searches from\n"
        "  coarse to fine layers, O(log n) complexity.\n"
        "• 'Top-K selection?' → Retrieve enough for coverage (3-5),\n"
        "  not so many that irrelevant context floods the LLM.\n\n"
        "[yellow]NEXT STEP:[/yellow] rag_pipeline.py — combining retrieval + LLM!",
        title="Summary",
        border_style="green"
    ))
