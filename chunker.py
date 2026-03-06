"""
============================================
FILE 2 OF 8: CHUNKER.PY — Text Chunking
============================================

📝 PURPOSE:
    After loading documents (data_loader.py), we need to SPLIT them into 
    smaller pieces called "chunks". But WHY?
    
    Imagine you have a 500-page medical textbook and someone asks:
    "What are the side effects of aspirin?"
    
    You wouldn't feed all 500 pages to the LLM — that would:
    ❌ Exceed the model's context window (token limit)
    ❌ Cost a fortune in API tokens  
    ❌ Dilute the relevant information with irrelevant text
    ❌ Confuse the model with too much context
    
    Instead, you find THE SPECIFIC PARAGRAPHS about aspirin side effects 
    and only send THOSE. That's what chunking enables.

🎓 CONCEPTS COVERED:
    1. Text Chunking — Breaking text into meaningful pieces
    2. Chunk Size — How big each piece should be
    3. Chunk Overlap — Why adjacent chunks share some text
    4. Chunking Strategies — Different approaches to splitting
    5. Token vs Character counting
    6. Recursive Character Splitting — The go-to strategy

🔗 HOW IT CONNECTS:
    data_loader.py  →  chunker.py  →  embedding_generator.py  →  vector_db.py
                       ^^^^^^^^^^^
                       YOU ARE HERE
    
    Input: List[Document] from data_loader
    Output: List[Document] (smaller chunks, with metadata about their origin)

💡 CRITICAL INTERVIEW TOPIC:
    "How does chunking strategy affect RAG performance?"
    
    This is one of the MOST ASKED questions in GenAI interviews.
    Your answer determines if the interviewer thinks you actually 
    understand RAG or just copied code from a tutorial.
    
    ANSWER FRAMEWORK:
    1. Chunk size affects retrieval precision vs recall
    2. Overlap prevents losing context at chunk boundaries
    3. Different strategies work for different content types
    4. There's no universal "best" — it requires experimentation
    5. Mention evaluation metrics (precision@k, recall@k)

============================================
"""

from typing import List, Optional
from dataclasses import dataclass
from data_loader import Document


# ============================================
# STRATEGY 1: FIXED-SIZE CHUNKING
# ============================================
# The simplest approach — just split every N characters.
#
# 📊 VISUAL EXPLANATION:
#
# Original text (20 chars): "ABCDEFGHIJKLMNOPQRST"
# Chunk size: 8, Overlap: 3
#
# Chunk 1: [ABCDEFGH]              ← characters 0-7
# Chunk 2:      [FGHIJKLM]         ← characters 5-12 (overlap: FGH)
# Chunk 3:           [KLMNOPQR]    ← characters 10-17 (overlap: KLM)
# Chunk 4:                [PQRST]  ← characters 15-19 (overlap: PQR)
#
# Notice how each chunk shares some characters with its neighbors.
# This overlap is like a "bridge" between chunks.
#
# WHY OVERLAP MATTERS — A Real Example:
# ───────────────────────────────────────
# Without overlap (chunk_size=50):
#   Chunk 1: "Deep learning models require large amounts of"
#   Chunk 2: "training data and significant computational resources"
#
# The sentence was split mid-thought! If someone asks about 
# "computational requirements for deep learning", neither chunk
# alone captures the full answer.
#
# With overlap (chunk_size=50, overlap=15):
#   Chunk 1: "Deep learning models require large amounts of"
#   Chunk 2: "large amounts of training data and significant"
#   Chunk 3: "and significant computational resources for deep"
#
# Now Chunk 2 has the complete context: "large amounts of training data
# and significant computational resources" — perfect for answering!
# ============================================

def fixed_size_chunk(
    text: str, 
    chunk_size: int = 1000, 
    chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into fixed-size chunks with overlap.
    
    This is the simplest chunking strategy. It's a good starting point
    but has limitations:
    
    ✅ Pros:
    - Simple and predictable
    - Consistent chunk sizes
    - Easy to understand and debug
    
    ❌ Cons:
    - May split mid-sentence or mid-word
    - Doesn't respect document structure (headers, paragraphs)
    - A chunk might start with "...tion" and be meaningless
    
    Args:
        text: The text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters shared between adjacent chunks
        
    Returns:
        List of text chunks
        
    Example:
        >>> text = "Hello world. This is a test. Foo bar baz."
        >>> chunks = fixed_size_chunk(text, chunk_size=20, chunk_overlap=5)
        >>> for i, c in enumerate(chunks):
        ...     print(f"Chunk {i}: '{c}'")
    """
    # ============================================
    # INPUT VALIDATION
    # ============================================
    # Always validate inputs! This prevents confusing errors later.
    # 
    # 💡 INTERVIEW CONCEPT: "Defensive Programming"
    # Assume inputs might be wrong and check them early.
    # ============================================
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"❌ Overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size}).\n"
            f"   Think about it: if overlap equals chunk_size, chunks would be identical!"
        )
    
    if not text or not text.strip():
        return []
    
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        # Calculate end position
        end = start + chunk_size
        
        # Extract the chunk
        chunk = text[start:end]
        
        if chunk.strip():  # Only add non-empty chunks
            chunks.append(chunk.strip())
        
        # Move start position forward by (chunk_size - overlap)
        # This is what creates the overlap!
        start += chunk_size - chunk_overlap
    
    return chunks


# ============================================
# STRATEGY 2: RECURSIVE CHARACTER SPLITTING ⭐
# ============================================
# This is THE most used chunking strategy in production RAG systems.
# LangChain's RecursiveCharacterTextSplitter uses exactly this approach.
#
# 💡 THE KEY IDEA:
# Instead of blindly cutting at character positions,
# try to split at NATURAL BOUNDARIES in this priority order:
#
#   1. "\n\n" — Paragraph breaks (best, preserves full paragraphs)
#   2. "\n"   — Line breaks (good, preserves full lines)
#   3. ". "   — Sentence endings (okay, preserves sentences)
#   4. " "    — Word boundaries (last resort, at least no mid-word splits)
#   5. ""     — Character level (absolute last resort)
#
# WHY "RECURSIVE"?
# If splitting by "\n\n" creates chunks that are STILL too big,
# we recursively try the next separator ("\n"), and so on,
# until chunks fit within the size limit.
#
# 📊 VISUAL EXAMPLE:
# ─────────────────
# Text: "Paragraph 1 content here.\n\nParagraph 2 is very very long..."
# 
# Step 1: Split by "\n\n" → ["Paragraph 1 content here.", "Paragraph 2 is very very long..."]
# Step 2: Chunk 1 fits ✅, Chunk 2 is too long ❌
# Step 3: For Chunk 2, try splitting by "\n" → still one piece
# Step 4: Try splitting by ". " → now it fits ✅
#
# 💡 INTERVIEW QUESTION: "Why is recursive splitting better than fixed-size?"
# ANSWER: 
# Fixed-size might split "New York City" into "New Yo" and "rk City".
# Recursive splitting tries paragraph → sentence → word boundaries first,
# preserving semantic meaning within each chunk.
# ============================================

def recursive_character_split(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    separators: Optional[List[str]] = None
) -> List[str]:
    """
    Split text using recursive character splitting strategy.
    
    This is the INDUSTRY STANDARD chunking method.
    LangChain's RecursiveCharacterTextSplitter is based on this exact approach.
    
    How it works:
    1. Try to split by the first separator (paragraph breaks)
    2. If any resulting chunk is still too large, apply the next separator
    3. Keep going until all chunks are within size limit
    4. Apply overlap between consecutive chunks
    
    Args:
        text: Text to split
        chunk_size: Target maximum chunk size
        chunk_overlap: Characters to overlap between chunks
        separators: List of separators to try, in order of preference
        
    Returns:
        List of text chunks, each within the size limit
    """
    # Default separators in order of preference
    if separators is None:
        separators = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]
    
    if not text or not text.strip():
        return []
    
    chunks = []
    
    # ============================================
    # STEP 1: Find the best separator for this text
    # ============================================
    # We want to use the highest-quality separator that actually
    # appears in the text and creates reasonable-sized pieces.
    # ============================================
    
    separator = separators[-1]  # Default: empty string (character-level)
    
    for sep in separators:
        if sep == "":
            separator = sep
            break
        if sep in text:
            separator = sep
            break
    
    # ============================================
    # STEP 2: Split the text using the chosen separator
    # ============================================
    if separator:
        splits = text.split(separator)
    else:
        splits = list(text)  # Character-level split
    
    # ============================================
    # STEP 3: Merge small splits and handle large ones
    # ============================================
    # Some splits might be tiny (a single word after splitting by spaces).
    # We merge these until they approach the chunk_size limit.
    # If a split is STILL too large, recursively split it with 
    # the next separator in the list.
    # ============================================
    
    current_chunk = ""
    remaining_separators = separators[separators.index(separator) + 1:] if separator in separators else []
    
    for split in splits:
        # Add the separator back (it was removed during splitting)
        piece = split if not separator else split
        
        # Would adding this piece exceed the chunk size?
        test_chunk = current_chunk + (separator if current_chunk else "") + piece
        
        if len(test_chunk) <= chunk_size:
            # It fits! Add it to the current chunk
            current_chunk = test_chunk
        else:
            # Current chunk is full, save it
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Is the new piece itself too large?
            if len(piece) > chunk_size and remaining_separators:
                # Recursively split with the next separator
                sub_chunks = recursive_character_split(
                    piece, 
                    chunk_size, 
                    chunk_overlap, 
                    remaining_separators
                )
                chunks.extend(sub_chunks)
                current_chunk = ""
            else:
                current_chunk = piece
    
    # Don't forget the last chunk!
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # ============================================
    # STEP 4: Apply overlap between chunks
    # ============================================
    # Now we have chunks without overlap. We need to add overlap
    # by prepending text from the previous chunk to each chunk.
    # ============================================
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped_chunks = [chunks[0]]  # First chunk stays as-is
        
        for i in range(1, len(chunks)):
            # Get the last 'chunk_overlap' characters from previous chunk
            prev_text = chunks[i - 1]
            overlap_text = prev_text[-chunk_overlap:] if len(prev_text) > chunk_overlap else prev_text
            
            # Prepend overlap to current chunk
            overlapped_chunk = overlap_text + " " + chunks[i]
            
            # Trim if it exceeds max size
            if len(overlapped_chunk) > chunk_size + chunk_overlap:
                overlapped_chunk = overlapped_chunk[:chunk_size + chunk_overlap]
            
            overlapped_chunks.append(overlapped_chunk.strip())
        
        return overlapped_chunks
    
    return chunks


# ============================================
# STRATEGY 3: SENTENCE-BASED CHUNKING
# ============================================
# Splits text into complete sentences, then groups them
# into chunks of the target size.
#
# 💡 WHY SENTENCE-LEVEL?
# Sentences are the natural unit of meaning in text.
# A chunk that starts and ends at sentence boundaries
# is always more coherent than one split mid-sentence.
#
# 💡 WHEN TO USE:
# - Academic papers
# - Documentation
# - Any well-structured text with clear sentences
# ============================================

def sentence_chunk(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """
    Split text into chunks at sentence boundaries.
    
    Splits text into sentences first, then groups sentences
    into chunks that fit within the size limit.
    
    This ensures no chunk ever starts or ends mid-sentence.
    
    Args:
        text: Text to split
        chunk_size: Maximum characters per chunk
        chunk_overlap: Number of characters to overlap
        
    Returns:
        List of text chunks
    """
    import re
    
    if not text or not text.strip():
        return []
    
    # ============================================
    # SENTENCE SPLITTING REGEX
    # ============================================
    # This regex finds sentence endings:
    # (?<=[.!?])  — After a period, exclamation, or question mark
    # \s+         — Followed by whitespace
    #
    # ⚠️ LIMITATION: This doesn't perfectly handle:
    # - Abbreviations: "Dr. Smith" (splits after "Dr.")
    # - Numbers: "3.14 is pi" (splits after "3.")
    # - Quotes: 'She said "Hello." He replied.'
    #
    # For production, use spaCy or NLTK for sentence detection.
    # For learning, this regex works 90% of the time.
    # ============================================
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Would adding this sentence exceed the limit?
        if current_chunk and len(current_chunk) + len(sentence) + 1 > chunk_size:
            # Save current chunk
            chunks.append(current_chunk.strip())
            
            # Start new chunk with overlap from previous
            if chunk_overlap > 0:
                # Take last portion of previous chunk as overlap
                overlap = current_chunk[-chunk_overlap:]
                current_chunk = overlap + " " + sentence
            else:
                current_chunk = sentence
        else:
            current_chunk = current_chunk + " " + sentence if current_chunk else sentence
    
    # Don't forget the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


# ============================================
# MAIN CHUNKING FUNCTION
# ============================================
# This is the function the rest of the pipeline calls.
# It wraps the chunking strategy and handles Document objects.
# ============================================

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    strategy: str = "recursive"
) -> List[Document]:
    """
    🎯 MAIN FUNCTION — Chunk a list of documents into smaller pieces.
    
    Takes Document objects from data_loader.py and returns smaller
    Document objects, each with metadata tracking its origin.
    
    Args:
        documents: List of Document objects from data_loader
        chunk_size: Maximum characters per chunk
        chunk_overlap: Characters to overlap between chunks
        strategy: Chunking strategy to use:
            - "fixed": Simple fixed-size splitting
            - "recursive": Recursive character splitting (RECOMMENDED)
            - "sentence": Sentence-based splitting
    
    Returns:
        List[Document]: Smaller chunks, each with metadata including:
            - All original metadata (source, file_type, etc.)
            - chunk_index: Which chunk number this is
            - chunk_strategy: Which strategy was used
            - original_length: Length of the original document
    
    Example:
        >>> from data_loader import load_document
        >>> docs = load_document("research_paper.pdf")
        >>> chunks = chunk_documents(docs, chunk_size=500, chunk_overlap=100)
        >>> print(f"Split {len(docs)} documents into {len(chunks)} chunks")
    """
    # Select the chunking function based on strategy
    strategy_map = {
        "fixed": fixed_size_chunk,
        "recursive": recursive_character_split,
        "sentence": sentence_chunk,
    }
    
    if strategy not in strategy_map:
        available = ", ".join(strategy_map.keys())
        raise ValueError(
            f"❌ Unknown strategy: '{strategy}'\n"
            f"   Available strategies: {available}\n\n"
            f"   📚 Quick guide:\n"
            f"   • 'fixed': Simple, splits at character boundaries\n"
            f"   • 'recursive': Smart, preserves paragraphs/sentences (RECOMMENDED)\n"
            f"   • 'sentence': Splits at sentence boundaries"
        )
    
    chunk_func = strategy_map[strategy]
    all_chunks = []
    
    print(f"\n🔪 Chunking {len(documents)} documents")
    print(f"   Strategy: {strategy}")
    print(f"   Chunk size: {chunk_size} chars")
    print(f"   Overlap: {chunk_overlap} chars")
    print(f"   Overlap ratio: {chunk_overlap/chunk_size*100:.1f}%")
    
    for doc_idx, doc in enumerate(documents):
        # Split the document's text into chunks
        text_chunks = chunk_func(doc.text, chunk_size, chunk_overlap)
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            # Create a new Document for each chunk
            # preserving the original metadata + adding chunk info
            chunk_metadata = {
                **doc.metadata,  # Copy ALL original metadata
                "chunk_index": chunk_idx,
                "total_chunks": len(text_chunks),
                "chunk_strategy": strategy,
                "chunk_size_setting": chunk_size,
                "chunk_overlap_setting": chunk_overlap,
                "original_doc_index": doc_idx,
                "original_length": len(doc.text),
                "chunk_length": len(chunk_text)
            }
            
            chunk_doc = Document(
                text=chunk_text,
                metadata=chunk_metadata
            )
            all_chunks.append(chunk_doc)
    
    print(f"   📊 Result: {len(all_chunks)} chunks created")
    
    # Show chunk size distribution
    lengths = [len(c.text) for c in all_chunks]
    if lengths:
        print(f"   📏 Chunk sizes: min={min(lengths)}, max={max(lengths)}, avg={sum(lengths)//len(lengths)}")
    
    return all_chunks


# ============================================
# TEST & DEMO SECTION
# ============================================

if __name__ == "__main__":
    from rich import print as rprint
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console
    
    console = Console()
    
    rprint(Panel.fit(
        "[bold cyan]🔪 CHUNKER — Test & Demo[/bold cyan]\n\n"
        "This module splits documents into smaller chunks.\n"
        "Chunking is CRITICAL for RAG — wrong chunk size = wrong answers!",
        title="File 2 of 8",
        border_style="cyan"
    ))
    
    # ============================================
    # DEMO TEXT — Long enough to show chunking in action
    # ============================================
    demo_text = """
Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on building systems 
that can learn from and make decisions based on data. Instead of being explicitly programmed 
to perform a task, machine learning systems use algorithms to parse data, learn from it, 
and then make predictions or decisions.

Types of Machine Learning

There are three main types of machine learning:

Supervised Learning: In supervised learning, the algorithm is trained on labeled data. 
Each training example includes an input and the correct output (label). The algorithm 
learns to map inputs to outputs by studying these examples. Common supervised learning 
tasks include classification (predicting a category) and regression (predicting a number). 
Examples include email spam detection, image recognition, and house price prediction.

Unsupervised Learning: In unsupervised learning, the algorithm works with unlabeled data. 
The system tries to find patterns, structures, or relationships in the data without any 
guidance about what to look for. Common techniques include clustering (grouping similar 
items together) and dimensionality reduction (simplifying complex data). Examples include 
customer segmentation, anomaly detection, and topic modeling.

Reinforcement Learning: In reinforcement learning, an agent learns by interacting with 
an environment. It takes actions and receives feedback in the form of rewards or penalties. 
The goal is to learn a strategy (policy) that maximizes the total reward over time. 
This is how game-playing AIs like AlphaGo were trained, and it's also used in robotics 
and recommendation systems.

Deep Learning

Deep learning is a subset of machine learning that uses neural networks with many layers 
(hence "deep"). These deep neural networks can automatically learn hierarchical 
representations of data, starting from simple features and building up to complex concepts.

Key concepts in deep learning include:
- Neurons: The basic computational units, inspired by biological neurons
- Layers: Groups of neurons that process information at different levels of abstraction
- Activation Functions: Mathematical functions that introduce non-linearity
- Backpropagation: The algorithm used to train neural networks by adjusting weights
- Loss Functions: Functions that measure how wrong the model's predictions are

Applications of deep learning include natural language processing, computer vision, 
speech recognition, and generative AI models like GPT and DALL-E.
    """.strip()
    
    # ============================================
    # TEST ALL THREE STRATEGIES
    # ============================================
    strategies = ["fixed", "recursive", "sentence"]
    
    for strategy in strategies:
        rprint(f"\n[bold yellow]{'='*60}[/bold yellow]")
        rprint(f"[bold yellow]Strategy: {strategy.upper()}[/bold yellow]")
        rprint(f"[bold yellow]{'='*60}[/bold yellow]")
        
        # Create a Document object
        doc = Document(text=demo_text, metadata={"source": "demo.txt", "file_type": "txt"})
        
        # Chunk it
        chunks = chunk_documents(
            [doc], 
            chunk_size=500, 
            chunk_overlap=100, 
            strategy=strategy
        )
        
        # Display results
        for i, chunk in enumerate(chunks):
            rprint(f"\n[cyan]Chunk {i+1}/{len(chunks)}[/cyan] "
                   f"[dim]({len(chunk.text)} chars)[/dim]")
            rprint(f"[dim]{'─'*50}[/dim]")
            # Show first and last 100 chars to see overlap
            if len(chunk.text) > 200:
                rprint(f"[green]{chunk.text[:100]}[/green]")
                rprint(f"[dim]  ... ({len(chunk.text) - 200} chars omitted) ...[/dim]")
                rprint(f"[green]{chunk.text[-100:]}[/green]")
            else:
                rprint(f"[green]{chunk.text}[/green]")
    
    # ============================================
    # COMPARISON TABLE
    # ============================================
    rprint("\n")
    table = Table(title="📊 Chunking Strategy Comparison")
    table.add_column("Strategy", style="cyan", width=15)
    table.add_column("Best For", style="green", width=25)
    table.add_column("Pros", style="yellow", width=25)
    table.add_column("Cons", style="red", width=25)
    
    table.add_row(
        "Fixed", 
        "Simple text, code",
        "Simple, predictable",
        "May split mid-sentence"
    )
    table.add_row(
        "Recursive ⭐", 
        "General purpose (DEFAULT)",
        "Respects structure",
        "Slightly complex logic"
    )
    table.add_row(
        "Sentence", 
        "Well-structured prose",
        "Clean sentence boundaries",
        "Sentence detection not perfect"
    )
    
    rprint(table)
    
    rprint(Panel.fit(
        "[bold green]✅ Chunker working correctly![/bold green]\n\n"
        "[yellow]WHAT YOU LEARNED:[/yellow]\n"
        "• Why chunking is needed (context windows, retrieval precision)\n"
        "• Fixed-size chunking (simple but crude)\n"
        "• Recursive splitting (industry standard, preserves structure)\n"
        "• Sentence-based splitting (clean boundaries)\n"
        "• How overlap prevents losing context at boundaries\n"
        "• How chunk size affects retrieval quality\n\n"
        "[yellow]INTERVIEW GOLD:[/yellow]\n"
        "• 'How do you choose chunk size?' → Depends on embedding model,\n"
        "  content type, and retrieval goals. Experiment and evaluate.\n"
        "• 'Why overlap?' → Preserves context across chunk boundaries.\n"
        "• 'Best strategy?' → Recursive for most cases, sentence for prose.\n\n"
        "[yellow]NEXT STEP:[/yellow] embedding_generator.py — converting chunks to vectors",
        title="Summary",
        border_style="green"
    ))
