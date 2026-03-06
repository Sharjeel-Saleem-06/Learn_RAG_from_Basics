"""
============================================
FILE 8 OF 8: TEST_PIPELINE.PY — Test & Learn Script
============================================

📝 PURPOSE:
    This is your INTERACTIVE LEARNING script.
    Run it to test each component step by step
    and see how they work together.
    
    It's designed to help you understand the RAG flow
    by showing you exactly what happens at each stage.

🎓 HOW TO USE:
    python test_pipeline.py
    
    This will:
    1. Create a sample document
    2. Walk through each pipeline stage
    3. Show you the intermediate results
    4. Let you ask questions interactively

============================================
"""

import os
import sys
import time


def test_step_by_step():
    """
    Walk through the entire RAG pipeline step by step,
    showing what happens at each stage.
    """
    from rich import print as rprint
    from rich.panel import Panel
    from rich.table import Table
    from rich.console import Console
    
    console = Console()
    
    rprint(Panel.fit(
        "[bold cyan]🎓 RAG PIPELINE — Step-by-Step Walkthrough[/bold cyan]\n\n"
        "This script walks you through every stage of the RAG pipeline.\n"
        "Watch what happens to your data at each step!",
        border_style="cyan"
    ))
    
    # ============================================
    # STEP 0: Create Sample Document
    # ============================================
    rprint("\n[bold yellow]━━━ STEP 0: Creating Sample Document ━━━[/bold yellow]\n")
    
    sample_dir = "./sample_docs"
    os.makedirs(sample_dir, exist_ok=True)
    
    sample_text = """
Artificial Intelligence in Healthcare: A Comprehensive Overview

Introduction

Artificial intelligence (AI) is revolutionizing the healthcare industry by enabling more accurate 
diagnoses, personalized treatment plans, and efficient administrative processes. The integration of 
AI technologies in healthcare settings has shown promising results across multiple domains.

Medical Imaging and Diagnostics

AI-powered imaging analysis has achieved remarkable accuracy in detecting diseases from medical images. 
Deep learning models, particularly Convolutional Neural Networks (CNNs), can analyze X-rays, MRI scans, 
and CT images to identify abnormalities such as tumors, fractures, and infections. In some studies, 
AI systems have demonstrated diagnostic accuracy comparable to or exceeding that of experienced radiologists.

Key applications include:
- Detecting cancer in mammograms and lung CT scans
- Identifying diabetic retinopathy in eye scans
- Analyzing pathology slides for disease markers
- Detecting brain abnormalities in MRI images

Drug Discovery and Development

The traditional drug development process takes 10-15 years and costs billions of dollars. AI is 
dramatically accelerating this process through:

1. Target Identification: Machine learning models analyze biological data to identify potential 
   drug targets — specific molecules that can be affected by a drug to treat a disease.

2. Molecular Design: Generative AI models can design novel molecular structures with desired 
   properties, significantly reducing the search space for potential drug candidates.

3. Clinical Trial Optimization: AI analyzes patient data to identify ideal candidates for 
   clinical trials, predict outcomes, and optimize trial design.

4. Drug Repurposing: AI identifies existing drugs that could be effective for new conditions, 
   potentially bypassing years of development.

Natural Language Processing in Healthcare

NLP technologies enable computers to understand, interpret, and generate human language in 
healthcare contexts:

- Clinical Documentation: AI-powered transcription and summarization of doctor-patient conversations
- Medical Literature Analysis: Processing thousands of research papers to extract relevant information
- Patient Communication: Chatbots and virtual assistants for patient queries and triage
- Electronic Health Records: Extracting structured data from unstructured clinical notes

Challenges and Considerations

Despite its potential, AI in healthcare faces several challenges:

1. Data Privacy: Healthcare data is highly sensitive and subject to regulations like HIPAA. 
   AI systems must ensure patient data confidentiality and security.

2. Bias and Fairness: AI models trained on biased datasets may produce inequitable results. 
   Ensuring diverse, representative training data is crucial.

3. Explainability: Healthcare AI decisions must be interpretable. "Black box" models that 
   can't explain their reasoning face resistance from clinicians and regulators.

4. Integration: Implementing AI solutions in existing healthcare infrastructure requires 
   significant technical and organizational changes.

5. Regulatory Compliance: Medical AI devices must meet regulatory standards (FDA approval 
   in the US, CE marking in Europe) before clinical use.

Future Outlook

The future of AI in healthcare includes:
- Precision medicine tailored to individual patient genetics
- Real-time health monitoring through wearable sensors and AI analysis
- Robot-assisted surgery with enhanced precision
- Predictive analytics for disease outbreaks and public health
- AI-driven mental health support and intervention
    """.strip()
    
    sample_file = os.path.join(sample_dir, "ai_healthcare.txt")
    with open(sample_file, 'w') as f:
        f.write(sample_text)
    
    rprint(f"Created sample document: [cyan]{sample_file}[/cyan]")
    rprint(f"Document length: [green]{len(sample_text):,} characters[/green]")
    
    input("\n▶️  Press Enter to continue to Step 1 (Data Loading)...")

    # ============================================
    # STEP 1: DATA LOADING
    # ============================================
    rprint("\n[bold yellow]━━━ STEP 1: DATA LOADING ━━━[/bold yellow]\n")
    rprint("[dim]Loading the text file and converting it into a Document object...[/dim]\n")
    
    from data_loader import load_document
    
    documents = load_document(sample_file)
    
    for doc in documents:
        rprint(f"📄 Document loaded:")
        rprint(f"   Source: [cyan]{doc.metadata['source']}[/cyan]")
        rprint(f"   Type: [green]{doc.metadata['file_type']}[/green]")
        rprint(f"   Characters: [yellow]{len(doc.text):,}[/yellow]")
        rprint(f"   Preview: [dim]{doc.text[:150]}...[/dim]")
    
    rprint("\n[bold green]✅ Data loading complete![/bold green]")
    rprint("[dim]→ We now have Document objects with text + metadata[/dim]")
    
    input("\n▶️  Press Enter to continue to Step 2 (Chunking)...")

    # ============================================
    # STEP 2: CHUNKING
    # ============================================
    rprint("\n[bold yellow]━━━ STEP 2: TEXT CHUNKING ━━━[/bold yellow]\n")
    rprint("[dim]Splitting the document into smaller, manageable chunks...[/dim]\n")
    
    from chunker import chunk_documents
    
    chunks = chunk_documents(
        documents, 
        chunk_size=500, 
        chunk_overlap=100, 
        strategy="recursive"
    )
    
    rprint(f"\n📊 Created [bold]{len(chunks)}[/bold] chunks from the document\n")
    
    table = Table(title="📦 All Chunks")
    table.add_column("#", style="yellow", width=4)
    table.add_column("Length", style="cyan", width=8, justify="right")
    table.add_column("Preview", style="green", width=70)
    
    for i, chunk in enumerate(chunks):
        preview = chunk.text[:80].replace("\n", " ") + "..."
        table.add_row(str(i), f"{len(chunk.text)}", preview)
    
    rprint(table)
    
    rprint("\n[bold green]✅ Chunking complete![/bold green]")
    rprint("[dim]→ Each chunk is a self-contained piece of text with metadata[/dim]")
    
    input("\n▶️  Press Enter to continue to Step 3 (Embedding)...")

    # ============================================
    # STEP 3: EMBEDDING
    # ============================================
    rprint("\n[bold yellow]━━━ STEP 3: EMBEDDING GENERATION ━━━[/bold yellow]\n")
    rprint("[dim]Converting text chunks into numerical vectors...[/dim]\n")
    
    from embedding_generator import EmbeddingGenerator, cosine_similarity
    
    emb_gen = EmbeddingGenerator(model_type="local")
    
    texts, embeddings, metadatas, ids = emb_gen.embed_documents(chunks)
    
    rprint(f"\n📊 Generated [bold]{len(embeddings)}[/bold] embeddings")
    rprint(f"   Each embedding has [bold]{len(embeddings[0])}[/bold] dimensions\n")
    
    # Show a few embedding examples
    for i in range(min(3, len(embeddings))):
        rprint(f"Chunk {i}: [dim]{texts[i][:50]}...[/dim]")
        rprint(f"  Vector: [{embeddings[i][0]:.4f}, {embeddings[i][1]:.4f}, "
               f"{embeddings[i][2]:.4f}, ... , {embeddings[i][-1]:.4f}]")
    
    # Show similarity between first chunk and all others
    rprint(f"\n🎯 Similarity of Chunk 0 with all other chunks:")
    for i in range(1, min(len(embeddings), 6)):
        sim = cosine_similarity(embeddings[0], embeddings[i])
        bar_length = int(sim * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        rprint(f"  Chunk {i}: {bar} {sim:.4f}")
    
    rprint("\n[bold green]✅ Embedding generation complete![/bold green]")
    rprint("[dim]→ Each chunk is now a 384-dimensional vector[/dim]")
    
    input("\n▶️  Press Enter to continue to Step 4 (Vector Storage)...")

    # ============================================
    # STEP 4: VECTOR STORAGE
    # ============================================
    rprint("\n[bold yellow]━━━ STEP 4: VECTOR DATABASE STORAGE ━━━[/bold yellow]\n")
    rprint("[dim]Storing embeddings in ChromaDB for fast retrieval...[/dim]\n")
    
    from vector_db import VectorDB
    
    # Use a test directory
    test_db_dir = "./test_walkthrough_db"
    db = VectorDB(persist_directory=test_db_dir, collection_name="walkthrough")
    
    db.add_documents(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )
    
    stats = db.get_stats()
    rprint(f"\n📊 Database stats:")
    rprint(f"   Collection: [cyan]{stats['collection_name']}[/cyan]")
    rprint(f"   Documents: [green]{stats['document_count']}[/green]")
    
    rprint("\n[bold green]✅ Vector storage complete![/bold green]")
    rprint("[dim]→ All chunks are stored and indexed for fast similarity search[/dim]")
    
    input("\n▶️  Press Enter to continue to Step 5 (Retrieval)...")

    # ============================================
    # STEP 5: RETRIEVAL (Search)
    # ============================================
    rprint("\n[bold yellow]━━━ STEP 5: RETRIEVAL (SIMILARITY SEARCH) ━━━[/bold yellow]\n")
    rprint("[dim]Searching for chunks most relevant to a query...[/dim]\n")
    
    test_queries = [
        "How is AI used in medical imaging?",
        "What are the challenges of AI in healthcare?",
        "How does AI help with drug development?"
    ]
    
    for query in test_queries:
        rprint(f"\n[bold cyan]Query: '{query}'[/bold cyan]")
        results = db.search_with_text(query, emb_gen, n_results=3)
        
        for i, result in enumerate(results):
            rprint(f"  [{i+1}] Relevance: {result['relevance_score']:.4f}")
            rprint(f"      [dim]{result['text'][:120]}...[/dim]")
    
    rprint("\n[bold green]✅ Retrieval complete![/bold green]")
    rprint("[dim]→ We found the most relevant chunks for each query[/dim]")
    
    # Clean up test database
    db.delete_collection()
    import shutil
    if os.path.exists(test_db_dir):
        shutil.rmtree(test_db_dir)
    
    input("\n▶️  Press Enter to see the COMPLETE pipeline flow...")

    # ============================================
    # STEP 6: COMPLETE PIPELINE (if API key available)
    # ============================================
    rprint("\n[bold yellow]━━━ STEP 6: COMPLETE RAG PIPELINE ━━━[/bold yellow]\n")
    
    from config import config
    
    if config.groq_api_key and config.groq_api_key != "your_groq_api_key_here":
        rprint("[dim]Running complete pipeline with LLM generation...[/dim]\n")
        
        from rag_pipeline import RAGPipeline
        
        pipeline = RAGPipeline()
        
        # Ingest document
        pipeline.ingest_file(sample_file)
        
        # Ask questions
        questions = [
            "What are the main applications of AI in healthcare?",
            "What challenges does AI face in the medical field?",
        ]
        
        for q in questions:
            response = pipeline.query(q)
            rprint(f"\n[bold cyan]Q: {q}[/bold cyan]")
            rprint(f"\n[green]{response.answer}[/green]")
            rprint(f"\n[dim]⏱️ Retrieval: {response.retrieval_time}s | "
                   f"Generation: {response.generation_time}s[/dim]")
            rprint("─" * 60)
        
        # Interactive mode
        rprint("\n[bold yellow]🎮 INTERACTIVE MODE[/bold yellow]")
        rprint("[dim]Type your questions below (type 'quit' to exit)[/dim]\n")
        
        while True:
            user_input = input("\n❓ Your question: ").strip()
            if user_input.lower() in ('quit', 'exit', 'q'):
                break
            if not user_input:
                continue
            
            response = pipeline.query(user_input)
            rprint(f"\n[green]{response.answer}[/green]")
            rprint(f"[dim]Sources: {response.num_chunks_retrieved} chunks | "
                   f"Time: {response.total_time}s[/dim]")
        
        # Cleanup
        pipeline.reset()
    
    else:
        rprint("[yellow]⚠️ No API key found. Skipping LLM generation step.[/yellow]")
        rprint("[yellow]   To test the full pipeline:[/yellow]")
        rprint("[yellow]   1. Get a FREE key at https://console.groq.com/keys[/yellow]")
        rprint("[yellow]   2. Add it to .env: GROQ_API_KEY=your_key[/yellow]")
        rprint("[yellow]   3. Run this script again[/yellow]")
    
    # ============================================
    # FINAL SUMMARY
    # ============================================
    rprint(Panel.fit(
        "[bold green]🎉 WALKTHROUGH COMPLETE![/bold green]\n\n"
        "[yellow]WHAT YOU SAW:[/yellow]\n"
        "1. 📄 Data Loading → Text extracted from documents\n"
        "2. 🔪 Chunking → Text split into smaller pieces\n"
        "3. 🧮 Embedding → Text converted to numerical vectors\n"
        "4. 🗄️  Storage → Vectors stored in ChromaDB\n"
        "5. 🔍 Retrieval → Similar chunks found via cosine similarity\n"
        "6. 🤖 Generation → LLM generates answer from retrieved context\n\n"
        "[yellow]COMPLETE DATA FLOW:[/yellow]\n"
        "PDF → Text → Chunks → Vectors → VectorDB → Query → Top-K Chunks → LLM → Answer\n\n"
        "[yellow]NEXT STEPS:[/yellow]\n"
        "• Run the Streamlit app: streamlit run streamlit_app.py\n"
        "• Run the FastAPI server: python app.py\n"
        "• Upload your own documents and ask questions!\n"
        "• Experiment with chunk_size and chunk_overlap\n"
        "• Try different numbers of retrieved chunks (n_results)",
        title="Summary",
        border_style="green"
    ))


if __name__ == "__main__":
    test_step_by_step()
