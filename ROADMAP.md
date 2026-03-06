# 🚀 Generative AI Mastery Roadmap
## From Beginner to Interview-Ready (Project-Based Learning)

> **Your Background**: ML/DL fundamentals ✅
> **Starting Point**: RAG (Retrieval-Augmented Generation) 
> **Goal**: Deep conceptual understanding + practical implementation

---

## 📊 LEVEL 0: Foundation Concepts (Learn WHILE Building RAG)

These concepts will be taught **inline** as we build the RAG project.
You don't need to study them separately — I'll explain each one when we encounter it.

### 🔤 Tokenization
- **What**: How text is broken into smaller units (tokens) that AI models understand
- **Why it matters**: Every LLM thinks in tokens, not words. "unhappiness" → ["un", "happiness"]
- **Interview Buzzwords**: BPE (Byte Pair Encoding), WordPiece, SentencePiece, Token Limit, Context Window
- **When you'll learn**: During `embedding_generator.py`

### 🧮 Embeddings
- **What**: Converting text into numerical vectors (arrays of numbers) that capture semantic meaning
- **Why it matters**: Computers can't understand text — embeddings are how AI "reads"
- **Interview Buzzwords**: Dense vs Sparse embeddings, Dimensionality, Semantic Similarity, Cosine Similarity
- **When you'll learn**: During `embedding_generator.py` and `vector_db.py`

### 🎯 Attention Mechanism
- **What**: How Transformers decide which parts of input are most relevant
- **Why it matters**: This is THE breakthrough that made GPT, Claude, etc. possible
- **Interview Buzzwords**: Self-Attention, Multi-Head Attention, Query-Key-Value, Softmax
- **When you'll learn**: Concept explanation during `rag_pipeline.py`

### 🏗️ Transformer Architecture
- **What**: The neural network architecture behind ALL modern LLMs
- **Why it matters**: GPT = "Generative Pre-trained **Transformer**"
- **Interview Buzzwords**: Encoder-Decoder, Positional Encoding, Feed-Forward, Layer Normalization
- **When you'll learn**: Deep dive document provided separately

---

## 📦 LEVEL 1: RAG Chatbot (CURRENT PROJECT)

### What is RAG?
**Retrieval-Augmented Generation** = Instead of asking an LLM to answer from its training data alone,
you FIRST retrieve relevant documents, THEN feed them to the LLM as context.

**Why RAG exists**: LLMs have a **knowledge cutoff** and can **hallucinate** (make up facts).
RAG solves this by grounding the LLM's answers in YOUR actual documents.

### Project Architecture
```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  data_loader │────▶│   chunker    │────▶│ embedding_gen   │
│  (PDF/Text)  │     │ (Split Text) │     │ (Text→Vectors)  │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  LLM API    │◀────│ rag_pipeline │◀────│   vector_db     │
│  (Generate) │     │ (Orchestrate)│     │ (Store/Search)  │
└──────┬──────┘     └──────────────┘     └─────────────────┘
       │
       ▼
   Answer to User
```

### Files We Will Build (IN ORDER):
| # | File | Purpose | Key Concepts |
|---|------|---------|-------------|
| 1 | `data_loader.py` | Load PDF/Text files | Document parsing, Text extraction |
| 2 | `chunker.py` | Split text into chunks | Chunking strategy, Overlap, Token limits |
| 3 | `embedding_generator.py` | Create vector embeddings | Embeddings, Tokenization, Models |
| 4 | `vector_db.py` | Store & search vectors | Vector DB, Similarity Search, Indexing |
| 5 | `rag_pipeline.py` | Connect retrieval + LLM | RAG flow, Prompt Templates, Context Injection |
| 6 | `config.py` | Centralized configuration | API keys, Model settings |
| 7 | `app.py` | FastAPI backend | API endpoints, REST, Integration |
| 8 | Frontend | Streamlit UI | User interface, Chat experience |

### What You'll Master:
- ✅ Document ingestion and preprocessing
- ✅ Text chunking strategies (fixed size, recursive, semantic)
- ✅ Embedding models (OpenAI, HuggingFace, Sentence-Transformers)
- ✅ Vector databases (ChromaDB, FAISS, Pinecone)
- ✅ Similarity search (Cosine, Euclidean, Dot Product)
- ✅ Prompt engineering and template design
- ✅ LLM API integration (OpenAI, Groq, Claude)
- ✅ End-to-end pipeline orchestration
- ✅ API development with FastAPI
- ✅ Frontend with Streamlit

---

## 🔧 LEVEL 2: Advanced RAG Techniques

### After completing Level 1, you'll learn:
| Technique | What It Does | Why It Matters |
|-----------|-------------|---------------|
| **Re-Ranking** | Re-score retrieved chunks for relevance | Improves answer quality dramatically |
| **Hybrid Search** | Combine keyword + semantic search | Best of both worlds |
| **Query Expansion** | Rephrase query for better retrieval | Handles vague user questions |
| **HyDE** | Hypothetical Document Embeddings | Revolutionary retrieval technique |
| **Metadata Filtering** | Filter by date, source, category | Precision retrieval |
| **Multi-Query RAG** | Generate multiple search queries | Covers more ground |
| **RAGAS Evaluation** | Measure RAG quality scientifically | Answer relevancy, Faithfulness, Context relevancy |
| **Parent Document Retriever** | Retrieve parent chunks for more context | Better context understanding |

---

## 🔗 LEVEL 3: LangChain / LlamaIndex

### Frameworks that Simplify AI Development
| Framework | Strength | When to Use |
|-----------|----------|-------------|
| **LangChain** | Flexible chaining of LLM operations | Complex workflows, Agents |
| **LlamaIndex** | Superior data indexing & retrieval | Data-heavy applications |
| **Haystack** | Production-ready pipelines | Enterprise search |

### What You'll Build:
- Conversational RAG with memory
- Multi-document QA system
- SQL + Document hybrid system

---

## 🎯 LEVEL 4: Fine-Tuning LLMs

### Make LLMs Your Own
| Concept | What It Is |
|---------|-----------|
| **Full Fine-Tuning** | Update ALL model weights (expensive) |
| **LoRA** | Low-Rank Adaptation — update only small matrices |
| **QLoRA** | Quantized LoRA — fine-tune on consumer GPUs |
| **PEFT** | Parameter-Efficient Fine-Tuning — umbrella term |
| **Instruction Tuning** | Teach model to follow instructions |
| **RLHF** | Reinforce Learning from Human Feedback |
| **DPO** | Direct Preference Optimization (simpler than RLHF) |

### What You'll Build:
- Fine-tune a model on custom domain data
- Create a specialized chatbot for a specific use case

---

## 🤖 LEVEL 5: AI Agents

### The Future of AI
| Concept | What It Is |
|---------|-----------|
| **Tool Calling** | LLM decides which tools/APIs to use |
| **Function Calling** | LLM generates structured function calls |
| **ReAct Pattern** | Reasoning + Acting in a loop |
| **LangGraph** | Build stateful, multi-step agent workflows |
| **CrewAI** | Multi-agent collaboration framework |
| **AutoGen** | Microsoft's multi-agent framework |
| **MCP** | Model Context Protocol — standardized tool integration |

### What You'll Build:
- Research agent that searches the web
- Code generation agent
- Multi-agent system for complex tasks

---

## 🚀 LEVEL 6: Production & Deployment

### Taking AI to Production
| Topic | What You'll Learn |
|-------|------------------|
| **Guardrails** | Prevent harmful/wrong outputs |
| **Caching** | Reduce API costs with semantic caching |
| **Monitoring** | Track performance, latency, quality |
| **Cost Optimization** | Token management, model selection |
| **Scaling** | Handle concurrent users |
| **Security** | Prompt injection prevention |
| **Evaluation** | Systematic testing of AI systems |

---

## 📚 Interview Preparation Checklist

### Must-Know Concepts:
- [ ] What is RAG and why is it needed?
- [ ] Explain the difference between fine-tuning and RAG
- [ ] What are embeddings and how do they work?
- [ ] How does cosine similarity work?
- [ ] What chunking strategies exist and when to use each?
- [ ] What is a vector database?
- [ ] What is the Transformer architecture?
- [ ] Explain self-attention mechanism
- [ ] What is tokenization and why does it matter?
- [ ] What is prompt engineering?
- [ ] What is hallucination and how to prevent it?
- [ ] Explain LoRA and QLoRA
- [ ] What is RLHF?
- [ ] What is an AI Agent?
- [ ] What is function/tool calling?
- [ ] What is the context window?
- [ ] What is temperature in LLM generation?
- [ ] Explain top-p (nucleus sampling)
- [ ] What are guardrails in AI systems?
- [ ] How do you evaluate RAG systems?

---

## 🎯 Your Immediate Next Steps

1. ✅ Read this roadmap
2. 🔜 Set up the project environment
3. 🔜 Build `data_loader.py` (File 1 of 8)
4. 🔜 Understand document parsing concepts
5. 🔜 Test with sample PDFs

**Let's begin! 🚀**
