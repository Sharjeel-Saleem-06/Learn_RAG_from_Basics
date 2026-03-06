# 🧠 LLM Foundations — From Zero to AI Engineer

> **A complete, interactive learning platform for mastering Large Language Models — from tokenization to AI agents in production. Built for software engineers transitioning into Generative AI.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Netlify-00C7B7?style=for-the-badge&logo=netlify)](https://your-site.netlify.app)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?style=for-the-badge&logo=github)](https://github.com/Sharjeel-Saleem-06/Learn_LLM_foundations_from_Basics)

---

## 🎯 Who Is This For?

This project is designed for **software engineers** (like you!) who:
- Know how to code but have zero AI background
- Want to understand **every concept** behind ChatGPT, Claude, and LLaMA
- Need to be **interview-ready** for AI Engineer roles
- Want to write AI code from an LLM co-pilot (Claude/Cursor) with full conceptual understanding

**Goal:** After completing this, you can look at any LLM-powered system and understand exactly how every piece works.

---

## 📚 What's Covered (14 Chapters + Interview Prep)

### 🟢 Phase 1 — Foundations (Chapters 01–05)

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| 01 | **Tokenization & BPE** | How text becomes numbers, BPE algorithm, token cost estimation |
| 02 | **Embeddings** | Dense vectors, cosine similarity, embedding model comparison |
| 03 | **Attention Mechanism** | QKV mechanism, multi-head attention, self vs cross-attention |
| 04 | **Transformer Architecture** | Complete decoder-only flow, residual connections, layer norm |
| 05 | **LLM Types & Models** | Model landscape 2025, MoE architecture, choosing the right model |

### 🔵 Phase 2 — Core Skills (Chapters 06–09)

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| 06 | **Prompt Engineering** | Zero-shot, Few-shot, CoT, Tree-of-Thought, Role prompting |
| 07 | **Sampling & Generation** | Temperature, Top-P, Top-K, stopping criteria |
| 08 | **Context & Memory** | Context windows, memory strategies, managing conversation history |
| 09 | **Function / Tool Calling** | How AI calls APIs, tool definitions, integration patterns |

### 🔴 Phase 3 — Advanced (Chapters 10–14)

| Chapter | Topic | Key Concepts |
|---------|-------|-------------|
| 10 | **Fine-Tuning & LoRA** | LoRA math, QLoRA, fine-tuning vs RAG decision tree |
| 11 | **RLHF, DPO & Alignment** | How ChatGPT learned to be helpful, DPO simplification |
| 12 | **AI Agents & ReAct** | ReAct pattern, LangGraph, CrewAI, multi-agent systems |
| 13 | **Local Models & Quantization** | Ollama, GGUF, INT4 quantization, running LLMs free |
| 14 | **Evaluation & LLMOps** | RAGAS metrics, BLEU/ROUGE/Perplexity, production checklist |

### 🏆 Mastery Section
- **30 Interview Questions** with professional-grade answers (expandable)
- **Complete Glossary** — every term defined in one place (30+ terms)

---

## 🚀 How to Run Locally

### Option 1: Python HTTP Server (Recommended for Content Loading)
```bash
# Clone the repo
git clone https://github.com/Sharjeel-Saleem-06/Learn_LLM_foundations_from_Basics.git
cd Learn_LLM_foundations_from_Basics

# Start server
python3 -m http.server 8080

# Open: http://localhost:8080
```

### Option 2: VS Code Live Server
1. Open folder in VS Code
2. Install "Live Server" extension
3. Right-click `index.html` → Open with Live Server

> **Note:** Must use a local server (not `file://`) so the browser can fetch `content2.html` for the advanced chapters due to CORS policy.

---

## 🏗️ Project Structure

```
LLM_Foundations/
├── index.html        # Main page — hero + Chapters 01-05 + sidebar navigation
├── content2.html     # Advanced chapters 06-14 + Interview Prep + Glossary
├── styles.css        # Complete design system — dark theme, all components
├── app.js            # Scroll reveal, reading progress, sidebar, dynamic loading
└── README.md         # This file
```

### Architecture Decisions

| Decision | Why |
|----------|-----|
| **Split into 2 HTML files** | Keep initial load fast; chapters 1-5 load instantly, advanced content loads asynchronously |
| **No framework** | Pure HTML/CSS/JS — zero build step, instantly deployable to Netlify/GitHub Pages |
| **JetBrains Mono for code** | Industry-standard coding font for maximum readability of code examples |
| **Intersection Observer** | Efficient scroll animations without layout thrashing |
| **Python code examples** | Most practical language for AI/ML, maximum library availability |

---

## 🎨 Design Features

- **Dark theme** with indigo/purple/pink gradient accents
- **Reading progress bar** in the fixed header
- **Sticky sidebar** with active section highlighting
- **Scroll-reveal animations** using Intersection Observer
- **Collapsible Q&A** for interview prep (expand/collapse)
- **Code syntax highlighting** with custom CSS (no library needed)
- **Responsive** — mobile sidebar with toggle button
- **ASCII diagrams** for architecture visualization
- **Data tables** with hover states throughout

---

## 📋 Key Concepts Explained (Quick Reference)

### Why LLMs Need Tokenization
LLMs don't process letters — they process **tokens** (subword units mapped to integers). BPE iteratively merges the most frequent character pairs until reaching a target vocabulary size (~50K-100K tokens). This handles any word including new words by decomposing them into known subwords.

### Why Attention Changed Everything
Before Transformers (2017), RNNs processed text sequentially left-to-right, causing "forgetting" of early context. **Attention** lets every token directly relate to every other token simultaneously, enabling perfect long-range dependencies regardless of distance. O(n²) complexity with sequence length.

### LoRA in One Sentence
Instead of training all billions of parameters, LoRA freezes the original weights and adds tiny trainable matrices (typically 0.1% of total params) that capture task-specific updates — enabling fine-tuning on consumer hardware.

### The ReAct Agent Loop
```
User Goal → Think → Act (call tool) → Observe (see result) → Think → Act → ... → Final Answer
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|------------|---------|
| HTML5 semantic | Structure & content |
| Vanilla CSS3 | Styling, animations, responsive layout |
| JavaScript (ES6+) | Interactivity, dynamic loading, scroll effects |
| Google Fonts (Inter + JetBrains Mono) | Typography |
| Python (code examples) | All practical code examples |
| Groq API (examples) | Free LLM API for examples |
| Sentence Transformers (examples) | Free embeddings |

---

## 🔗 Related Learning Resources

- **[RAG: Zero to Hero](https://github.com/Sharjeel-Saleem-06/Learn_RAG_from_Basics)** — The companion project covering Retrieval-Augmented Generation
- [Groq API](https://console.groq.com) — Free LLM API to run the code examples
- [HuggingFace](https://huggingface.co) — Access to open-source models
- [Ollama](https://ollama.ai) — Run models locally for free
- [RAGAS](https://docs.ragas.io) — RAG evaluation framework

---

## 📈 Learning Path

```
1. Start here (LLM Foundations) → understand every concept
2. Move to RAG: Zero to Hero → apply concepts to retrieval systems
3. Build real projects using Claude/Cursor as co-pilot
4. Study interview questions in this repo → ace AI Engineer interviews
```

---

## 🤝 Contributing

This is a personal learning project. If you find errors or want to improve explanations, feel free to open an issue or PR.

---

*Built with ❤️ for mastering Generative AI as a software engineer*
