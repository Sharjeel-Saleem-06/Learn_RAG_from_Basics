"""
============================================
FILE 7 OF 8: STREAMLIT_APP.PY — Interactive Chat UI
============================================

📝 PURPOSE:
    This creates a beautiful, interactive web interface for your RAG chatbot.
    Users can:
    - Upload PDF/text documents
    - Chat with their documents
    - See source citations
    - View system statistics

🎓 CONCEPTS COVERED:
    1. Streamlit — Python framework for building web apps quickly
    2. Session State — Maintaining data between page interactions
    3. File Upload Widgets — Handling user file uploads
    4. Chat Interface — Building a ChatGPT-like experience

🔗 HOW IT CONNECTS:
    rag_pipeline.py → streamlit_app.py (directly calls the pipeline)
    
    Unlike app.py (which creates an API), this Streamlit app
    directly imports and uses the RAGPipeline class.

============================================
"""

import streamlit as st
import os
import time

# ============================================
# PAGE CONFIGURATION
# ============================================
# Must be the FIRST Streamlit command
# ============================================
st.set_page_config(
    page_title="📚 RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================
# CUSTOM CSS FOR BEAUTIFUL UI
# ============================================
st.markdown("""
<style>
    /* Main background */
    .main {
        background-color: #0e1117;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 12px;
        margin-bottom: 8px;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #1a1a2e;
    }
    
    /* Custom header */
    .custom-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 12px;
        margin-bottom: 20px;
        text-align: center;
    }
    
    .custom-header h1 {
        color: white;
        margin: 0;
        font-size: 2em;
    }
    
    .custom-header p {
        color: #e0e0e0;
        margin: 5px 0 0 0;
    }
    
    /* Stats cards */
    .stat-card {
        background: #1a1a2e;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
    }
    
    /* Source citation */
    .source-chip {
        display: inline-block;
        background: #2d2d44;
        padding: 4px 12px;
        border-radius: 20px;
        margin: 2px;
        font-size: 0.85em;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# INITIALIZE SESSION STATE
# ============================================
# Streamlit reruns the ENTIRE script on every interaction.
# Session state persists data between reruns.
#
# 💡 Without session state:
# - Every button click resets everything
# - Chat history would be lost
# - Pipeline would reinitialize on every action
# ============================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
    st.session_state.initialized = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_uploaded" not in st.session_state:
    st.session_state.documents_uploaded = 0


# ============================================
# INITIALIZE PIPELINE (CACHED)
# ============================================
@st.cache_resource(show_spinner=False)
def init_pipeline():
    """
    Initialize the RAG pipeline.
    
    @st.cache_resource ensures this only runs ONCE,
    even if the page refreshes. The pipeline object is cached.
    """
    try:
        from rag_pipeline import RAGPipeline
        return RAGPipeline()
    except Exception as e:
        st.error(f"❌ Failed to initialize pipeline: {e}")
        return None


# ============================================
# SIDEBAR — Document Upload & Settings
# ============================================
with st.sidebar:
    st.markdown("## 📁 Document Management")
    
    # Initialize button
    if not st.session_state.initialized:
        if st.button("🚀 Initialize RAG System", type="primary", use_container_width=True):
            with st.spinner("Loading models and setting up... (first time may take a minute)"):
                pipeline = init_pipeline()
                if pipeline:
                    st.session_state.pipeline = pipeline
                    st.session_state.initialized = True
                    st.success("✅ RAG System ready!")
                    st.rerun()
    else:
        st.success("✅ System is running")
    
    st.divider()
    
    # File Upload
    st.markdown("### 📤 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, TXT, or DOCX files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files and st.session_state.initialized:
        if st.button("📥 Process Documents", type="primary", use_container_width=True):
            for uploaded_file in uploaded_files:
                # Save file temporarily
                temp_dir = "./uploaded_docs"
                os.makedirs(temp_dir, exist_ok=True)
                file_path = os.path.join(temp_dir, uploaded_file.name)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest into RAG pipeline
                with st.spinner(f"Processing {uploaded_file.name}..."):
                    try:
                        stats = st.session_state.pipeline.ingest_file(file_path)
                        st.success(
                            f"✅ {uploaded_file.name}\n"
                            f"   {stats['chunks_created']} chunks created"
                        )
                        st.session_state.documents_uploaded += 1
                    except Exception as e:
                        st.error(f"❌ Error: {e}")
    
    st.divider()
    
    # Database Stats
    st.markdown("### 📊 System Stats")
    if st.session_state.initialized and st.session_state.pipeline:
        try:
            stats = st.session_state.pipeline.get_db_stats()
            col1, col2 = st.columns(2)
            col1.metric("📄 Chunks", stats.get("document_count", 0))
            col2.metric("📁 Files", st.session_state.documents_uploaded)
        except:
            st.info("No stats available yet")
    else:
        st.info("Initialize the system first")
    
    st.divider()
    
    # Settings
    st.markdown("### ⚙️ Settings")
    n_results = st.slider("Chunks to retrieve", 1, 10, 5)
    show_sources = st.checkbox("Show source citations", value=True)
    
    st.divider()
    
    # Reset
    if st.button("🗑️ Reset Database", type="secondary", use_container_width=True):
        if st.session_state.pipeline:
            st.session_state.pipeline.reset()
            st.session_state.messages = []
            st.session_state.documents_uploaded = 0
            st.success("Database reset!")
            st.rerun()


# ============================================
# MAIN AREA — Chat Interface
# ============================================

# Header
st.markdown("""
<div class="custom-header">
    <h1>🤖 RAG Chatbot</h1>
    <p>Upload documents and chat with them using AI!</p>
</div>
""", unsafe_allow_html=True)

# Check system status
if not st.session_state.initialized:
    st.info(
        "👋 Welcome! Click **'🚀 Initialize RAG System'** in the sidebar to get started.\n\n"
        "**What this app does:**\n"
        "1. Upload your PDF/Text/Word documents\n"
        "2. The AI reads, chunks, and embeds them\n"
        "3. Ask any question about your documents\n"
        "4. Get AI-powered answers with source citations!\n\n"
        "**Requirements:**\n"
        "- A `GROQ_API_KEY` in your `.env` file (free from [Groq Console](https://console.groq.com/keys))"
    )
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message and show_sources:
                with st.expander("📄 Source Citations"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source.get('source', 'Unknown')}** "
                            f"(Relevance: {source.get('relevance', 0):.2f})"
                        )
                        st.caption(source.get('preview', ''))
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Check if documents are uploaded
        stats = st.session_state.pipeline.get_db_stats()
        if stats.get("document_count", 0) == 0:
            st.warning("⚠️ No documents uploaded yet! Please upload documents in the sidebar first.")
        else:
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🤔 Thinking..."):
                    try:
                        response = st.session_state.pipeline.query(
                            question=prompt,
                            n_results=n_results
                        )
                        
                        # Display answer
                        st.markdown(response.answer)
                        
                        # Display timing
                        st.caption(
                            f"⏱️ Retrieval: {response.retrieval_time}s | "
                            f"Generation: {response.generation_time}s | "
                            f"Total: {response.total_time}s | "
                            f"Model: {response.model_used}"
                        )
                        
                        # Format sources
                        sources = []
                        for s in response.sources:
                            sources.append({
                                "source": s["metadata"].get("source", "unknown"),
                                "relevance": s.get("relevance_score", 0),
                                "preview": s["text"][:200] + "..."
                            })
                        
                        # Show sources
                        if show_sources and sources:
                            with st.expander("📄 Source Citations"):
                                for source in sources:
                                    st.markdown(
                                        f"**{source['source']}** "
                                        f"(Relevance: {source['relevance']:.2f})"
                                    )
                                    st.caption(source['preview'])
                                    st.divider()
                        
                        # Save to chat history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": sources
                        })
                    
                    except Exception as e:
                        error_msg = f"❌ Error: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })


# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "Built with ❤️ using RAG | Powered by Groq & ChromaDB"
    "</div>",
    unsafe_allow_html=True
)
